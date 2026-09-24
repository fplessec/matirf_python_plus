"""
Running a long series of reconstructions safely: every result kept, any crash resumable.

A benchmark is hundreds of runs and hours of computation. Three things must hold whatever
happens — a crash, a kill, a power cut, a Ctrl-C:

    NOTHING DONE IS LOST      each run is saved the moment it finishes, in its own folder,
                              and its status is written atomically (write, fsync, rename):
                              a status file is either the old one or the new one, never half.
    RESTART = SAME COMMAND    a run is identified by a hash of its full configuration, so
                              relaunching the campaign skips what is done and resumes at the
                              exact run that was interrupted — even if the list was reordered.
    STOP ON DEMAND            a `STOP` file in the campaign folder (dropped by `matirf
                              benchmark stop`, or by hand) makes the loop exit cleanly before
                              the next run — a safe checkpoint regardless of how it was
                              launched. Ctrl-C on a foreground run does the same for the run
                              in flight (marked 'interrupted', redone on resume).
    A CRASH STAYS CONTAINED   each run executes in its own process, with a time limit. A
                              segfault, an out-of-memory kill or a hang ends that run as
                              'failed' or 'timeout'; the campaign itself survives.

And one thing specific to reconstruction: a run can finish "successfully" and still be
meaningless (core/diagnostics.py — collapsed at the interface, the same image on every
plane, f ~ g, f ~ a least-squares estimate). By default the campaign STOPS on such a run,
so it can be investigated before hours are spent on more of the same. Relaunching the same
command then retries it; `accept_rejected=True` keeps it as 'rejected' and moves on.

    campaign/
        journal.jsonl            one line per finished run, appended as it happens
        runs/<run_id>/
            status.json          running | done | rejected | failed | timeout | interrupted
            config.toml, f.<ext>, messages.txt, metrics.json, f_true.<ext>   (pipeline)

Usage:

    specs = [RunSpec("matirf", config, tags={"solver": "ADAM", "lambda": 0.1}), ...]
    run_campaign(specs, "benchmarks/results/main")          # stops on the first rejection
"""

import hashlib
import json
import multiprocessing
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

DONE, REJECTED, FAILED, TIMEOUT, INTERRUPTED, RUNNING = (
    "done", "rejected", "failed", "timeout", "interrupted", "running")
## statuses a relaunch does not redo (a rejected run is retried unless accepted)
FINISHED = {DONE}


@dataclass
class RunSpec:
    """One reconstruction: the problem, its full config, and labels for the report."""
    problem: str
    config: dict
    tags: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        """Stable across sessions and orderings: a hash of everything that defines the run."""
        payload = json.dumps({"problem": self.problem, "config": self.config},
                             sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]


class CampaignStopped(Exception):
    """A run was rejected by the realism check and the campaign was asked to stop on it."""

    def __init__(self, spec: RunSpec, record: dict, run_dir: Path):
        self.spec, self.record, self.run_dir = spec, record, run_dir
        super().__init__(
            f"run {spec.run_id} rejected — {record.get('diagnosis', '')}\n"
            f"    tags   : {spec.tags}\n"
            f"    folder : {run_dir}\n"
            f"Investigate it (the reconstruction and its log are in the folder). Then relaunch "
            f"the same command to retry it, or with accept_rejected to keep it and go on.")


# ── safe writes ───────────────────────────────────────────────────────────────

def write_atomically(path: Path, text: str) -> None:
    """Replace `path` with `text` so that a crash leaves either the old or the new file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_line(path: Path, record: dict) -> None:
    """Append one JSON line and force it to disk before going on."""
    with open(path, "a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_status(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ── one run, in whatever process executes it ──────────────────────────────────

def execute(problem_name: str, config: dict, run_dir: str) -> dict:
    """Run one reconstruction headless, save it into run_dir, and summarize it."""
    import importlib
    from pipeline import Pipeline

    problem = importlib.import_module(f"problems.{problem_name}").PROBLEM
    started = time.time()
    pipeline = Pipeline.create(problem, config)
    errors = []
    pipeline.on_error = errors.append
    pipeline.start()
    while pipeline.is_running:
        time.sleep(0.05)
    Pipeline.remove(pipeline)
    if errors or pipeline.result.f is None:
        raise RuntimeError("; ".join(map(str, errors)) or "the run produced no reconstruction")
    pipeline.save_results(run_dir)

    diagnosis = pipeline.result.diagnosis
    metrics = {key: value for key, value in (pipeline.result.metrics or {}).items()
               if isinstance(value, (int, float))}
    if isinstance((pipeline.result.metrics or {}).get("FSC"), dict):
        metrics["FSC"] = pipeline.result.metrics["FSC"].get("summary")
    metrics.update(_curated_metrics(pipeline, config))   # the benchmark's curated set
    noise = next((line for line in pipeline.result.messages.splitlines()
                  if line.startswith("Noise model")), "")
    import resource
    import sys
    ## peak resident memory of this (isolated) run — ru_maxrss is bytes on macOS, KiB on Linux
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mem_mb = round(maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)
    return {"seconds": round(time.time() - started, 2),
            "peak_mem_mb": peak_mem_mb,
            "metrics": metrics,
            "rejected": bool(diagnosis is not None and diagnosis.rejected),
            "findings": [finding.code for finding in (diagnosis.findings if diagnosis else [])],
            "diagnosis": diagnosis.summary() if diagnosis else "",
            "noise": noise}


def _curated_metrics(pipeline, config: dict) -> dict:
    """
    The benchmark's curated metrics (benchmarks/metrics.py), computed on this run:
    nmse and chi2_ratio always, depth_error_nm / stack_recovery for a 3D (MA-TIRF) result.
    `regularization_share` is NOT here — it needs the paired lambda=0 run, so phase B's
    analysis computes it across the sweep. Non-finite values (e.g. no stacked columns) become
    None so the record stays valid JSON.
    """
    import math

    from benchmarks import metrics as M
    from pipeline import noise_parameters

    f, truth, g = pipeline.result.f, pipeline.result.f_true, pipeline.result.g
    if f is None:
        return {}
    _, a, b, _ = noise_parameters(config, g)
    out = {"chi2_ratio": M.chi2_ratio(f, pipeline.prepared.operator, g, a, b)}
    if truth is not None:
        out["nmse"] = M.nmse(f, truth)
        out["psnr"] = M.psnr(f, truth)          # dB companion to nmse (same ranking)
        if f.dim() == 3:                        # depth metrics only mean something in 3D
            oper = config.get("oper-params", {})
            nz = float(oper.get("nz", f.shape[0])) or float(f.shape[0])
            dz = (float(oper.get("zN", nz)) - float(oper.get("z0", 0.0))) / nz
            out["depth_error_nm"] = M.depth_error_nm(f, truth, dz)
            out["stack_recovery"] = M.stack_recovery(f, truth)
        elif f.dim() == 2:                       # SSIM is reliable only on the 2D deconv image
            out["ssim"] = M.ssim(f, truth)
    return {k: (v if isinstance(v, float) and math.isfinite(v) else None) for k, v in out.items()}


def _child(problem_name, config, run_dir, queue):
    try:
        queue.put(("ok", execute(problem_name, config, run_dir)))
    except BaseException:
        queue.put(("error", traceback.format_exc()))


def _execute_isolated(spec: RunSpec, run_dir: Path, timeout: float):
    """(status, record) — the run in a fresh process, killed if it exceeds `timeout`."""
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_child, args=(spec.problem, spec.config, str(run_dir), queue))
    process.start()
    deadline = time.time() + timeout
    outcome = None
    try:
        while outcome is None and time.time() < deadline:
            try:
                outcome = queue.get(timeout=0.5)
            except Exception:                       # queue.Empty: still running
                if not process.is_alive():
                    break
    except KeyboardInterrupt:
        process.terminate()
        process.join()
        raise
    if outcome is None:
        alive = process.is_alive()
        process.terminate()
        process.join()
        if alive:
            return TIMEOUT, {"error": f"exceeded {timeout:.0f} s"}
        return FAILED, {"error": f"the process died (exit code {process.exitcode})"}
    process.join()
    kind, payload = outcome
    if kind == "error":
        return FAILED, {"error": payload}
    return (REJECTED if payload["rejected"] else DONE), payload


def _execute_inline(spec: RunSpec, run_dir: Path):
    try:
        payload = execute(spec.problem, spec.config, str(run_dir))
    except Exception:
        return FAILED, {"error": traceback.format_exc()}
    return (REJECTED if payload["rejected"] else DONE), payload


# ── the campaign ──────────────────────────────────────────────────────────────

def run_campaign(specs, directory, *, stop_on_reject: bool = True, accept_rejected: bool = False,
                 isolate: bool = True, timeout: float = 3600.0, log=print) -> dict:
    """
    Execute every spec not already done, saving as it goes. Returns {status: count}.

    >> stop_on_reject  : raise CampaignStopped at the first run the realism check rejects
    >> accept_rejected : treat an earlier 'rejected' run as settled instead of retrying it
    >> isolate         : one process per run (crash-proof); False runs inline (tests, debug)
    >> timeout         : seconds after which an isolated run is killed and marked 'timeout'
    """
    directory = Path(directory)
    (directory / "runs").mkdir(parents=True, exist_ok=True)
    journal = directory / "journal.jsonl"
    ## the STOP sentinel: `matirf benchmark stop` (or anyone) drops this file, and the loop
    ## exits cleanly BEFORE the next run — a safe checkpoint whatever the launch mode
    ## (foreground, background, grid). Cleared here so relaunching resumes.
    stop_flag = directory / "STOP"
    stop_flag.unlink(missing_ok=True)
    settled = FINISHED | ({REJECTED} if accept_rejected else set())
    counts = {}
    ids = [spec.run_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("two specs have the same configuration (same run_id)")

    todo = [spec for spec in specs
            if read_status(directory / "runs" / spec.run_id).get("status") not in settled]
    log(f"Campaign {directory}: {len(specs)} runs, {len(specs) - len(todo)} already settled, "
        f"{len(todo)} to go.")

    for position, spec in enumerate(todo, start=1):
        if stop_flag.exists():
            log("Stop requested (STOP file) — exiting cleanly; relaunch to resume here.")
            break
        run_dir = directory / "runs" / spec.run_id
        run_dir.mkdir(exist_ok=True)
        write_atomically(run_dir / "status.json", json.dumps(
            {"status": RUNNING, "started": time.time(), "problem": spec.problem,
             "tags": spec.tags, "config": spec.config}, default=str, indent=1))
        log(f"[{position}/{len(todo)}] {spec.run_id} {spec.tags}")
        try:
            status, record = (_execute_isolated(spec, run_dir, timeout) if isolate
                              else _execute_inline(spec, run_dir))
        except KeyboardInterrupt:
            write_atomically(run_dir / "status.json", json.dumps(
                {"status": INTERRUPTED, "problem": spec.problem, "tags": spec.tags,
                 "config": spec.config}, default=str, indent=1))
            log("Interrupted — relaunch the same command to resume at this run.")
            raise

        entry = {"run_id": spec.run_id, "status": status, "problem": spec.problem,
                 "tags": spec.tags, "finished": time.time(), **record}
        write_atomically(run_dir / "status.json", json.dumps(
            {**entry, "config": spec.config}, default=str, indent=1))
        append_line(journal, entry)
        counts[status] = counts.get(status, 0) + 1
        log(f"    -> {status}" + (f" ({record.get('diagnosis', record.get('error', ''))[:160]})"
                                  if status != DONE else f" in {record.get('seconds')} s"))
        if status == REJECTED and stop_on_reject:
            raise CampaignStopped(spec, entry, run_dir)
    return counts


def campaign_records(directory) -> list:
    """The latest status of every run in a campaign, read from the run folders."""
    runs = Path(directory) / "runs"
    return [read_status(run_dir) for run_dir in sorted(runs.iterdir())
            if (run_dir / "status.json").exists()] if runs.exists() else []
