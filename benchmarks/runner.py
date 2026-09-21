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
    noise = next((line for line in pipeline.result.messages.splitlines()
                  if line.startswith("Noise model")), "")
    return {"seconds": round(time.time() - started, 2),
            "metrics": metrics,
            "rejected": bool(diagnosis is not None and diagnosis.rejected),
            "findings": [finding.code for finding in (diagnosis.findings if diagnosis else [])],
            "diagnosis": diagnosis.summary() if diagnosis else "",
            "noise": noise}


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
