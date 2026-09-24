"""
benchmarks/analyze.py — turn a finished (or partial) campaign into the report's committed outputs.

Reads the durable per-run log ``benchmarks/results/<dir>/journal.jsonl`` and writes, INTO the
LaTeX report repo (``../matirf_benchmark_report`` by default, next to this one):

    results/summary/*.md      the atlas tables — cross-comparison, effective ranges, rejects+cost.
                              THE anti-loss synthesis: committed to the report repo, so the
                              conclusions survive even if benchmarks/results/ is wiped again.
    results/<figure_id>/      for the best run of each (solver x dataset x noise): config.toml +
                              metrics.json (reproducible), and — for the showcase — a PNG of the
                              depth map + profiles (matirf) exported from the GUI viewers.

    python -m benchmarks analyze --minimal
    python -m benchmarks analyze                # the full atlas (results/main)
    python -m benchmarks analyze --report-dir /path/to/report

It tolerates a partial campaign: it reports only the cells that have finished, so it can be run
while the run is still going. Re-running overwrites the outputs from the current records.
"""
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]                     # matirf_python_plus
DEFAULT_REPORT_DIR = REPO.parent / "matirf_benchmark_report"   # the sibling LaTeX repo

## tags that identify the run rather than a swept parameter
STRUCTURAL_TAGS = {"phase", "problem", "solver", "dataset", "noise", "role"}
## the curated metrics, in report order (names from core.metrics + chi2 from the runner)
MATIRF_METRICS = ["NMSE", "PSNR", "Depth Error (nm)", "Stack Recovery", "chi2_ratio"]
DECONV_METRICS = ["NMSE", "PSNR", "SSIM", "chi2_ratio"]
NOISE_ORDER = {0.0: 0, 0.02: 1, 0.05: 2, "native": 3}


# ── loading ─────────────────────────────────────────────────────────────────

def load_records(campaign_dir: Path) -> list:
    """Every finished run, from journal.jsonl (deduplicated by run_id, last line wins)."""
    journal = campaign_dir / "journal.jsonl"
    records = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[rec.get("run_id", id(rec))] = rec
    else:                                                       # fall back to the per-run status
        for status in (campaign_dir / "runs").glob("*/status.json"):
            rec = json.loads(status.read_text())
            records[rec.get("run_id", status.parent.name)] = rec
    return list(records.values())


# ── small helpers ─────────────────────────────────────────────────────────────

def _params(rec: dict) -> str:
    """The swept parameters of a run, as a compact 'k=v' string (drops the structural tags)."""
    tags = rec.get("tags", {})
    parts = [f"{k}={_fmt(v)}" for k, v in tags.items() if k not in STRUCTURAL_TAGS]
    return ", ".join(parts) if parts else "—"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _metric(rec: dict, name: str):
    v = (rec.get("metrics") or {}).get(name)
    return v if isinstance(v, (int, float)) else None


def _cell(rec: dict):
    t = rec.get("tags", {})
    return (t.get("problem"), t.get("dataset"), t.get("noise"))


def _noise_key(noise):
    return NOISE_ORDER.get(noise, 9)


def _done(rec: dict) -> bool:
    return rec.get("status") == "done" and not rec.get("rejected", False)


def _best(records: list, key: str = "NMSE", target=None):
    """The record optimizing `key` — smallest NMSE, or closest to `target` (chi2 → 1)."""
    scored = [(r, _metric(r, key)) for r in records]
    scored = [(r, v) for r, v in scored if v is not None]
    if not scored:
        return None
    if target is None:
        return min(scored, key=lambda rv: rv[1])[0]
    return min(scored, key=lambda rv: abs(rv[1] - target))[0]


def _fmt_metric(rec: dict, name: str) -> str:
    v = _metric(rec, name)
    if v is None:
        return "—"
    if name == "Depth Error (nm)":
        return f"{v:.0f}"
    if name in ("PSNR",):
        return f"{v:.1f}"
    return f"{v:.3g}"


# ── the tables (Markdown) ─────────────────────────────────────────────────────

def cross_comparison(records: list, problem: str) -> str:
    """Per (dataset, noise): which solver wins (best NMSE), with its curated metrics."""
    metrics = MATIRF_METRICS if problem == "matirf" else DECONV_METRICS
    usable = [r for r in records if r.get("tags", {}).get("problem") == problem
              and _done(r) and r.get("tags", {}).get("noise") != "native"]
    cells = sorted({_cell(r) for r in usable}, key=lambda c: (str(c[1]), _noise_key(c[2])))
    header = "| dataset | noise | best (by NMSE) | params | " + " | ".join(metrics) + " |"
    sep = "|" + "---|" * (4 + len(metrics))
    lines = [f"## Cross-comparison — {problem}: which solver wins, and where", "",
             header, sep]
    for problem_, dataset, noise in cells:
        group = [r for r in usable if _cell(r) == (problem_, dataset, noise)]
        winner = _best(group, "NMSE")
        if winner is None:
            continue
        cols = " | ".join(_fmt_metric(winner, m) for m in metrics)
        lines.append(f"| {dataset} | {_fmt(noise)} | **{winner['tags']['solver']}** | "
                     f"{_params(winner)} | {cols} |")
    return "\n".join(lines) + "\n"


def effective_ranges(records: list, problem: str = "matirf") -> str:
    """Per solver, per noise: the parameter setting that won (the effective best), by NMSE."""
    usable = [r for r in records if r.get("tags", {}).get("problem") == problem
              and _done(r) and r.get("tags", {}).get("noise") != "native"]
    solvers = sorted({r["tags"]["solver"] for r in usable})
    lines = [f"## Effective parameter ranges — {problem} (the atlas)", "",
             "The setting that minimised NMSE at each noise level; compare to the predicted "
             "ranges in `docs/algorithms/`.", ""]
    for solver in solvers:
        runs = [r for r in usable if r["tags"]["solver"] == solver]
        noises = sorted({r["tags"]["noise"] for r in runs}, key=_noise_key)
        lines += [f"### {solver}", "", "| noise | best params | NMSE | depth err (nm) |", "|---|---|---|---|"]
        for noise in noises:
            group = [r for r in runs if r["tags"]["noise"] == noise]
            best = _best(group, "NMSE")
            if best is None:
                continue
            lines.append(f"| {_fmt(noise)} | {_params(best)} | {_fmt_metric(best, 'NMSE')} | "
                         f"{_fmt_metric(best, 'Depth Error (nm)')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def rejects_and_cost(records: list) -> str:
    """Per solver: rejection rate, dominant findings, runtime and peak memory, failures."""
    lines = ["## Rejections, failures and cost", "",
             "| solver | runs | rejected | failed/timeout | median s | max mem (MB) | top findings |",
             "|---|---|---|---|---|---|---|"]
    solvers = sorted({r.get("tags", {}).get("solver", "?") for r in records})
    for solver in solvers:
        runs = [r for r in records if r.get("tags", {}).get("solver") == solver]
        n = len(runs)
        rejected = sum(1 for r in runs if r.get("rejected"))
        broken = sum(1 for r in runs if r.get("status") in ("failed", "timeout"))
        times = [r["seconds"] for r in runs if isinstance(r.get("seconds"), (int, float))]
        mems = [r["peak_mem_mb"] for r in runs if isinstance(r.get("peak_mem_mb"), (int, float))]
        findings = {}
        for r in runs:
            for f in r.get("findings", []):
                findings[f] = findings.get(f, 0) + 1
        top = ", ".join(f"{k}×{v}" for k, v in sorted(findings.items(), key=lambda kv: -kv[1])[:3])
        median = sorted(times)[len(times) // 2] if times else float("nan")
        lines.append(f"| {solver} | {n} | {rejected} ({100*rejected//max(n,1)}%) | {broken} | "
                     f"{median:.1f} | {max(mems) if mems else float('nan'):.0f} | {top or '—'} |")
    return "\n".join(lines) + "\n"


# ── the showcase (config + metrics + PNG per winning cell) ────────────────────

def _figure_id(rec: dict) -> str:
    t = rec["tags"]
    noise = t.get("noise")
    tag = "native" if noise == "native" else f"noise{str(noise).replace('.', '')}"
    return f"{t['problem']}_{t['dataset']}_{t['solver']}_{tag}"


def export_showcase(records: list, campaign_dir: Path, report_dir: Path) -> list:
    """For the best run of each (solver x dataset x noise): copy config + metrics (+ PNG)."""
    runs_dir = campaign_dir / "runs"
    exported = []
    ## the winner of every cell, per solver — so each solver appears in the showcase figures
    usable = [r for r in records if _done(r)]
    cells = {}
    for r in usable:
        key = (r["tags"].get("problem"), r["tags"].get("dataset"),
               r["tags"].get("noise"), r["tags"].get("solver"))
        cells.setdefault(key, []).append(r)
    for key, group in cells.items():
        target = 1.0 if key[2] == "native" else None      # esoubies: chi2 -> 1; else min NMSE
        best = _best(group, "chi2_ratio" if key[2] == "native" else "NMSE", target=target)
        if best is None:
            continue
        run_dir = runs_dir / best["run_id"]
        if not run_dir.exists():
            continue
        out = report_dir / "results" / _figure_id(best)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("config.toml",):
            src = run_dir / name
            if src.exists():
                shutil.copy2(src, out / name)
        (out / "metrics.json").write_text(json.dumps(best.get("metrics", {}), indent=2))
        exported.append((best, run_dir, out))
    return exported


def render_pngs(exported: list) -> int:
    """Best-effort headless PNG of the depth map + profiles for each 3D (matirf) showcase run.

    Uses the GUI viewers under an offscreen Qt; any failure is logged and skipped so the tables
    and the config/metrics exports are never held hostage to a rendering glitch.
    """
    matirf = [(rec, run_dir, out) for rec, run_dir, out in exported
              if rec["tags"].get("problem") == "matirf"]
    if not matirf:
        return 0
    try:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        from fileio import load_tif
        from gui.widgets.depth_map_viewer import DepthMapViewer
        from gui.widgets.profiles_viewer import ProfilesViewer
        from gui.widgets.figure_export import export_depth_and_profiles
    except Exception as error:                              # pragma: no cover - environment
        print(f"  (PNG export skipped — {type(error).__name__}: {error})")
        return 0

    app = QApplication.instance() or QApplication([])
    count = 0
    for rec, run_dir, out in matirf:
        tif = run_dir / "f.TIF"
        if not tif.exists():
            continue
        oper = rec.get("config", {}).get("oper-params", {})
        z0, zN = float(oper.get("z0", 0.0)), float(oper.get("zN", 300.0))
        try:
            image = load_tif(tif)
            depth = DepthMapViewer(image, z0, zN, title=rec["tags"]["solver"])
            profiles = ProfilesViewer(image, z0, zN)
            export_depth_and_profiles(depth, profiles, str(out / "f.png"))
            depth.deleteLater(); profiles.deleteLater()
            count += 1
        except Exception as error:                          # pragma: no cover - rendering
            print(f"  (PNG failed for {out.name} — {type(error).__name__}: {error})")
    _ = app
    return count


# ── orchestration ─────────────────────────────────────────────────────────────

def analyze(campaign_dir: Path, report_dir: Path, with_pngs: bool = True) -> None:
    records = load_records(campaign_dir)
    if not records:
        print(f"No records in {campaign_dir} (journal.jsonl / runs/). Nothing to analyze.")
        return
    print(f"Analyzing {len(records)} runs from {campaign_dir}")

    summary = report_dir / "results" / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "cross_comparison_matirf.md").write_text(cross_comparison(records, "matirf"))
    (summary / "cross_comparison_deconv.md").write_text(cross_comparison(records, "deconv"))
    (summary / "effective_ranges.md").write_text(effective_ranges(records, "matirf"))
    (summary / "rejects_and_cost.md").write_text(rejects_and_cost(records))
    print(f"  wrote 4 summary tables to {summary}")

    exported = export_showcase(records, campaign_dir, report_dir)
    print(f"  exported {len(exported)} showcase runs (config.toml + metrics.json)")
    if with_pngs:
        n = render_pngs(exported)
        print(f"  rendered {n} depth-map/profile PNGs")
