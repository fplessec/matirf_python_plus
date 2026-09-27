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


## the report's algorithm chapters, in order (tag values → display used in the filenames)
SOLVER_ORDER = ["ADAM", "PPXA", "ADMM", "PNP", "ADMM-PnP", "MCMC"]


DENOISER_SOLVERS = ["PNP", "ADMM-PnP", "MCMC"]


def denoiser_comparison(records: list, solver: str) -> str:
    """For a denoiser-using solver, which denoiser wins on each structure. Per (dataset, noise),
    one row per denoiser (role='denoiser'), best NMSE bolded — the winner may depend on the object."""
    runs = [r for r in records if r.get("tags", {}).get("solver") == solver
            and r.get("tags", {}).get("role") == "denoiser" and _done(r)]
    lines = [f"## {solver} — denoiser comparison (which denoiser wins per structure)", ""]
    if not runs:
        return "\n".join(lines + ["*(no completed runs yet)*", ""]) + "\n"
    for problem in ("matirf", "deconv"):
        here = [r for r in runs if r.get("tags", {}).get("problem") == problem]
        if not here:
            continue
        metrics = MATIRF_METRICS if problem == "matirf" else DECONV_METRICS
        header = "| structure | noise | denoiser | " + " | ".join(metrics) + " |"
        lines += [f"### {problem}", "", header, "|" + "---|" * (3 + len(metrics))]
        cells = sorted({_cell(r) for r in here}, key=lambda c: (_noise_key(c[2]), str(c[1])))
        for _, dataset, noise in cells:
            group = [r for r in here if _cell(r) == (problem, dataset, noise)]
            winner = _best(group, "NMSE")
            for r in sorted(group, key=lambda r: _metric(r, "NMSE") if _metric(r, "NMSE") is not None else 9):
                den = r["tags"].get("denoiser", "?")
                den = f"**{den}**" if r is winner else den
                cols = " | ".join(_fmt_metric(r, m) for m in metrics)
                lines.append(f"| {dataset} | {_fmt(noise)} | {den} | {cols} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def gallery(records: list, solver: str) -> str:
    """Per-algo gallery (raw LaTeX): the best reconstruction on each structure at noise 0.02, with
    the run's parameters in each subcaption so any panel is reproducible."""
    runs = [r for r in records if r.get("tags", {}).get("solver") == solver
            and r.get("tags", {}).get("problem") == "matirf"
            and r.get("tags", {}).get("noise") == 0.02 and _done(r)]
    panels = []
    for s in ("vesicles", "fibres", "cell"):
        best = _best([r for r in runs if r["tags"].get("dataset") == s], "NMSE")
        if best is not None:
            panels.append((s, _figure_id(best), _tex_escape(_params(best)), _fmt_metric(best, "NMSE")))
    if not panels:
        return ""
    ## full depth-map + yz/zx profiles, stacked one per row so the profiles stay legible
    out = [r"\begin{figure}[H]\centering"]
    for s, fid, params, nmse in panels:
        out.append(r"\begin{subfigure}{0.56\linewidth}\centering")
        out.append(r"\IfFileExists{figures/%s.png}{\includegraphics[width=\linewidth]{%s}}{}" % (fid, fid))
        out.append(r"\caption*{\footnotesize\texttt{%s} --- %s (NMSE %s)}" % (s, params, nmse))
        out.append(r"\end{subfigure}\par\smallskip")
    out.append(r"\caption{\textbf{%s} --- best reconstruction on each structure at noise 0.02 "
               r"(depth map + $yz/zx$ profiles; the parameters under each panel reproduce it).}" % solver)
    out.append(r"\end{figure}")
    return "\n".join(out) + "\n"


def truth_gallery(records: list, campaign_dir: Path, report_dir: Path) -> str:
    """Render each structure's ground truth (depth map + profiles) and emit a LaTeX figure of the
    three, for the start of the report. f_true.TIF is saved in every run folder."""
    try:
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        from fileio import load_tif
        from gui.widgets.depth_map_viewer import DepthMapViewer
        from gui.widgets.figure_export import export_depth_and_profiles
        from gui.widgets.profiles_viewer import ProfilesViewer
    except Exception as error:                                  # pragma: no cover - environment
        print(f"  (truth render skipped — {type(error).__name__}: {error})")
        return ""
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    panels = []
    for s in ("vesicles", "fibres", "cell"):
        run = next((r for r in records if r.get("tags", {}).get("dataset") == s
                    and (campaign_dir / "runs" / r.get("run_id", "") / "f_true.TIF").exists()), None)
        if run is None:
            continue
        tif = campaign_dir / "runs" / run["run_id"] / "f_true.TIF"
        oper = run.get("config", {}).get("oper-params", {})
        z0, zN = float(oper.get("z0", 0.0)), float(oper.get("zN", 300.0))
        try:
            image = load_tif(tif)
            depth = DepthMapViewer(image, z0, zN, title=s)
            profiles = ProfilesViewer(image, z0, zN)
            export_depth_and_profiles(depth, profiles, str(figures / f"truth_{s}.png"))
            depth.deleteLater(); profiles.deleteLater()
            panels.append(s)
        except Exception as error:                              # pragma: no cover - rendering
            print(f"  (truth PNG failed for {s} — {type(error).__name__}: {error})")
    _ = app
    if not panels:
        return ""
    out = [r"\begin{figure}[H]\centering"]
    for s in panels:
        out.append(r"\begin{subfigure}{0.56\linewidth}\centering")
        out.append(r"\IfFileExists{figures/truth_%s.png}{\includegraphics[width=\linewidth]{truth_%s}}{}" % (s, s))
        out.append(r"\caption*{\footnotesize\texttt{%s}}" % s)
        out.append(r"\end{subfigure}\par\smallskip")
    out.append(r"\caption{The three ground-truth structures --- \texttt{vesicles} (point-like), "
               r"\texttt{fibres} (filaments), \texttt{cell} (continuous membrane) --- as depth map "
               r"+ $yz/zx$ profiles.}")
    out.append(r"\label{fig:truths}")
    out.append(r"\end{figure}")
    return "\n".join(out) + "\n"


def denoiser_winners(records: list) -> str:
    """ONE compact table: for each (structure, noise) the winning denoiser (best NMSE) of each
    plug-and-play solver. Directly answers 'which denoiser', replacing the three per-algo tables."""
    solvers = [("PNP", "PnP"), ("ADMM-PnP", "ADMM-PnP"), ("MCMC", "MCMC")]
    runs = [r for r in records if r.get("tags", {}).get("role") == "denoiser"
            and r.get("tags", {}).get("problem") == "matirf" and _done(r)]
    lines = ["## Denoiser winner per structure and algorithm (best NMSE)", ""]
    if not runs:
        return "\n".join(lines + ["*(no completed runs yet)*", ""]) + "\n"
    header = "| structure | noise | " + " | ".join(disp for _, disp in solvers) + " |"
    lines += [header, "|" + "---|" * (2 + len(solvers))]
    cells = sorted({(r["tags"]["dataset"], r["tags"]["noise"]) for r in runs},
                   key=lambda c: (_noise_key(c[1]), str(c[0])))
    for dataset, noise in cells:
        row = [str(dataset), _fmt(noise)]
        for tag, _ in solvers:
            grp = [r for r in runs if r["tags"]["solver"] == tag
                   and r["tags"]["dataset"] == dataset and r["tags"]["noise"] == noise]
            best = _best(grp, "NMSE")
            row.append(f"{best['tags'].get('denoiser')} ({_fmt_metric(best, 'NMSE')})" if best else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def adam_shape(records: list) -> str:
    """Adam's λ-shape at the reference condition (noise 0.02): NMSE and depth error vs λ, for the
    sparse (L1) and the smooth (Tikhonov) prior — the illustration in the Adam chapter."""
    runs = [r for r in records if r.get("tags", {}).get("solver") == "ADAM"
            and r.get("tags", {}).get("role") == "shape" and _done(r)
            and r.get("tags", {}).get("reg") in ("l1", "tikhonov")]
    lines = ["## Adam λ-shape — reference structure, noise 0.02", "",
             "| prior | lambda | NMSE | depth err (nm) |", "|---|---|---|---|"]
    for reg in ("l1", "tikhonov"):
        group = sorted((r for r in runs if r["tags"].get("reg") == reg),
                       key=lambda r: r["tags"].get("lambda", 0))
        for r in group:
            lines.append(f"| {reg} | {_fmt(r['tags'].get('lambda'))} | "
                         f"{_fmt_metric(r, 'NMSE')} | {_fmt_metric(r, 'Depth Error (nm)')} |")
    return "\n".join(lines) + "\n"


def per_algorithm(records: list, solver: str) -> str:
    """One algorithm's `standard vs optimal` table — the core of its report chapter.

    For each (dataset, noise) this solver was run at, the STANDARD (role='standard', the default
    config) row and, when different, the OPTIMAL row (the best NMSE — or, on the real esoubies
    data with no truth, the chi2_ratio closest to 1). Params and curated metrics come straight
    from the run. Split per problem (MA-TIRF / deconv) so the metric columns match.
    """
    runs = [r for r in records if r.get("tags", {}).get("solver") == solver
            and (_done(r) or r.get("tags", {}).get("noise") == "native")]
    lines = [f"## {solver} — standard vs optimal", ""]
    if not runs:
        return "\n".join(lines + ["*(no completed runs yet)*", ""]) + "\n"

    for problem in ("matirf", "deconv"):
        here = [r for r in runs if r.get("tags", {}).get("problem") == problem]
        if not here:
            continue
        metrics = MATIRF_METRICS if problem == "matirf" else DECONV_METRICS
        ## the standard (default) config and the optimal (best) one; the params column tells them
        ## apart (a 'use' label is redundant with it). Rows sorted by noise, esoubies (native) last.
        header = "| dataset | noise | params | " + " | ".join(metrics) + " |"
        sep = "|" + "---|" * (3 + len(metrics))
        lines += [f"### {problem}", "", header, sep]
        cells = sorted({_cell(r) for r in here}, key=lambda c: (_noise_key(c[2]), str(c[1])))
        for _, dataset, noise in cells:
            group = [r for r in here if _cell(r) == (problem, dataset, noise)]
            native = noise == "native"
            standard = next((r for r in group if r.get("tags", {}).get("role") == "standard"), None)
            optimal = _best(group, "chi2_ratio" if native else "NMSE", target=1.0 if native else None)
            emitted = []
            for rec in (standard, optimal):
                if rec is None or rec["run_id"] in emitted:
                    continue
                emitted.append(rec["run_id"])
                cols = " | ".join(_fmt_metric(rec, m) for m in metrics)
                lines.append(f"| {dataset} | {_fmt(noise)} | {_params(rec)} | {cols} |")
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
        from gui.widgets.figure_export import (
            export_depth_and_profiles,
            export_depth_only,
        )
        from gui.widgets.profiles_viewer import ProfilesViewer
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
        ## write the figure straight into figures/ (referenced by the report); the reconstruction
        ## itself is NOT duplicated into results/ — that dir keeps only config.toml + metrics.json.
        figures = out.parents[1] / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        try:
            image = load_tif(tif)
            depth = DepthMapViewer(image, z0, zN, title=rec["tags"]["solver"])
            profiles = ProfilesViewer(image, z0, zN)
            export_depth_and_profiles(depth, profiles, str(figures / f"{out.name}.png"))
            depth.deleteLater(); profiles.deleteLater()
            count += 1
        except Exception as error:                          # pragma: no cover - rendering
            print(f"  (PNG failed for {out.name} — {type(error).__name__}: {error})")
    _ = app
    return count


# ── Markdown → LaTeX (pipe tables) ────────────────────────────────────────────

def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash "), ("_", r"\_"), ("%", r"\%"),
                 ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def _tex_cells(cells: list) -> list:
    out = []
    for c in cells:
        c = c.strip()
        bold = c.startswith("**") and c.endswith("**")
        c = c[2:-2] if bold else c
        c = _tex_escape(c)
        out.append(f"\\textbf{{{c}}}" if bold else c)
    return out


def md_to_latex(md: str) -> str:
    """Turn the pipe tables (and `### ` labels) of a summary .md into booktabs tabulars.

    Prose and `##` headings are dropped — the report section carries those; only the tables are
    \\input. Keeps this dependency-free (no pandoc)."""
    lines, tex, i = md.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### "):
            tex.append(r"\paragraph{" + _tex_escape(line[4:].strip()) + "}")
            i += 1
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            header = block[0].strip("|").split("|")
            ## adjustbox shrinks a too-wide table to the text width (and leaves narrow ones alone)
            tex.append(r"\begin{center}\begin{adjustbox}{max width=\linewidth}")
            tex.append(r"\begin{tabular}{" + "l" * len(header) + "}")
            tex.append(r"\toprule")
            tex.append(" & ".join(_tex_cells(header)) + r" \\")
            tex.append(r"\midrule")
            for row in block[2:]:                       # skip the |---| separator
                tex.append(" & ".join(_tex_cells(row.strip("|").split("|"))) + r" \\")
            tex.append(r"\bottomrule")
            tex.append(r"\end{tabular}\end{adjustbox}\end{center}")
        else:
            i += 1
    return "\n".join(tex) + "\n"


# ── orchestration ─────────────────────────────────────────────────────────────

def analyze(campaign_dir: Path, report_dir: Path, with_pngs: bool = True) -> None:
    records = load_records(campaign_dir)
    if not records:
        print(f"No records in {campaign_dir} (journal.jsonl / runs/). Nothing to analyze.")
        return
    print(f"Analyzing {len(records)} runs from {campaign_dir}")

    summary = report_dir / "results" / "summary"
    tables = report_dir / "tables"
    summary.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    reports = {
        "cross_comparison_matirf": cross_comparison(records, "matirf"),
        "cross_comparison_deconv": cross_comparison(records, "deconv"),
        "rejects_and_cost": rejects_and_cost(records),
    }
    ## one standard-vs-optimal table per algorithm — the core of each report chapter
    for solver in SOLVER_ORDER:
        reports[f"algo_{solver.replace('-', '_').lower()}"] = per_algorithm(records, solver)
    reports["adam_shape"] = adam_shape(records)          # the Adam chapter's L1-vs-Tikhonov λ figure
    reports["denoiser_winners"] = denoiser_winners(records)   # ONE combined denoiser table
    for solver in DENOISER_SOLVERS:                      # detailed per-algo denoiser tables (durable data)
        reports[f"denoisers_{solver.replace('-', '_').lower()}"] = denoiser_comparison(records, solver)
    for name, md in reports.items():
        (summary / f"{name}.md").write_text(md)        # human-readable, committed synthesis
        (tables / f"{name}.tex").write_text(md_to_latex(md))   # \input by the report
    ## per-algo galleries are already LaTeX (subfigures + params captions) — written as-is
    for solver in SOLVER_ORDER:
        tex = gallery(records, solver)
        if tex:
            (tables / f"gallery_{solver.replace('-', '_').lower()}.tex").write_text(tex)
    print(f"  wrote {len(reports)} tables + galleries (.md in {summary.name}/, .tex in {tables.name}/)")

    exported = export_showcase(records, campaign_dir, report_dir)
    print(f"  exported {len(exported)} showcase runs (config.toml + metrics.json)")
    if with_pngs:
        n = render_pngs(exported)                          # writes straight into figures/
        print(f"  rendered {n} depth-map/profile PNGs into figures/")
        truth_tex = truth_gallery(records, campaign_dir, report_dir)   # the 3 ground truths, for the intro
        if truth_tex:
            (tables / "truth_gallery.tex").write_text(truth_tex)
            print("  rendered the 3 ground-truth figures")
