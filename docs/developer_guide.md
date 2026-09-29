# Developer guide

How `matirf_python_plus` is put together, and how to extend it: add an inverse problem, a
solver, a metric, a denoiser or a regulariser. For using the tool see the
[tutorial](tutorial/tutorial.md); for the algorithms' theory see the
[algorithms reference](algorithms/reconstruction_algorithms.pdf).

---

## 1. The architecture in one picture

Three objects (`core/`) are the whole contract; everything else is built on them.

```
                 ┌─────────────────┐
   physics  ───► │ ForwardOperator │  H: apply / adjoint (+ solve_normal, ridge_inverse…)
                 └────────┬────────┘
                          │  a problem carries one
                 ┌────────▼────────┐
   declaration ► │ InverseProblem  │  features, validate(config), prepare(config)→(H, g, f_true)
                 └────────┬────────┘
                          │  prepare builds an
                 ┌────────▼────────┐
   what to min ► │   Objective     │  (1-λ)·D(Hf,g) + λ·R(f):  value / grad / prox
                 └────────┬────────┘
                          │  every solver consumes one
                 ┌────────▼────────┐
   algorithms  ► │     Solver      │  solve(objective, f0, params) → f
                 └─────────────────┘
```

**The layering rule (do not break it):** a **solver** depends only on `Objective`, never on a
problem; a **problem** depends only on `ForwardOperator`, never on a solver; **`core/` never
imports `gui/` or `solvers/`.** This is why a new problem needs *no algorithm code* and a new
solver works on *every* problem.

---

## 2. Directory map

```
core/            the contract: operator.py, objective.py, problem.py, features.py,
                 metrics/, result.py, diagnostics.py, noise.py, normalization.py
solvers/         every algorithm, written once for all problems: adam, ppxa, admm, pnp,
                 pnp_admm, mcmc; base.py (Solver), objective_params.py (shared UI params),
                 denoisers/, regularizers/, fidelities/, differential_operators.py; old/
problems/        one package per inverse problem
    matirf/      operator.py, problem.py, ui.py, __init__.py (+ physics.py, parameters.py…)
    deconv/      operator.py, problem.py, ui.py, __init__.py
gui/             the interface, built generically from each problem's ui declaration
                 (factory.py) — no problem contributes window code
fileio/          TOML / TIF / PNG I/O
benchmarks/      the campaign (campaign.py), runner, metrics glue, analyze.py
settings/        user settings + the dark/light palettes
pipeline.py      wires a problem + config into a run (prepare → solve → evaluate)
cli.py           one entry point; auto-discovers every package under problems/
```

---

## 3. Add an inverse problem

A new problem is **two core files** — an operator and a declaration — plus a UI declaration and
a small `__init__`. No algorithm code, no window code. Use `problems/deconv/` as the template.

### 3.1 `operator.py` — the physics

Subclass `ForwardOperator`. **Only `apply` and `adjoint` are required**; every other capability
(`gram`, `solve_normal`, `ridge_inverse`, `lipschitz`) has a matrix-free default built from them,
so the problem is usable immediately.

```python
from core import Feature, ForwardOperator, features

class DeconvOperator(ForwardOperator):
    name = "deconv"
    features = features(Feature.TWO_D)          # THE source of truth for the problem's features

    @classmethod
    def from_config(cls, config): ...           # build a configured instance from the TOML

    def apply(self, f):    return ...           # H f
    def adjoint(self, y):  return ...           # Hᵀ y

    def solve_normal(self, b, lam=0.0, **kw):   # OPTIONAL override, for speed only
        ...                                     # deconv: a Wiener division in Fourier
```

Override `solve_normal` only when the structure of H gives a faster route (deconv inverts in
Fourier; MA-TIRF inverts the small dense matrix). It speeds up PPXA, ADMM, PnP and ADMM-PnP at
once, because their data step *is* `solve_normal`. The `features` you declare here decide which
solvers and metrics apply — nothing restates them.

### 3.2 `problem.py` — the declaration

Build one `InverseProblem`. It answers three questions (kind? valid? what am I solving?) and
carries the file hooks:

```python
from core import InverseProblem

DECONV = InverseProblem(
    name="deconv",
    operator_class=DeconvOperator,          # provides features + from_config
    load_measurement=load_measurement,      # REAL mode: read g from disk
    load_truth=load_truth,                   # SYNTHETIC mode: read f_true (None ⇒ real only)
    simulate=simulate,                       # SYNTHETIC: g from f_true (default: operator.apply)
    validate=validate,                       # -> list of human messages ([] when ready)
    save_image=save_png, load_image=load_png, image_extension="png", raw_path_key="png",
    default_config=DEFAULT_DECONV_CONFIG,
    ui=DECONV_UI,                            # the interface declaration (§3.3)
)
```

`validate` returns *all* problems with a config at once (never raises) so the GUI can show them
together. A loader may return `(data, refined_config)` when loading changes the geometry (MA-TIRF
drops background angle stacks, so H is built from the refined config).

### 3.3 `ui.py` — the interface (typed, no GUI code)

Declare a `gui.spec.ProblemUI` (titles, previews, figure views, the difference viewer). The GUI
factory draws it; the problem never imports `gui` drawing code, so the interface can be replaced
without touching the problem. `problems/matirf/ui.py` is the reference (two figure views, a depth
preview, a metrics/difference section).

### 3.4 `__init__.py` + entry point

Export the problem and its paths (`CONFIG_PATH`, `DEFAULT_*_CONFIG`, `MEASUREMENTS_DIR`,
`results_dir`) and expose `PROBLEM = DECONV`. Then register the console command in
`pyproject.toml`:

```toml
[project.scripts]
superres = "cli:main"
```

`pip install -e .`, and `superres gui` / `superres cli` work — `cli.py` auto-discovers any
`problems/<name>/` that has a `problem.py`.

---

## 4. Add a solver

One file in `solvers/`, subclassing `Solver`. It sees only the `Objective` (and, through it, the
operator's `solve_normal` / `ridge_inverse`), so it is automatically available to every problem
whose `features` satisfy its `requires`.

```python
from solvers.base import Solver

class GradientDescent(Solver):
    name = "GD"
    estimator_type = "MAP"                      # or "MMSE"
    requires = NO_FEATURES                      # e.g. {Feature.THREE_D} to restrict
    supported_noise_models = NOISE_MODEL_NAMES  # or frozenset({"gaussian"})
    uses_regularization = True                  # then reg/λ/δ appear in its UI automatically
    uses_denoiser = False
    ui_params = {"max_iter": {...}, "lr": {...}}

    def solve(self, objective, f0, params):
        f = f0
        for _ in range(int(params["max_iter"])):
            f = (f - params["lr"] * objective.grad(f)).clamp(min=0.0)
        return f
```

`initial_guess` defaults to the shared ridge start (`ridge_start`, λ_rr = s2²); override only if
needed. Register the class in `solvers/__init__.py` (`SOLVERS`). It then appears in the GUI, the
CLI and the benchmark for every problem it supports — no per-problem code.

---

## 5. Add a metric

One file in `core/metrics/reusable/`, subclassing `Metric`. It compares a reconstruction to a
truth and returns a scalar (or a curve dict).

```python
from core.features import Feature
from core.metrics.base import Metric

class DepthError(Metric):
    name = "Depth Error (nm)"
    requires = {Feature.THREE_D}                # dropped automatically on 2D problems
    def compute(self, f, f_true, features=set(), dz=1.0, **kw):
        ...                                     # scale-align via self._prepare if needed
```

Register it in `core/metrics/reusable/__init__.py` (`_ALL_METRICS`). `compute_all_metrics` runs
every metric whose `requires ⊆ features`; extra inputs (like `dz`) arrive as keyword arguments
from `pipeline._evaluate`. The benchmark records a curated subset (`benchmarks/runner.py`
`CURATED_3D` / `CURATED_2D`); the GUI shows them all.

---

## 6. Add a denoiser or a regulariser

- **Denoiser** — a file in `solvers/denoisers/` with `name`, `supports_3d`,
  `supports_anisotropy`, and `denoise(self, y, sigma, delta=1.0)` (σ on the 0–255 scale); add it
  to the denoiser registry. It then appears for every solver with `uses_denoiser`.
- **Regulariser** — a file in `solvers/regularizers/` with `display_name`, `uses_diff_ops`, and
  `loss(self, f, diff_ops)` + `prox(self, f, lambda_reg, diff_ops)`; add it to
  `REGULARIZATION_LIST`. It then appears for every solver with `uses_regularization`.

---

## 7. Tests & conventions

- Each package has a `_tests.py` run as a module, e.g.:
  ```bash
  PYTHONPATH=. env/bin/python -m solvers._tests
  PYTHONPATH=. env/bin/python -m core._tests
  QT_QPA_PLATFORM=offscreen env/bin/python _gui_smoke_test.py   # 44 GUI checks
  ```
- **Layering:** `core/` imports neither `gui/` nor `solvers/`; a solver never imports a problem.
- **Docstrings** explain *why*, not just *what*; math is written in plain text/Unicode so it
  reads without a renderer.
- One solver per algorithm; the superseded variants were removed after the benchmark (the `v1.0` tag preserves them).
