============================================================================== \
*Pour Charles: \
https://www.youtube.com/watch?v=2twLJU_ggGI \
https://www.youtube.com/watch?v=iaoBBwcXYKQ*

*Deux branches sur le github:* \
*- v1: mon travail humain* \
*- v2 (dorénavant la branche main): claude code pour faire un beau projet globalement \
(mise a jour des algos et quelques points différents, mais surtout architecture, propreté etc)*

*Merci à tous·tes !* \
\==============================================================================

# matirf_python_plus


Generic inverse problem solver. Reconstructs an unknown image **f** from a measurement **g** and a forward operator **H** by solving iteratively:

```
min_f (1 - lambda) * D(Hf, g) + lambda * R(f)
```

Currently implements two inverse problems:
- **matirf** : 3D MA-TIRF reconstruction (multi-angle TIRF microscopy)
- **deconv** : 2D image deconvolution

Each problem can be solved with multiple algorithms (ADAM, MCMC, PnP, ADMM-PnP, PPXA, ADMM), multiple data fidelity terms (Gaussian, Poisson, Poisson-Gaussian), and multiple regularizations (TV, Tikhonov, Hessian, SHV, ...).

The project supports both a **GUI** (graphical interface to configure and visualize reconstructions) and a **CLI** (command line for batch processing).

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### 1. Clone the repository

```bash
git clone https://github.com/fplessec/matirf_python_plus.git
cd matirf_python_plus
```

### 2. Create a virtual environment

**macOS / Linux :**

```bash
python3.10 -m venv env
source env/bin/activate
```

**Windows (PowerShell) :**

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**Windows (cmd) :**

```cmd
python -m venv env
env\Scripts\activate.bat
```

### 3. Install the project

```bash
pip install -e .
```

This installs the project in editable mode: any code change is immediately available without reinstalling.

### 4. Verify the installation

```bash
list
```

Expected output:

```
Available inverse problems:

  deconv               (no config)
  matirf               (no config)
```

---

## Usage

### Show available commands

```bash
help
```

### List available inverse problems

```bash
list
```

### Launch the GUI

```bash
matirf gui
deconv gui
```

### Launch the CLI

```bash
matirf cli
deconv cli
```

### Launch the synthetic ground-truth generator (matirf only)

```
matirf synth              # or: python -m problems.matirf.synthetic
```

**What it is for.** To judge a reconstruction you need data whose exact answer you already know. `matirf synth` designs that answer: a plausible, reproducible 3D object `f_true` — a small piece of an adherent cell as MA-TIRF sees it, a slab a few hundred nm deep (300 nm by default, where reconstructions are reliable) under a TIRF field of view. The pipeline then simulates the measurement from it (`g = H · f_true`, plus noise), reconstructs, and compares.

**Three kinds of objects**, each one class in `problems/matirf/synthetic/objects/`:

| Object | Represents |
|---|---|
| `Ellipsoid` | vesicles and endosomes (small; at random depths, or on an inclined plane or a sphere), or focal adhesions (large, flat, against the glass) |
| `Filament` | actin stress fibres, microtubules: smooth curves nearly parallel to the glass, thickness drawn per fibre |
| `Membrane` | the basal membrane of an adherent cell: a footprint with an edge, close to the glass at the edge and higher under the cell body |

**One scene file.** A scene is a single TOML (`[grid]`, `[sampling]` seed, then one section per object with its `count` and sizes, all in nm). The window edits `problems/matirf/synthetic/cache/scene.toml`; ready-made scenes — the benchmark's truths — are in `problems/matirf/synthetic/presets/`: `vesicles` (on an inclined plane), `fibres` (thin and thick, crossing at different depths), `cell`, and `cell_fibres_vesicles` (the three together).

**The benchmark truths are shipped** in `problems/matirf/data/measurements/synthetic/`: `vesicles.TIF`, `fibres.TIF`, `cell.TIF`, `cell_fibres_vesicles.TIF` (each with its `.truth.json`), and `measurement_parameters.json` — the MA-TIRF microscope (13 angles, 62.6–69.8°, NA 1.33, 491 nm) that builds `H_synthetic` and simulates `g` from them. Select a truth and that .json in synthetic mode, with `z0 = 0`, `zN = 300` and `nz` = 50 (or 30, 25, 15, 10).

**A truth file describes itself.** Saving writes the TIF and a record `<name>.truth.json` with its geometry and its complete scene: the truth can be reproduced from it bit for bit, and MA-TIRF refuses to reconstruct it on another depth range than the one it was generated in.

**Finer than the reconstruction.** The truth has `nz × z_oversampling` planes (150 by default): the measurement is simulated from the fine truth and the reconstruction, on `nz` planes, is compared with the truth averaged back. One truth can thus be reconstructed on 50, 30, 25, 15 or 10 planes with an exact reference.

**Typical workflow.** `matirf synth` → edit or *Load scene…* → *Generate / preview* → *Use as MA-TIRF truth* (saves the truth and sets the MA-TIRF config: synthetic mode, the file, `nz`, `z0`, `zN`) → `matirf gui`, choose the noise and an algorithm, run.

**Noise is not part of the truth**: it is added when the measurement is simulated (`[add-noise]`), so one truth serves every noise level.

**From a script** (what the benchmark does):

```python
from problems.matirf.synthetic import load_preset, generate, save_truth

scene = load_preset("cell")
save_truth(generate(scene), "truths/cell.TIF", scene)       # + truths/cell.truth.json
```

### Reset the cached configuration

Each inverse problem caches its configuration in a `config.toml` file. To reset it to defaults:

```bash
matirf reset
deconv reset
```

### Settings

Application settings (device, dtype, theme, font sizes, window dimensions) are stored in `settings.toml` at the project root. They persist between sessions and can be managed from the CLI or a dedicated GUI.

```bash
settings show                # display all settings with current/default values
settings set device cuda     # change a setting
settings set dark_style false
settings reset device        # reset one setting to its default
settings reset               # reset all settings to defaults
settings gui                 # open a graphical settings editor
```

---

## Project structure

```
matirf_python_plus/
    cli.py            one entry point; auto-discovers every problems/<name>/
    pipeline.py       wires a problem + config into a run (prepare -> solve -> evaluate)
    core/             the contract: operator.py, objective.py, problem.py, features.py, metrics/
    solvers/          every algorithm, once for all problems (adam, ppxa, admm, pnp, pnp_admm,
                      mcmc) + denoisers/, regularizers/, fidelities/
    problems/
        matirf/       operator.py, problem.py, ui.py, __init__.py (+ physics.py, synthetic/)
        deconv/       operator.py, problem.py, ui.py, __init__.py
    gui/              the interface, built generically from each problem's ui (factory.py)
    fileio/           TOML / TIF / PNG I/O
    benchmarks/       the comparison campaign (campaign.py, runner.py, analyze.py)
    settings/         user settings + dark / light palettes
    docs/             reconstruction_algorithms.pdf, tutorial/, developer_guide.md
```

---

## Extending the framework

See the **[developer guide](docs/developer_guide.md)** for the architecture (three core objects:
`ForwardOperator`, `Objective`, `InverseProblem`) and step-by-step recipes. In short:

- **A new inverse problem** = two core files under `problems/<name>/` — `operator.py` (a
  `ForwardOperator`: `apply` + `adjoint`) and `problem.py` (an `InverseProblem`) — plus a `ui.py`
  declaration and a `pyproject.toml` entry point. **No algorithm code, no window code.**
- **A new solver** = one file in `solvers/` subclassing `Solver`; it works on every problem.
- **A new metric / denoiser / regulariser** = one file in `core/metrics/reusable/`,
  `solvers/denoisers/` or `solvers/regularizers/`; it appears everywhere automatically.

## Documentation

Everything is in `docs/`:

- **[Tutorial](docs/tutorial/tutorial.md)** — using the GUI and CLI, with screenshots
  (also as [tutorial.pdf](docs/tutorial/tutorial.pdf)).
- **[Algorithms reference](docs/reconstruction_algorithms.pdf)** — the per-algorithm report:
  how each of the six solvers (Adam, PPXA, ADMM, PnP-HQS, ADMM-PnP, MCMC) works, its key
  parameters, the benchmark results on three ground truths (vesicles, fibres, cell) with 2D
  deconvolution as a well-posed companion, and the intrinsic limits of MA-TIRF. Its LaTeX source
  is the separate repo [matirf_benchmark_report](https://github.com/fplessec/matirf_benchmark_report).
- **[Developer guide](docs/developer_guide.md)** — the architecture (`ForwardOperator`,
  `Objective`, `InverseProblem`) and step-by-step recipes to extend it.
