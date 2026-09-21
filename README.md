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
| `Ellipsoid` | vesicles and endosomes (small, anywhere in depth), or focal adhesions (large, flat, against the glass) |
| `Filament` | actin stress fibres, microtubules: smooth curves, nearly parallel to the glass |
| `Membrane` | the basal membrane of an adherent cell: a footprint with an edge, close to the glass at the edge and higher under the cell body |

**One scene file.** A scene is a single TOML (`[grid]`, `[sampling]` seed, then one section per object with its `count` and sizes, all in nm). The window edits `problems/matirf/synthetic/cache/scene.toml`; ready-made scenes — the benchmark's truths — are in `problems/matirf/synthetic/presets/`: `vesicles`, `adhesions_fibres`, `cell`.

**The benchmark truths are shipped** in `problems/matirf/data/measurements/synthetic/`: `vesicles.TIF`, `adhesions_fibres.TIF`, `cell.TIF` (each with its `.truth.json`), and `measurement_parameters.json` — the MA-TIRF microscope (13 angles, 62.6–69.8°, NA 1.33, 491 nm) that builds `H_synthetic` and simulates `g` from them. Select a truth and that .json in synthetic mode, with `z0 = 0`, `zN = 300` and `nz` = 50 (or 30, 25, 15, 10).

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
    cli.py                          # centralized entry point
    common/                         # shared components
        algorithms/                 # algorithm stack (three layers)
            base/                   #   machinery: Algorithm, LossComputer, differential operators
            reusable/               #   optimization terms (interfaces + implementations)
                data_fidelities/    #     noise models (Gaussian, Poisson, ...)
                regularizations/    #     priors (TV, Tikhonov, Hessian, SHV, ...)
            specializable/          #   algorithm skeletons with hooks
                adam/               #     ADAM (MAP estimator)
                mcmc/               #     MCMC (MMSE estimator)
                pnp/                #     Plug-and-Play (HQS) + ADMM-PnP
                ppxa/               #     PPXA (proximal)
                admm/               #     ADMM
        denoisers/                  # denoiser implementations
        gui/                        # shared GUI components
        core/                       # pipeline base classes
        in_out/                     # file I/O (TOML, TIF, ...)
    matirf/                         # 3D MA-TIRF inverse problem
        algorithms/                 #   problem-specific algorithm subclasses
        core/                       #   pipeline, forward operator
        gui/                        #   custom GUI elements
        synthetic/                  #   plausible, reproducible ground truths (matirf synth); presets/ = benchmark truths
        cache/                      #   cached config.toml
        data/                       #   measurements and results
    deconv/                         # 2D deconvolution inverse problem
        algorithms/                 #   problem-specific algorithm subclasses
        core/                       #   pipeline, forward operator (FFT convolution)
        gui/                        #   custom GUI elements
        cache/                      #   cached config.toml
        data/                       #   measurements and results
```

---

## Adding a new inverse problem

To add a new inverse problem (e.g. `superres`):

1. Create the package:

```
superres/
    __init__.py         # define paths, default config
    main.py             # define main() that calls run_app(...)
    cache/
        __init__.py
    core/
        ...             # pipeline, forward operator
    algorithms/
        __init__.py     # ALGORITHMS = {cls.name: cls for cls in [...]}
        my_algo.py      # class MyAlgo(BaseAdam):
                        #     features = {"2d"}
                        #     def apply_forward(self, H, f): ...
                        #     def apply_adjoint(self, H, x): ...
    gui/
        ...
```

2. Add the entry point in `pyproject.toml`:

```toml
[project.scripts]
superres = "cli:main"
```

3. Reinstall:

```bash
pip install -e .
```

4. Done:

```bash
superres gui
list
```

---

## Adding a new algorithm

Algorithm skeletons live in `common/algorithms/specializable/`. To add a new algorithm:

1. Create a sub-package in `common/algorithms/specializable/` (e.g. `common/algorithms/specializable/my_algo/`)
2. Implement the base class inheriting from `Algorithm`
3. Set the class attributes: `name`, `ui_params`, `estimator_type`, `uses_denoiser`, `uses_regularization`
4. Export it from `common/algorithms/specializable/__init__.py` (and re-export from `common/algorithms/__init__.py`)
5. Create the problem-specific subclass in `matirf/algorithms/` or `deconv/algorithms/` (with `apply_forward`, `apply_adjoint`, `supported_features`) and register it in that problem's `algorithms/__init__.py` (`ALGORITHMS`)

---

## Adding a new denoiser

1. Create a new file in `common/denoisers/reusable/` (e.g. `denoise_bm3d.py`)
2. Implement a class inheriting from `Denoiser` with attributes `name`, `supports_3d`, `supports_anisotropy`
3. Implement the `denoise(self, y, sigma, delta=1.0)` method (`sigma` is a noise level on the [0, 255] scale)
4. Add the instance to `_ALL_DENOISERS` in `common/denoisers/reusable/denoiser_list.py`

The denoiser will automatically appear in the UI for all algorithms that use denoisers.

---

## Adding a new regularization

1. Create a new file in `common/algorithms/reusable/regularizations/` (e.g. `wavelet.py`)
2. Implement a class inheriting from `Regularization` with `name`, `display_name`, `uses_diff_ops`
3. Implement `loss(self, f, diff_ops)` and `prox(self, f, lambda_reg, diff_ops)`
4. Add the class to `_ALL_REGULARIZATIONS` in `common/algorithms/reusable/regularizations/__init__.py`

The regularization will automatically appear in the UI for all algorithms that use regularizations.
