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

```bash
matirf synth              # or: python -m matirf.synthetic
```

**What it is for.** To judge whether a reconstruction algorithm is good, you need data whose exact answer you already know. `matirf synth` designs such an answer: a reproducible 3D object `f_true`. The reconstruction pipeline then simulates the measurement from it (`g = H · f_true`, plus noise), runs an algorithm, and compares the result against `f_true` with the quality metrics. This is how the algorithms (ADAM, PPXA, ADMM, PnP, ADMM-PnP, MCMC) are compared on equal footing.

**Two files, two concerns.** The generator is driven entirely by TOML, stored in `matirf/synthetic/cache/`:

| File | Describes |
|---|---|
| `ground_truth.toml` | *what* the objects are — one section per object type, each with a `count` and its characteristic parameters, expressed in nanometres (grid-independent). Plus a master `seed`. |
| `grid.toml` | *how* that continuous truth is sampled onto the flat anisotropic MA-TIRF slab: `nz`, `z0_nm`, `zN_nm`, lateral extent. |

Separating them means you can re-sample the same objects on a finer grid, or change the objects without touching the grid.

**Object types available.** Each is a `SyntheticObjectType` registered in `OBJECT_TYPES` — the exact analogue of an entry in the algorithm registry. Setting a type's `count` to `0` disables it.

| Type | Shape |
|---|---|
| `Ellipsoid` | Gaussian blobs, flat-ish (MA-TIRF observes a thin slab), `sharpness > 1` gives a flat-top "soft binary" |
| `Filament3D` | curved filaments |
| `Membrane` | a corrugated membrane sheet |

**Typical workflow.**

1. `matirf synth` — the window shows one section per TOML section on the left, a preview on the right.
2. Adjust the grid, the seed and the object counts/parameters. Every edit is written straight to the TOML.
3. **Generate / preview** — builds `f_true`, normalised to `[0, 1]`, and displays it (depth map + profiles, or image + 3D histogram).
4. Then either:
   - **Save as TIF…** — write the truth to disk and use it however you like, or
   - **Use as MA-TIRF synthetic truth** — saves the TIF *and* updates the MA-TIRF config for you: sets mode to `synthetic-data`, points `input-paths.tif` at the file, and copies `nz`, `z0`, `zN` into `oper-params` so the operator matches the grid.
5. `matirf gui` — pick a measurement JSON and an algorithm, then run. The reconstruction is compared to your truth automatically.

**Reproducibility.** Objects are sampled from a master `seed`, and the registry order is fixed, so the RNG is consumed in a fixed order: the same two TOML files always produce the exact same `f_true`. Keep them next to your results and the experiment is reproducible.

**Noise is deliberately not part of the truth.** `f_true` is noiseless. The measurement noise is applied by the reconstruction pipeline through the `[add-noise]` section of the MA-TIRF config, so you can re-run the same truth at several noise levels.

**Programmatic use** (for scripted benchmarks, no GUI):

```python
from problems.matirf.synthetic import load_grid_config, load_gt_config, generate_ground_truth

f_true = generate_ground_truth(load_grid_config(), load_gt_config())   # (nz, ny, nx) in [0, 1]
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
        synthetic/                  #   reproducible synthetic ground-truth generator (matirf synth)
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
