# Architecture and Logic Flow

## Entry point

`main.py` (one per problem: `matirf/main.py`, `deconv/main.py`) calls `run_app` with 4 arguments that specialize the generic framework:

```python
run_app(
    control_window_class=ControlWindow,      # the GUI window class
    pipeline_class=MaTirfPipeline,           # the pipeline class
    load_config_fn=partial(...),             # how to load a config.toml
    name="MA-TIRF Reconstruction",           # display name
)
```

`run_app` (`common/gui/app.py`) parses CLI arguments:
- `matirf gui` → opens the GUI via `_open_gui`
- `matirf cli -c config.toml -o output/` → runs headless via `_run_cli`

`matirf/main.py` intercepts one sub-command *before* `run_app`:
- `matirf synth` → opens the synthetic ground-truth generator (see below)

`cli.py` advertises `synth` for any problem whose package contains a `synthetic/`
sub-package, so the help text stays correct if another problem gains one.


## Three packages

```
common/     generic framework (pipeline, algorithms, GUI, I/O)
matirf/     MA-TIRF specialization (3D, scale-ambiguous, anisotropic)
deconv/     deconvolution specialization (2D)
```

Each specialization provides:
- a `BasePipeline` subclass with 7 class attributes (no methods to override)
- a `PipelineOperations` subclass with problem-specific static methods
- a `ControlWindow` subclass with UI section descriptors
- a `DisplayWindow` subclass with figure/synthetic hooks


## GUI mode

### ControlWindow

`BaseControlWindow` is configured entirely through class attributes:
- `sections_left` / `sections_right` : UI section descriptors
- `pipeline_class` : which pipeline to create
- `display_window_manager_class` : manages open display windows
- `cached_config_path` / `default_config` / `results_dir`

The user fills in parameters (input files, algorithm, hyperparameters).
Every change is saved into a cached `config.toml`.

When the user clicks **Run**:
1. loads the cached config.toml
2. `PipelineClass.create(config)` — instantiates and registers in `_pipelines`
3. creates a `DisplayWindow` which wires the 4 callback attributes
4. `pipeline.start()` launches the reconstruction

### DisplayWindow

`BaseDisplayWindow` connects to the pipeline via `PipelineQtBridge`,
a thread-safe bridge (Qt signals with `QueuedConnection`) between the
algorithm thread and the Qt main thread.

Callback wiring (in `_connect_pipeline_callbacks`):
```
pipeline.on_message      → qt_bridge.message.emit      → _print (display in text panel)
pipeline.on_finished     → qt_bridge.finished.emit     → _on_finished (update figures, enable Save)
pipeline.on_error        → qt_bridge.error.emit        → _on_error (display error)
pipeline.on_state_changed → qt_bridge.state_changed.emit → _on_state_changed (start/stop live preview timer)
```

Live preview: a `QTimer` polls `algorithm._latest_f` every 250ms to update
figures during computation, instead of relying on Qt signals (which would
pile up and freeze the UI).


## CLI mode

`_run_cli` creates the pipeline, wires callbacks directly (no Qt):
```python
pipeline.on_message  = lambda msg: print(msg)
pipeline.on_finished = lambda result: pipeline.save_results(output_path)
pipeline.on_error    = lambda err: print("[ERROR]", err)
```
Then calls `pipeline.start()` and waits with `while pipeline.is_running`.


## Pipeline life cycle

`BasePipeline` is fully concrete. Subclasses only set 7 class attributes:

| Attribute            | Purpose                                          |
|----------------------|--------------------------------------------------|
| `RESULT_CLASS`       | the Result dataclass (e.g. `ReconstructionResult`)|
| `PROBLEM_FEATURES`   | set of feature constants (e.g. `{THREE_D, ...}`) |
| `ALGORITHM_REGISTRY` | dict mapping algo name → algo class              |
| `PIPELINE_OPERATIONS`| `PipelineOperations` subclass                    |
| `SAVE_IMAGE`         | `callable(tensor, path)` to save image data      |
| `LOAD_IMAGE`         | `callable(path) -> tensor` to load image data    |
| `IMAGE_EXTENSION`    | file extension (e.g. `'TIF'`, `'png'`)           |

### start()

```
start()
  ├─ validate_config()     via PIPELINE_OPERATIONS
  ├─ _set_state(LOADING)
  ├─ setup()               builds g, H via PIPELINE_OPERATIONS.build_g_H_real/synthetic
  │   └─ _create_algorithm()   instantiates the selected algo, injects active features
  ├─ _set_state(COMPUTING)
  └─ run()                 wires algo._print / _on_finished / _on_error, launches thread
       ├─ algo._print(msg)        → pipeline._print → result.messages + on_message
       ├─ algo._on_finished(f)    → store f, compute metrics, COMPLETED, on_finished
       └─ algo._on_error(err)     → FAILED, on_error
```

### Pipeline states

```
IDLE → LOADING → COMPUTING → COMPLETED
                           → FAILED
                           → INTERRUPTED
```

### Pipeline registry

Each subclass maintains its own `_pipelines` list (via `__init_subclass__`,
called once at class definition / import time). This allows running multiple
reconstructions in parallel (each in its own thread) within a single session.

```python
pipeline = MaTirfPipeline.create(config)    # instantiate + register
MaTirfPipeline.get_all()                     # list active pipelines
MaTirfPipeline.remove(pipeline)              # unregister one
MaTirfPipeline.stop_all()                    # stop + clear all
```


## Callback architecture

No callback dictionaries. Direct function attributes at two levels:

### Algorithm → Pipeline

Wired by `BasePipeline.run()`:
```
algorithm._print       = pipeline._print
algorithm._on_finished = pipeline._on_algo_finished
algorithm._on_error    = pipeline._on_algo_error
```

### Pipeline → GUI / CLI

Wired by `DisplayWindow` (GUI) or `_run_cli` (CLI):
```
pipeline.on_message        (msg)
pipeline.on_finished       (result)
pipeline.on_error          (err)
pipeline.on_state_changed  (old_state, new_state)
```

`_print(msg)` always logs into `result.messages` before notifying the listener,
so `messages.txt` contains all messages (pipeline + algorithm).


## Algorithm

`Algorithm` (ABC) runs in a background thread. Key methods:
- `run(g, H, params)` — the iterative algorithm (abstract, implemented by each algo)
- `_print(msg)` — wired to pipeline, logs + notifies
- `_update_figure(f)` — stores a snapshot for live preview (no signal, polled by timer)
- `is_stop_requested()` — checks `_stop_event` inside the iteration loop
- `stop_running()` — sets `_stop_event` and joins the thread

Threading: `_run()` launches `_run_wrapper` in a daemon thread.
`_run_wrapper` calls `run()`, then `_on_finished(result)` or `_on_error(err)`.


## PipelineOperations

Static method namespace for problem-specific logic:
- `build_g_H_real(config)` → `(g, H)`
- `build_g_H_synthetic(config)` → `(g, H, f_true)`
- `validate_config(config)` → list of error strings
- `compute_synthetic_outputs(result, config, features)` → fills metrics, delta, diff
- `compute_preprocessing_preview(config, mode)` → preview data for the GUI


## Save / Load

`BasePipeline.save_results(save_dir)` writes:
```
save_dir/
  f.{ext}           reconstructed image
  config.toml       reconstruction parameters
  messages.txt      all messages (pipeline + algorithm)
  f_true.{ext}      ground truth (synthetic mode only)
  metrics.json      quality metrics (synthetic mode only)
```

`BasePipeline.load_results(directory)` reads the folder back, recomputes
synthetic outputs if applicable, and sets state to COMPLETED.


## Synthetic ground truth (`matirf synth`)

A side entry point, independent of the reconstruction pipeline. It produces the
`f_true` that synthetic mode consumes, so that algorithms can be compared on data
whose exact answer is known.

```
matirf synth  ──►  SyntheticTruthGeneratorWindow  (matirf/synthetic/gui.py)
                        │
                        ├─ reads/writes  cache/grid.toml          [grid]      how it is sampled
                        ├─ reads/writes  cache/ground_truth.toml  [sampling]  master seed
                        │                                         [<object>]  one per type
                        │
                        ├─ generate_ground_truth(grid_cfg, gt_cfg)  ──►  f_true  (nz,ny,nx) in [0,1]
                        │       └─ iterates OBJECT_TYPES in registry order (fixed RNG order)
                        │
                        └─ "Use as MA-TIRF synthetic truth"
                                saves the .TIF, then updates the MA-TIRF config:
                                    input-paths.mode = "synthetic-data"
                                    input-paths.tif  = <path>
                                    oper-params.nz / z0 / zN  = the grid's values
```

The design deliberately mirrors the algorithm layer: each object type is a
`SyntheticObjectType` with `name` / `toml_key` / `ui_params` and a `sample()`, registered
in `OBJECT_TYPES` exactly as an algorithm is registered in `ALGORITHMS`. Adding a class
to that registry automatically creates both its GUI section and its TOML section — no
GUI code to write.

Two invariants worth knowing:

- **Objects are continuous, defined in nm**, then integrated onto the anisotropic voxel
  grid (midpoint super-sampling). They are never drawn directly onto voxels, which avoids
  grid-aligned artefacts.
- **Noise is not baked into the truth.** `f_true` is noiseless; `g = H · f_true` and its
  noise are produced by the reconstruction pipeline via the config's `[add-noise]`
  section. The same truth can therefore be replayed at several noise levels.

From there the flow rejoins the normal pipeline: `build_g_H_synthetic(config)` loads
`f_true` from `input-paths.tif`, and `compute_synthetic_outputs` fills the metrics and
the difference image.
