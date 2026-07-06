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
