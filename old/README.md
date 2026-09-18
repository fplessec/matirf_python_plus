# old — superseded implementation, kept for reference

Everything here was **replaced during the v2 rewrite** and is scheduled for deletion once
v2 is finished. Nothing in the running project imports it: it is kept only so the old
approach can still be consulted while the migration is being reviewed.

**Do not add to it, and do not import from it.** It is inert — its internal imports still
use the old absolute paths (`common.algorithms...`, `matirf.core...`), so it will not even
load from here. That is deliberate: it makes an accidental dependency impossible.

## What was replaced by what

| v1 (here)                          | v2 (in the project)                        |
|------------------------------------|--------------------------------------------|
| `common/algorithms/`               | `solvers/` — six solvers, problem-agnostic  |
| `common/algorithms/reusable/*`     | `solvers/fidelities/`, `solvers/regularizers/` |
| `common/core/base_pipeline.py`     | `pipeline.py`                               |
| `common/core/base_result.py`       | `core/result.py`                            |
| `common/core/pipeline_operations.py` | `core/problem.py` (`InverseProblem`)      |
| `matirf/core/operations.py`        | `problems/matirf/physics.py`                |
| `matirf/core/pipeline_operations.py` + `matirf_pipeline.py` | `problems/matirf/problem.py` |
| `matirf/algorithms/` (6 subclasses) | nothing — solvers are generic now          |
| `deconv/core/operations.py`        | `problems/deconv/operator.py`               |
| `deconv/algorithms/` (2 subclasses) | nothing — solvers are generic now          |

About 3 100 lines, of which roughly 470 were pure plumbing: per-problem algorithm
subclasses whose only content was supplying `apply_forward` / `apply_adjoint`.

## The guarantee that the physics did not change

The v2 tests originally compared against this code, running side by side. That check had
to outlive the code it was written against — otherwise the guarantee would disappear
exactly when it starts mattering, the day someone tidies up `physics.py`.

So the outputs produced **by the code in this folder** were recorded before it was retired:

    problems/matirf/_v1_reference.json
    problems/deconv/_v1_reference.json

Small results (the operator matrices, the PSF, the angle bounds, delta) are stored whole;
large ones (a 13x350x350 preprocessed stack, a 644x1024 blurred image) are stored as a
fingerprint — shape, sum, mean, std, min, max and samples at fixed indices. Any real change
to the optics moves at least one of those. `test_physics_matches_v1` and
`test_matches_v1` compare against those files and keep working after this folder is gone.

## Deleting it

When v2 is accepted:

    git rm -r old/

Nothing else needs to change. The reference files above stay where they are.
