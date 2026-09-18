"""
Adam — gradient-based MAP reconstruction.

Minimizes L(f) = (1 - lambda_reg) * D(Hf, g) + lambda_reg * R(f) with torch.optim.Adam,
using the objective's autograd gradient. Two refinements over plain Adam:

    > the learning rate is halved whenever the loss stops decreasing (StepLR), which
      recovers from a step size that was too ambitious without restarting the run;
    > the run stops early once the loss improves by less than EPS, so an easy problem does
      not pay for max_iter iterations.

f is projected onto f >= 0 after every step: these are images, and negative intensity is
not a physically meaningful answer.

This file replaces v1's `common/algorithms/specializable/adam/` plus `AdamAlgo` in both
matirf and deconv. There is nothing problem-specific left to subclass — the operator
arrives inside the objective, and the ridge warm start that used to be a MA-TIRF-only
override is now the `init = "ridge"` parameter, available to every problem.
"""

import time

import torch

from solvers.base import Solver
from solvers.objective_params import INIT_UI_PARAM


ADAM_UI_PARAMS = {
    "max_iter": {
        "title": "Maximum iteration number",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "\\text{max\\_iter}",
                       "default": 1000},
    },
    "lr": {
        "title": "Learning rate",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\text{lr}",
                       "default": 0.1},
    },
    "K": {
        "title": "K check",
        "type": "value",
        "param_info": {"dtype": int, "unit": "", "latex_name": "K", "default": 10},
    },
    "EPS": {
        "title": "Stopping criterion",
        "type": "value",
        "param_info": {"dtype": float, "unit": "", "latex_name": "\\epsilon",
                       "default": 1e-8},
    },
    **INIT_UI_PARAM,
}


class Adam(Solver):
    """Adam optimizer with learning-rate decay, early stopping and positivity projection."""

    name = "ADAM"
    estimator_type = "MAP"
    uses_regularization = True
    ui_params = ADAM_UI_PARAMS

    def solve(self, objective, f0, params):
        self.fix_randomness()

        max_iter = int(params.get("max_iter", 1000))
        lr = float(params.get("lr", 0.1))
        K = max(1, int(params.get("K", 10)))          # report/check every K iterations
        EPS = float(params.get("EPS", 1e-8))          # stop when the loss barely moves

        f = f0.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([f], lr=lr)
        ## halves the learning rate on demand; stepped only when the loss goes back up:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        started = time.time()
        self.report(objective.describe())
        previous_loss = objective.value(f)
        self.report(f"iter    0 | loss={previous_loss.item():.3e} | lr={lr:.2e}")

        for iteration in range(1, max_iter + 1):
            if self.interrupted:
                self.report(f"Interrupted at iteration {iteration}.")
                return f.detach()

            optimizer.zero_grad()
            loss = objective.value(f)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                f.clamp_(min=0.0)                      # images are non-negative

            if iteration % K:
                continue

            delta = (loss - previous_loss).item()
            current_lr = optimizer.param_groups[0]["lr"]
            self.report(f"iter {iteration:4d} | loss={loss.item():.3e} | "
                        f"lr={current_lr:.2e} | dloss={delta:+.3e}")
            self.publish(f)

            if delta >= 0:
                scheduler.step()                       # loss rose: the step was too large
            elif delta > -EPS:
                self.report(f"Stopping criterion EPS reached after {iteration} iterations.")
                return self._finish(f, started)
            previous_loss = loss

        self.report(f"Maximum iteration number ({max_iter}) reached.")
        return self._finish(f, started)

    def _finish(self, f, started):
        self.report(f"Execution in {time.time() - started:.2f} s.")
        return f.detach()
