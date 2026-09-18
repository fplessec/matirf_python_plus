"""
The objective function a solver minimizes — assembled by the problem, consumed by solvers.

    L(f) = D(H f, g)  +  lambda_reg * R(f)
           \________/     \____________/
            fidelity        regularization
        "explain the        "and be a plausible
         measurement"        image while doing it"

--------------------------------------------------------------------------------------
Why the problem builds this, and not the solver
--------------------------------------------------------------------------------------

In v1 each algorithm built its own loss: it reached into the registries, picked a data
fidelity and a regularizer out of a raw params dict, and assembled a LossComputer. Three
consequences, all bad:

    > the dependency ran backwards — the solver decided what problem was being solved;
    > proximal algorithms bypassed LossComputer entirely, so "the loss" was not actually
      the single source of truth about what was being minimized;
    > every algorithm repeated the same assembly code.

Here the objective is built once, by the problem, and handed to the solver. A solver's job
shrinks to: minimize this, however you like. That is what makes one implementation of Adam
(or PPXA, or MCMC) serve every inverse problem.

--------------------------------------------------------------------------------------
The contract solvers rely on
--------------------------------------------------------------------------------------

    value(f)          L(f), differentiable                  — every solver
    grad(f)           dL/df                                 — gradient solvers (Adam, ...)
    residual(f)       H f - g                               — data-consistency steps
    prox_reg(f, tau)  prox of tau * lambda_reg * R          — proximal solvers (PPXA, ADMM)
    operator          the ForwardOperator                   — MCMC, warm starts, step sizes

A solver declares which of these it needs; a term that cannot provide one (a regularizer
with no proximal operator, say) fails loudly at setup rather than silently mid-run.

--------------------------------------------------------------------------------------
Convention change from v1 — read this
--------------------------------------------------------------------------------------

v1 weighted the two terms as  (1 - lambda_reg) * D + lambda_reg * R,  which forces
lambda_reg into [0, 1] and couples the two weights: raising the regularization necessarily
weakens the fidelity. v2 uses the standard form  D + lambda_reg * R,  where lambda_reg is
a free positive number and the terms are independent.

Consequence for saved configs: a v1 value lam1 corresponds to  lam2 = lam1 / (1 - lam1).
For example 0.1 -> 0.111, 0.5 -> 1.0, 0.9 -> 9.0. `lambda_from_v1()` below does it.
"""

from typing import Optional

import torch

from .operator import ForwardOperator


class Objective:
    """
    L(f) = D(H f, g) + lambda_reg * R(f), for one problem instance and one configuration.

    ----------
    > Parameters :
    ----------

    >> operator : ForwardOperator
        The physics. The objective calls `operator.apply` to form H f, and exposes the
        operator so solvers that need more (a ridge inverse, a step size) can ask for it.

    >> g : torch.Tensor
        The measurement, in measurement space.

    >> data_fidelity : DataFidelity
        The noise model D(Hf, g). Must provide `loss(Hf, g)`; may provide `prox(...)`.

    >> regularization : Regularization or None
        The prior R(f). Must provide `loss(f, diff_ops)`; may provide
        `prox(f, weight, diff_ops)`. None means no regularization.

    >> lambda_reg : float
        Weight of the regularization, >= 0, in the standard convention D + lambda_reg * R
        (see the module docstring). 0 disables the regularization term entirely.

    >> diff_ops : DifferentialOperators or None
        Spatial derivative helpers (gradient, divergence, laplacian, hessian) that
        regularizers need, carrying the anisotropy ratio delta. Required only when the
        regularizer uses them.

    ----------
    > Example :
    ----------

        objective = Objective(operator, g, GaussianFidelity(), TotalVariation(),
                              lambda_reg=0.05, diff_ops=DifferentialOperators(delta=0.05))
        f_hat = AdamSolver().solve(objective, f0)
    """

    def __init__(self, operator: ForwardOperator, g: torch.Tensor,
                 data_fidelity, regularization=None, lambda_reg: float = 0.0,
                 diff_ops=None):
        if lambda_reg < 0:
            raise ValueError(f"lambda_reg must be >= 0, got {lambda_reg}")
        self.operator = operator
        self.g = g
        self.data_fidelity = data_fidelity
        self.regularization = regularization
        self.lambda_reg = float(lambda_reg)
        self.diff_ops = diff_ops

    # ── the objective itself ─────────────────────────────────────────────────

    @property
    def is_regularized(self) -> bool:
        """True when a regularization term actually contributes (non-None and weighted)."""
        return self.regularization is not None and self.lambda_reg > 0.0

    def residual(self, f: torch.Tensor) -> torch.Tensor:
        """H f - g. The data-consistency error, used directly by MCMC proposals."""
        return self.operator.apply(f) - self.g

    def data_term(self, f: torch.Tensor) -> torch.Tensor:
        """D(H f, g) alone — useful for reporting the two terms separately."""
        return self.data_fidelity.loss(self.operator.apply(f), self.g)

    def reg_term(self, f: torch.Tensor) -> torch.Tensor:
        """R(f) alone, UNweighted (the caller multiplies by lambda_reg if it wants L)."""
        if self.regularization is None:
            return torch.zeros((), dtype=f.dtype, device=f.device)
        return self.regularization.loss(f, self.diff_ops)

    def value(self, f: torch.Tensor) -> torch.Tensor:
        """
        L(f) = D(H f, g) + lambda_reg * R(f), as a differentiable scalar tensor.

        Kept differentiable on purpose: gradient solvers can simply call .backward() on it,
        exactly as v1's Adam did, so no gradient has to be derived by hand.
        """
        total = self.data_term(f)
        if self.is_regularized:
            total = total + self.lambda_reg * self.reg_term(f)
        return total

    def __call__(self, f: torch.Tensor) -> torch.Tensor:
        """Alias for `value`, so an Objective can be passed anywhere a loss callable is."""
        return self.value(f)

    def grad(self, f: torch.Tensor) -> torch.Tensor:
        """
        dL/df, via autograd.

        Convenience for solvers that want an explicit gradient rather than managing
        `requires_grad` and `.backward()` themselves. It does not disturb `f`: the
        computation runs on a detached copy.
        """
        x = f.detach().requires_grad_(True)
        (gradient,) = torch.autograd.grad(self.value(x), x)
        return gradient

    # ── proximal interface ───────────────────────────────────────────────────

    def prox_reg(self, f: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """
        Proximal operator of `tau * lambda_reg * R`, for splitting methods (PPXA, ADMM, PnP).

        Returns f unchanged when there is no regularization, which is the mathematically
        correct prox of the zero function — so a solver needs no special case.

        Raises NotImplementedError with an actionable message when the chosen regularizer
        has no prox: that is a real incompatibility (a proximal solver cannot use a
        prox-less prior) and it must surface at setup, not as a wrong result.
        """
        if not self.is_regularized:
            return f
        prox = getattr(self.regularization, "prox", None)
        if prox is None:
            raise NotImplementedError(
                f"Regularization {type(self.regularization).__name__} provides no prox, "
                f"so it cannot be used with a proximal solver. Use a gradient-based solver "
                f"(e.g. ADAM), or choose a regularization that defines prox()."
            )
        return prox(f, tau * self.lambda_reg, self.diff_ops)

    def describe(self) -> str:
        """One-line summary of what is being minimized — for the run log and saved metadata."""
        fidelity = getattr(self.data_fidelity, "display_name", type(self.data_fidelity).__name__)
        if not self.is_regularized:
            return f"L(f) = {fidelity}(Hf, g)"
        reg = getattr(self.regularization, "display_name", type(self.regularization).__name__)
        return f"L(f) = {fidelity}(Hf, g) + {self.lambda_reg:g} * {reg}(f)"

    def __repr__(self) -> str:
        return f"<Objective {self.describe()}>"


def lambda_from_v1(lambda_v1: float) -> float:
    """
    Convert a v1 regularization weight to its v2 equivalent.

    v1 minimized (1 - l) * D + l * R with l in [0, 1); v2 minimizes D + lam * R. Dividing
    the v1 objective by (1 - l) leaves the minimizer unchanged, so lam = l / (1 - l).
    """
    if not 0.0 <= lambda_v1 < 1.0:
        raise ValueError(f"a v1 lambda_reg lies in [0, 1), got {lambda_v1}")
    return lambda_v1 / (1.0 - lambda_v1)
