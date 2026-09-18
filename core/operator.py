"""
The forward operator H of an inverse problem — all of the physics, in one object.

An inverse problem asks: given a measurement g, recover f such that H f ~ g. Everything a
solver needs to know about the physics is therefore "what does H do, and what can I do
with it". This class is that contract.

--------------------------------------------------------------------------------------
Why this object exists (the central design decision of v2)
--------------------------------------------------------------------------------------

In v1 a solver received a raw tensor H and had to know how to use it: MA-TIRF multiplied
matrices, deconvolution convolved via FFT. So every problem re-declared every algorithm
just to supply `apply_forward` / `apply_adjoint` — about 470 lines whose only content was
plumbing, plus a mixin per problem.

Here the solver never sees H. It receives a ForwardOperator and calls `apply` / `adjoint` /
`ridge_inverse`. The physics lives in ONE file per problem, and a solver is written once,
for all problems. Adding an inverse problem therefore requires no algorithm code at all.

--------------------------------------------------------------------------------------
What a concrete operator must provide
--------------------------------------------------------------------------------------

Only two methods are abstract:

    apply(f)      ->  H f        map a reconstruction to a measurement
    adjoint(y)    ->  H^T y      map a measurement back to reconstruction space

Everything else has a working default built from those two, so a new problem is usable
immediately:

    gram(f)               H^T H f
    ridge_inverse(y, lam) (H^T H + lam I)^-1 H^T y   by conjugate gradient
    lipschitz(like)       ||H^T H||, by power iteration

Override a default only when the structure of H gives a better route — that is a genuine
optimization, not a requirement:

    > MA-TIRF   H is a small dense matrix  -> ridge_inverse in closed form (torch.inverse)
    > deconv    H is a convolution         -> ridge_inverse as a Wiener filter in Fourier

Both defaults are deliberately matrix-free: they only ever call `apply` and `adjoint`, so
they work even when H is never formed explicitly.
"""

from abc import ABC, abstractmethod

import torch

from .features import NO_FEATURES


class ForwardOperator(ABC):
    """
    The forward model of one inverse problem, for one configuration.

    An instance is bound to a specific measurement geometry (angles, PSF, grid, ...); it is
    built by the problem from a config, then handed to the solver. It holds no solver state
    and no measurement — only the physics.

    ----------
    > Class attributes :
    ----------

    >> name : str
        Short identifier used in messages and saved metadata, e.g. "matirf".

    >> features : frozenset[Feature]
        The properties this operator induces on the problem (see core/features.py), e.g.
        {THREE_D, ANISOTROPIC, SCALE_AMBIGUOUS}. They are a fact about the physics, which is
        exactly why they are declared here rather than restated by every algorithm.

    ----------
    > To implement :
    ----------

    >> apply(f) -> y
        Compute H f. `f` lives in reconstruction space, the result in measurement space.

    >> adjoint(y) -> f
        Compute H^T y, the exact transpose of `apply`. Solvers rely on this being the true
        adjoint: <H f, y> == <f, H^T y>. `check_adjoint()` below verifies it numerically,
        and it is the first thing to run when a new operator misbehaves.

    ----------
    > Example :
    ----------

        class DeconvOperator(ForwardOperator):
            name = "deconv"
            features = features(Feature.TWO_D)

            def __init__(self, psf):
                self.psf = psf

            def apply(self, f):    return convolve(f, self.psf)
            def adjoint(self, y):  return convolve(y, self.psf, adjoint=True)
    """

    name: str = ""
    features: frozenset = NO_FEATURES

    # ── the two methods that define the physics ──────────────────────────────

    @abstractmethod
    def apply(self, f: torch.Tensor) -> torch.Tensor:
        """H f : reconstruction space -> measurement space."""

    @abstractmethod
    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """H^T y : measurement space -> reconstruction space. Must be the true adjoint."""

    # ── derived operations, matrix-free by default ───────────────────────────

    def gram(self, f: torch.Tensor) -> torch.Tensor:
        """H^T H f — the normal operator, which drives every least-squares method."""
        return self.adjoint(self.apply(f))

    def ridge_inverse(self, y: torch.Tensor, lam: float = 0.0,
                      *, n_iter: int = 50, tol: float = 1e-6) -> torch.Tensor:
        """
        Solve the ridge (Tikhonov-regularized least squares) problem

            argmin_x  ||H x - y||^2 + lam ||x||^2        i.e.  (H^T H + lam I) x = H^T y

        Used as a warm-start initialization and as the data-consistency step of MCMC.
        `lam` > 0 stabilizes an ill-conditioned H; the larger it is, the more the result is
        pulled toward zero.

        The default is conjugate gradient, which needs only `apply` and `adjoint` — so it
        works for any operator, including ones too large to form explicitly. It converges
        in at most rank(H) iterations, and `tol` stops it early once the residual is small.

        Override when H has structure that gives a direct solve (see the module docstring).
        """
        b = self.adjoint(y)
        x = torch.zeros_like(b)
        r = b.clone()                       # residual b - A x, with x = 0
        p = r.clone()                       # search direction
        rs = torch.dot(r.flatten(), r.flatten())
        if rs.sqrt() <= tol:                # H^T y is already ~0: x = 0 is the solution
            return x
        for _ in range(n_iter):
            Ap = self.gram(p) + lam * p     # A p, with A = H^T H + lam I
            pAp = torch.dot(p.flatten(), Ap.flatten())
            if pAp <= 0:                    # A not positive definite here: stop safely
                break
            alpha = rs / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            rs_next = torch.dot(r.flatten(), r.flatten())
            if rs_next.sqrt() <= tol:
                break
            p = r + (rs_next / rs) * p
            rs = rs_next
        return x

    def lipschitz(self, like: torch.Tensor, *, n_iter: int = 50, tol: float = 1e-6) -> float:
        """
        Largest eigenvalue of H^T H, by power iteration.

        This is the Lipschitz constant of the gradient of ||H f - g||^2 / 2, so it is what
        sets a safe step size for gradient and proximal methods (step <= 1 / L). Computing
        it frees the user from hand-tuning a learning rate per problem.

        `like` is any tensor shaped like a reconstruction — it is only used for the shape
        and the device/dtype of the random starting vector.
        """
        x = torch.randn_like(like)
        x = x / x.norm()
        value = 0.0
        for _ in range(n_iter):
            Hx = self.gram(x)
            norm = Hx.norm()
            if norm == 0:
                return 0.0
            x = Hx / norm
            previous, value = value, float(norm)
            if abs(value - previous) <= tol * max(value, 1.0):
                break
        return value

    # ── self-check ───────────────────────────────────────────────────────────

    def check_adjoint(self, like_f: torch.Tensor, like_y: torch.Tensor,
                      *, tol: float = 1e-4) -> float:
        """
        Verify numerically that `adjoint` really is the transpose of `apply`.

        Draws random f and y and compares <H f, y> with <f, H^T y>; they are equal for a
        true adjoint. Returns the relative error and raises AssertionError beyond `tol`.

        Worth running once for every new operator: a wrong adjoint does not crash, it
        silently makes every gradient-based solver converge to the wrong answer — the most
        expensive class of bug in this codebase.
        """
        f = torch.randn_like(like_f)
        y = torch.randn_like(like_y)
        left = torch.dot(self.apply(f).flatten(), y.flatten())
        right = torch.dot(f.flatten(), self.adjoint(y).flatten())
        scale = max(abs(float(left)), abs(float(right)), 1e-12)
        error = abs(float(left - right)) / scale
        assert error <= tol, (
            f"{type(self).__name__}.adjoint is not the transpose of .apply "
            f"(<Hf, y> = {float(left):.6g} but <f, H^t y> = {float(right):.6g}, "
            f"relative error {error:.2e})"
        )
        return error

    def __repr__(self) -> str:
        feats = ", ".join(sorted(str(f) for f in self.features)) or "none"
        return f"<{type(self).__name__} name={self.name!r} features={{{feats}}}>"
