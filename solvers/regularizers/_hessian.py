"""
The Hessian norm shared by two regularizations — Hessian-Frobenius and SHV.

--------------------------------------------------------------------------------------
The definition
--------------------------------------------------------------------------------------

At every pixel x, the Hessian H f(x) is the symmetric matrix of second derivatives. Its
Frobenius norm is

    2D   ||H f(x)||_F = sqrt( fyy^2 + fxx^2 + 2 fxy^2 )
    3D   ||H f(x)||_F = sqrt( fxx^2 + fyy^2 + fzz^2 + 2 fxy^2 + 2 fxz^2 + 2 fyz^2 )

(the off-diagonal terms appear twice because the matrix is symmetric). The standard
second-order regularizer of image processing and inverse problems is the SUM of that norm
over all pixels — a mixed L1-L2 norm, the second-order counterpart of total variation:

    R_HF(f) = sum_x ||H f(x)||_F

It is the Hessian-Schatten norm of order 2 of Lefkimmiatis, Bourquard and Unser
("Hessian-based norm regularization for image restoration", IEEE TIP 2012 and 2013).
Being L1 across pixels, it favours images whose Hessian is zero almost everywhere — that
is, piecewise-LINEAR images — without the staircasing total variation produces.

The Sparse Hessian Variation of SPITFIRe (Prigent et al.) adds a sparsity term inside the
same square root, with a weight w in [0, 1]:

    R_SHV(f) = sum_x sqrt( w^2 ||H f(x)||_F^2 + (1 - w)^2 f(x)^2 )

so Hessian-Frobenius is exactly SHV with w = 1. Both are implemented here once.

--------------------------------------------------------------------------------------
The discretisation — copied from SPITFIRe, stencil for stencil
--------------------------------------------------------------------------------------

Second derivatives are centred ([-1, 2, -1]); mixed derivatives are FORWARD differences
(f[i+1,j+1] - f[i+1,j] - f[i,j+1] + f[i,j]); the axial direction is scaled by delta
(delta^2 on fzz, delta on fxz and fyz). Everything is evaluated on the interior of the
image only, with no padding — as in SPITFIRe's `hv_loss` / `hv_loss_3d`, which
`solvers/regularizers/_tests.py` checks against a verbatim copy.

The operator L used below stacks the terms whose squares are summed:

    L f = ( w * fxx, w * fyy, [w * fzz], w*sqrt2 * fxy, [w*sqrt2 * fxz, w*sqrt2 * fyz],
            (1 - w) * f )

so that R(f) = sum_x || (L f)(x) ||_2. The sparsity component is dropped when w = 1.

--------------------------------------------------------------------------------------
The proximal operator — why the old one had to go
--------------------------------------------------------------------------------------

prox_{lam R}(f) = argmin_u  1/2 ||u - f||^2 + lam * sum_x ||(L u)(x)||_2

R is non-smooth (it is an L1 norm across pixels), so its prox cannot be a few gradient
steps. It is computed by the standard dual method — the same one Chambolle's algorithm
uses for total variation (see tv.py), with the Hessian in place of the gradient:

    u = f - lam * L^T p,      p projected onto { ||p(x)||_2 <= 1 } at every pixel
    p <- Proj( p + (1 / (lam ||L||^2)) * L u )

The step 1 / (lam ||L||^2) is safe because ||L||^2 is bounded analytically below.

Both L and its adjoint L^T are derived from ONE table of stencils (`_stencils`): each
component is a weighted sum of shifted views of f, so L gathers those views and L^T
scatters the dual variable back through the same shifts. Writing the adjoint by hand,
and accumulating the forward operator straight into the dual variable, takes the prox
from 5.6 s (autograd adjoint) to about 1 s on a 50x350x350 volume — and the adjoint cannot
drift from the forward operator because both read the same table.
`_tests.py` still certifies it against autograd, where L^T p = d/du <L u, p> exactly.
"""

import math

import torch

## Inside the square root of the LOSS only: sqrt(0) has an infinite derivative, and
## positivity-projected reconstructions are exactly zero in large regions, so without it
## the gradient would be NaN there. SPITFIRe itself has no epsilon; this one changes the
## value by at most sqrt(1e-12) = 1e-6 per pixel.
EPS = 1e-12

SQRT2 = math.sqrt(2.0)


def _stencils(ndim: int, delta: float, weight: float) -> list:
    """
    The components of L as (scale, [(coefficient, slices), ...]) — the single source of truth.

    A component evaluates to  scale * sum(coefficient * f[slices]).  The centre is
    f[1:-1, ...]; slices shift it by one voxel. The table reproduces SPITFIRe exactly.
    """
    S, w = slice, weight
    s2 = SQRT2
    if ndim == 2:
        c = (S(1, -1), S(1, -1))
        table = [
            (w,      [(-1, (S(2, None), S(1, -1))), (2, c), (-1, (S(None, -2), S(1, -1)))]),  # fyy
            (w,      [(-1, (S(1, -1), S(2, None))), (2, c), (-1, (S(1, -1), S(None, -2)))]),  # fxx
            (w * s2, [(1, (S(2, None), S(2, None))), (-1, (S(2, None), S(1, -1))),
                      (-1, (S(1, -1), S(2, None))), (1, c)]),                                  # fxy
        ]
    elif ndim == 3:
        c = (S(1, -1), S(1, -1), S(1, -1))
        I, P, M = S(1, -1), S(2, None), S(None, -2)    # interior, plus one, minus one
        table = [
            (w,               [(-1, (I, I, P)), (2, c), (-1, (I, I, M))]),                    # fxx
            (w,               [(-1, (I, P, I)), (2, c), (-1, (I, M, I))]),                    # fyy
            (w * delta ** 2,  [(-1, (P, I, I)), (2, c), (-1, (M, I, I))]),                    # fzz
            (w * s2,          [(1, (I, P, P)), (-1, (I, I, P)), (-1, (I, P, I)), (1, c)]),    # fxy
            (w * s2 * delta,  [(1, (P, I, P)), (-1, (I, I, P)), (-1, (P, I, I)), (1, c)]),    # fxz
            (w * s2 * delta,  [(1, (P, P, I)), (-1, (I, P, I)), (-1, (P, I, I)), (1, c)]),    # fyz
        ]
    else:
        raise ValueError(f"Hessian regularization needs a 2D or 3D image, got {ndim}D")
    if weight < 1.0:
        table.append((1.0 - weight, [(1, c)]))                                              # sparsity
    return table


def components(f: torch.Tensor, delta: float, weight: float) -> list:
    """
    The terms of L f, on the interior of f — differentiable (used by the loss).

    Their squares sum to w^2 ||H f||_F^2 + (1 - w)^2 f^2 at every interior pixel.
    """
    terms = []
    for scale, stencil in _stencils(f.dim(), delta, weight):
        coefficient, where = stencil[0]
        term = coefficient * f[where]
        for coefficient, where in stencil[1:]:
            term = term + coefficient * f[where]
        terms.append(scale * term)
    return terms


def norm_map(f: torch.Tensor, delta: float, weight: float) -> torch.Tensor:
    """sqrt( sum of squared components ) at every interior pixel — differentiable."""
    terms = components(f, delta, weight)
    squared = terms[0] * terms[0]
    for term in terms[1:]:
        squared = squared + term * term
    return torch.sqrt(squared + EPS)


def operator_norm_squared(ndim: int, delta: float, weight: float) -> float:
    """
    An upper bound on ||L||^2, which sets a safe step for the dual iteration.

    A centred second difference [-1, 2, -1] has norm at most 4 (squared: 16); a forward
    mixed difference is a product of two forward differences of norm at most 2 each
    (squared: 16), doubled by the sqrt2 factor (32). Stacking operators adds their squared
    norms. In 2D this gives the 64 w^2 found in the Hessian-regularization literature.
    """
    w2 = weight * weight
    if ndim == 2:
        hessian = 16 + 16 + 32
    else:
        d2 = delta * delta
        hessian = 16 + 16 + 16 * d2 * d2 + 32 + 32 * d2 + 32 * d2
    return w2 * hessian + (1.0 - weight) ** 2


def adjoint(p: list, like: torch.Tensor, delta: float, weight: float) -> torch.Tensor:
    """
    L^T p: scatter each dual component back through the shifts it was gathered from.

    Read from the same table as `components`, so the two cannot disagree; certified against
    autograd in `_tests.py`. Not differentiable (in-place) — only the prox calls it.
    """
    out = torch.zeros_like(like)
    for (scale, stencil), pi in zip(_stencils(like.dim(), delta, weight), p):
        for coefficient, where in stencil:
            out[where].add_(pi, alpha=scale * coefficient)
    return out


def prox(f: torch.Tensor, lam: float, delta: float, weight: float,
         n_iter: int) -> torch.Tensor:
    """
    argmin_u 1/2 ||u - f||^2 + lam * sum_x ||(L u)(x)||_2, by projected dual iterations.

    `lam` weights the SUM over pixels, as in the TV prox (tv.py). See the module docstring
    for the algorithm.

    Written for speed, since it runs inside every iteration of a proximal solver: the dual
    update p += step * L u is accumulated straight into p from shifted views of u, so the
    forward operator never materialises a single intermediate volume, and the projection
    and the adjoint work in place. Nothing here is differentiated.
    """
    if lam <= 0:
        return f.clone()
    f = f.detach()
    table = _stencils(f.dim(), delta, weight)
    step = 1.0 / (lam * operator_norm_squared(f.dim(), delta, weight))
    interior = f[(slice(1, -1),) * f.dim()]
    p = [torch.zeros_like(interior) for _ in table]
    magnitude = torch.empty_like(interior)

    with torch.no_grad():
        for _ in range(n_iter):
            u = adjoint(p, f, delta, weight)
            u.mul_(-lam).add_(f)                               # u = f - lam * L^T p
            for (scale, stencil), pi in zip(table, p):         # p += step * L u
                for coefficient, where in stencil:
                    pi.add_(u[where], alpha=step * scale * coefficient)
            torch.mul(p[0], p[0], out=magnitude)               # project onto ||p(x)|| <= 1
            for pi in p[1:]:
                magnitude.addcmul_(pi, pi)
            magnitude.sqrt_().clamp_(min=1.0)
            for pi in p:
                pi.div_(magnitude)
        u = adjoint(p, f, delta, weight)
        return u.mul_(-lam).add_(f)
