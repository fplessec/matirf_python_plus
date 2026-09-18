"""
Seeded random helpers — the single, portable source of randomness for the package.

Everything is driven by a ``torch.Generator`` so that one seed reproduces a ground
truth bit-for-bit. The generator lives on the CPU (portable across cpu / cuda / mps);
tensors are only moved to the compute device once the objects are built.
"""

import math

import torch


def uniform(gen: torch.Generator, lo: float, hi: float, size=()) -> torch.Tensor:
    """Uniform sample(s) in [lo, hi) from the seeded generator 'gen' (CPU, float64)."""
    return lo + (hi - lo) * torch.rand(size, generator=gen, dtype=torch.float64)


def signed_uniform(gen: torch.Generator, lo: float, hi: float) -> float:
    """Uniform magnitude in [lo, hi) with a random sign — used to remove directional bias."""
    mag = uniform(gen, lo, hi).item()
    return mag if torch.rand((), generator=gen).item() < 0.5 else -mag


def randint(gen: torch.Generator, n: int) -> int:
    """A single integer in [0, n) drawn from the seeded generator."""
    return int(torch.randint(0, n, (1,), generator=gen).item())


# ── rotations (torch, float64, CPU) ──────────────────────────────────────────

def _Rx(a): return torch.tensor([[1, 0, 0], [0, math.cos(a), -math.sin(a)],
                                 [0, math.sin(a), math.cos(a)]], dtype=torch.float64)
def _Ry(a): return torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                                 [-math.sin(a), 0, math.cos(a)]], dtype=torch.float64)
def _Rz(a): return torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0],
                                 [0, 0, 1]], dtype=torch.float64)


def random_flat_rotation(gen: torch.Generator, max_tilt_deg: float = 0.0) -> torch.Tensor:
    """
    Rotation that keeps the third (axial) principal axis close to z: a free in-plane
    rotation about z, composed with an optional small out-of-plane tilt (<= max_tilt_deg).

    This is the orientation appropriate for the flat MA-TIRF slab — an ovaloid can point
    anywhere in the (x, y) plane but stays essentially flat in z, never poking out of the
    shallow (<= ~500 nm) reconstructable depth.
    """
    R = _Rz(uniform(gen, 0.0, 2 * math.pi).item())
    if max_tilt_deg > 0:
        t = max_tilt_deg * math.pi / 180.0
        R = _Rx(signed_uniform(gen, 0.0, t)) @ _Ry(signed_uniform(gen, 0.0, t)) @ R
    return R


def random_rotation_matrix(gen: torch.Generator) -> torch.Tensor:
    """
    A uniformly distributed 3x3 rotation (element of SO(3)) — a FREE 3D orientation.

    Kept for completeness; not used for MA-TIRF ovaloids because a free rotation would
    let an elongated axis poke out of the thin slab. Uses the QR trick.
    """
    a = torch.randn(3, 3, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q
