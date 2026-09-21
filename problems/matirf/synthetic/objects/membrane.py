"""
Membrane — the basal membrane of an adherent cell: the typical MA-TIRF object.

MA-TIRF is used above all to measure how far a cell's membrane sits from the glass. So the
membrane here is not an infinite sheet but the underside of ONE cell:

    footprint    a smooth, irregular outline (a circle deformed by three random harmonics,
                 `irregularity`), with a soft edge of width `edge_width_nm` — the cell ends
    height       close to the glass at the edge (`height_edge_nm`, the lamella, adhesions)
                 and higher under the cell body (`height_center_nm`), rising smoothly
                 (parabolically) from one to the other
    corrugation  two random waves of amplitude `corrugation_amp_nm` and wavelength
                 `corrugation_period_nm` on top: membranes are never flat
    thickness    a Gaussian profile across the membrane, `thickness_nm` its standard
                 deviation — a labelled membrane is a few nm thick, so this mostly sets how
                 sharp an axial structure the reconstruction is asked to resolve

Real cells are 20-50 um across; the field here is usually smaller, so `footprint_radius_nm`
is best set to show the edge of the cell inside the field (the defaults do, on 12.8 um).
"""

import math

import torch

import settings as settings
from ..rng import uniform, signed_uniform
from .base import SyntheticObject, REACH, value_param, count_param


class Membrane(SyntheticObject):
    """amplitude * footprint(x, y) * exp(-0.5 ((z - height(x, y)) / thickness)^2)."""

    name = "Membrane (adherent cell)"
    toml_key = "membrane"
    ui_params = {
        "count": count_param("cells", default=0),
        "footprint_radius_nm": value_param("Cell footprint radius", "R", 4500.0, "nm"),
        "irregularity": value_param("Outline irregularity (0-0.5)", "\\epsilon", 0.25),
        "edge_width_nm": value_param("Edge softness", "w", 400.0, "nm"),
        "height_edge_nm": value_param("Height at the cell edge", "h_{edge}", 40.0, "nm"),
        "height_center_nm": value_param("Height under the cell body", "h_{centre}", 160.0, "nm"),
        "corrugation_amp_nm": value_param("Corrugation amplitude", "a", 20.0, "nm"),
        "corrugation_period_nm": value_param("Corrugation wavelength", "\\lambda_c", 2000.0, "nm"),
        "thickness_nm": value_param("Thickness (profile std)", "\\tau", 20.0, "nm"),
        "amplitude": value_param("Peak amplitude", "A", 1.0),
    }

    def __init__(self, center_nm, radius_nm, harmonics, edge_width_nm, height_edge_nm,
                 height_center_nm, waves, thickness_nm, amplitude=1.0):
        self.center_nm, self.radius_nm, self.harmonics = center_nm, radius_nm, harmonics
        self.edge_width_nm, self.thickness_nm, self.amplitude = edge_width_nm, thickness_nm, amplitude
        self.height_edge_nm, self.height_center_nm, self.waves = height_edge_nm, height_center_nm, waves
        lateral = radius_nm * (1 + sum(abs(a) for a, _, _ in harmonics)) + REACH * edge_width_nm
        corrugation = sum(abs(a) for a, _, _ in waves)
        low = min(height_edge_nm, height_center_nm) - corrugation - REACH * thickness_nm
        high = max(height_edge_nm, height_center_nm) + corrugation + REACH * thickness_nm
        cx, cy = center_nm
        self._bounds = (cx - lateral, cx + lateral, cy - lateral, cy + lateral, low, high)

    @classmethod
    def sample(cls, p, gen, grid):
        cx = uniform(gen, 0.4 * grid.Lx_nm, 0.6 * grid.Lx_nm).item()
        cy = uniform(gen, 0.4 * grid.Ly_nm, 0.6 * grid.Ly_nm).item()
        irregularity = min(max(float(p["irregularity"]), 0.0), 0.5)
        harmonics = [(signed_uniform(gen, 0.0, irregularity / k) if irregularity else 0.0,
                      k, uniform(gen, 0.0, 2 * math.pi).item()) for k in (2, 3, 4)]
        period, amp = float(p["corrugation_period_nm"]), float(p["corrugation_amp_nm"])
        waves = []
        for _ in range(2):
            direction = uniform(gen, 0.0, math.pi).item()
            waves.append((amp / 2, (math.cos(direction) / period, math.sin(direction) / period),
                          uniform(gen, 0.0, 2 * math.pi).item()))
        return cls(center_nm=(cx, cy), radius_nm=float(p["footprint_radius_nm"]),
                   harmonics=harmonics, edge_width_nm=float(p["edge_width_nm"]),
                   height_edge_nm=float(p["height_edge_nm"]),
                   height_center_nm=float(p["height_center_nm"]),
                   waves=waves, thickness_nm=float(p["thickness_nm"]),
                   amplitude=float(p["amplitude"]))

    def bounds(self):
        return self._bounds

    def density(self, X, Y, Z):
        dx, dy = X - self.center_nm[0], Y - self.center_nm[1]
        rho, phi = torch.sqrt(dx * dx + dy * dy), torch.atan2(dy, dx)
        outline = torch.full_like(rho, self.radius_nm)
        for a, k, phase in self.harmonics:
            outline = outline + self.radius_nm * a * torch.cos(k * phi + phase)
        footprint = 0.5 * (1 + torch.tanh((outline - rho) / self.edge_width_nm))
        inside = (rho / outline).clamp(max=1.0)
        height = self.height_edge_nm + (self.height_center_nm - self.height_edge_nm) * (1 - inside ** 2)
        for a, (kx, ky), phase in self.waves:
            height = height + a * torch.sin(2 * math.pi * (kx * X + ky * Y) + phase)
        return self.amplitude * footprint * torch.exp(-0.5 * ((Z - height) / self.thickness_nm) ** 2)
