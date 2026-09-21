"""
Tests of the synthetic ground-truth generator — what the benchmark relies on.

Run it:
    python -m problems.matirf.synthetic._tests

    REPRODUCIBLE   a scene gives the same truth every time; a TIF's record reproduces it;
                   the v1 truths shipped in data/ are reproduced exactly by this code
    PLAUSIBLE      every preset is a non-negative volume in [0, 1] inside the slab; the cell
                   has an edge and rises from it; adhesions stay at the glass
    NO INVERSE     a truth finer than the reconstruction is averaged exactly, and MA-TIRF
    CRIME          simulates the measurement from the FINE truth, reconstructs on the coarse
                   grid, and refuses a depth range different from the truth's
"""

import tempfile
from pathlib import Path

import torch

from core import DataMode
from core.diagnostics import depth_profile
from problems.matirf import MATIRF, MATIRF_MEASUREMENTS_DIR
from problems.matirf.synthetic import (
    presets, load_preset, generate, save_truth, read_record, regenerate, downsample_z,
    OBJECTS, Grid,
)
from problems.matirf.synthetic.generator import build_objects
from problems.matirf.synthetic.scene import complete


def test_reproducible():
    scene = load_preset("adhesions_fibres")
    assert torch.equal(generate(scene), generate(scene)), "same scene, same truth"
    other = {**scene, "sampling": {"seed": 99}}
    assert not torch.equal(generate(scene), generate(other)), "another seed, another truth"

    out = Path(tempfile.mkdtemp()) / "truth.TIF"
    f = generate(scene)
    save_truth(f, out, scene)
    assert torch.equal(regenerate(out), f), "the record alone reproduces the truth"

    # the truths shipped with the project, made by the v1 generator, reproduced exactly
    for i in (0, 1):
        tif = MATIRF_MEASUREMENTS_DIR / f"synthetic_truth{i}.TIF"
        from fileio import load_tif
        assert torch.equal(regenerate(tif).to(torch.float32), load_tif(str(tif))), tif.name
    assert "cannot be regenerated" in read_record(MATIRF_MEASUREMENTS_DIR / "synthetic_truth2.TIF")["note"]

    empty = complete({"grid": {"nx": 16, "ny": 16, "nz": 10}, "sampling": {"seed": 0},
                      **{key: {"count": 0} for key in OBJECTS}})
    assert float(generate(empty).abs().max()) == 0.0, "no object, no fluorescence"
    print("  reproducible    same scene -> same truth; record reproduces it; v1 truths exact")


def test_presets_are_plausible():
    for name in presets():
        scene = load_preset(name)
        f = generate(scene)
        grid = Grid.from_params(scene["grid"])
        assert f.shape == grid.shape == (150, 128, 128), (name, f.shape)
        assert torch.isfinite(f).all() and float(f.min()) >= 0 and abs(float(f.max()) - 1) < 1e-6
        for obj in build_objects(scene, grid):       # every object reaches into the slab
            x0, x1, y0, y1, z0, z1 = obj.bounds()
            assert z1 > grid.z0_nm and z0 < grid.zN_nm, f"{name}: an object outside the slab"

    # the cell: an edge inside the field, and a membrane rising from edge to centre
    cell = generate(load_preset("cell"))
    footprint = cell.sum(0) > 0.05 * cell.sum(0).max()
    assert 0.2 < float(footprint.float().mean()) < 0.9, "the cell's edge must be in the field"
    membrane_only = generate({**load_preset("cell"), "ellipsoid": {"count": 0}})
    height = membrane_only.argmax(0).float() * 2.0               # nm, 2 nm truth planes
    near_outside = torch.nn.functional.max_pool2d(
        (~footprint).float()[None, None], 9, stride=1, padding=4)[0, 0].bool()
    edge = height[footprint & near_outside].median()             # a band along the edge
    centre = height[footprint & ~torch.nn.functional.max_pool2d(
        (~footprint).float()[None, None], 41, stride=1, padding=20)[0, 0].bool()].median()
    assert float(edge) < float(centre) - 50, "the membrane must rise from its edge"

    # adhesions stay against the glass: most of their fluorescence in the first 100 nm
    adhesions = load_preset("adhesions_fibres")
    adhesions = {**adhesions, "filament": {"count": 0}}
    profile = depth_profile(generate(adhesions))
    assert float(profile[:50].sum()) > 0.9, "adhesions must lie within 100 nm of the glass"
    print(f"  plausible       {len(presets())} presets in [0, 1] inside the slab; the cell has "
          f"an edge and rises {float(centre - edge):.0f} nm; adhesions at the glass")


def test_no_inverse_crime():
    scene = load_preset("vesicles")
    fine = generate(scene)                                            # 150 planes
    coarse = generate({**scene, "grid": {**scene["grid"], "z_oversampling": 1}})   # 50
    averaged = downsample_z(fine, 3)
    averaged, coarse = averaged / averaged.max(), coarse / coarse.max()
    assert float((averaged - coarse).abs().max()) < 0.05, \
        "averaging the fine truth must give the coarse one (voxel means)"

    out = Path(tempfile.mkdtemp()) / "vesicles.TIF"
    save_truth(fine, out, scene)
    config = {"input-paths": {"mode": DataMode.SYNTHETIC.value, "tif": str(out),
                              "json": str(MATIRF_MEASUREMENTS_DIR / "esoubies.json")},
              "oper-params": {"nz": 50, "z0": 0.0, "zN": 300.0, "normalize": False},
              "add-noise": {}}
    assert MATIRF.validate(config) == []
    prepared = MATIRF.prepare(config)
    assert prepared.operator.H.shape == (13, 50) and prepared.f_true.shape == (50, 128, 128)
    assert torch.allclose(prepared.f_true, downsample_z(fine, 3))
    ## g comes from the fine truth, so it differs from H applied to the coarse truth. For
    ## MA-TIRF that model error is tiny at 50 planes (~1e-4: H varies slowly in depth, far
    ## below any realistic noise) and grows on coarser grids (~3e-3 to 8e-3 at 10 planes)
    from core.normalization import normalize
    direct = normalize(prepared.operator.apply(prepared.f_true))
    gap = float((prepared.g - direct).norm() / direct.norm())
    assert 0 < gap < 0.05, f"simulation gap {gap:.2e}"

    wrong = {**config, "oper-params": {**config["oper-params"], "zN": 400.0}}
    assert any("generated with zN = 300" in e for e in MATIRF.validate(wrong))
    print(f"  no inverse crime  fine truth averaged exactly; g from the fine truth "
          f"({gap:.1e} from H f_coarse); a wrong depth range is refused")


def main():
    print("problems.matirf.synthetic — the ground-truth generator\n")
    test_reproducible()
    test_presets_are_plausible()
    test_no_inverse_crime()
    print("\nThe synthetic truths are reproducible, plausible, and free of inverse crime.")


if __name__ == "__main__":
    main()
