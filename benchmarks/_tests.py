"""
Tests of the campaign runner — the guarantees a multi-hour benchmark relies on.

Run it:
    python -m benchmarks._tests

Each test drives a real miniature campaign (tiny deconvolution runs, a few seconds each):

    SAVED       every finished run has its folder, its status and its journal line
    RESUMED     relaunching skips what is done, and redoes only what failed or was cut
    CONTAINED   a run that crashes or hangs ends as 'failed' / 'timeout'; the rest goes on
    STOPPED     a meaningless reconstruction stops the campaign, then is retried or accepted
"""

import json
import tempfile
from pathlib import Path

from core import DataMode
from problems.deconv import DECONV_MEASUREMENTS_DIR
from benchmarks.runner import RunSpec, run_campaign, CampaignStopped, read_status


def _spec(iterations=40, algorithm="ADAM", png="img_001.png", **extra):
    config = {
        "algorithm": algorithm,
        "input-paths": {"mode": DataMode.SYNTHETIC.value,
                        "png": str(DECONV_MEASUREMENTS_DIR / png),
                        "json": str(DECONV_MEASUREMENTS_DIR / "psf_params_example.json")},
        "add-noise": {"gaussian_noise": True, "sigma": 0.01},
        "noise-model": {"parameters": "unscaled"},
        "algo-params": {"max_iter": iterations, "lr": 0.02, "K": 10, "EPS": 0.0, **extra},
    }
    return RunSpec("deconv", config, tags={"iterations": iterations, **extra})


def _quiet(*args):
    pass


def test_saved_and_resumed():
    directory = Path(tempfile.mkdtemp())
    specs = [_spec(40), _spec(60), _spec(80)]
    counts = run_campaign(specs, directory, isolate=False, log=_quiet)
    assert counts == {"done": 3}, counts

    for spec in specs:
        run_dir = directory / "runs" / spec.run_id
        assert read_status(run_dir)["status"] == "done"
        assert {"config.toml", "f.png", "messages.txt", "metrics.json"} <= \
            {p.name for p in run_dir.iterdir()}
    lines = (directory / "journal.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all("PSNR" in json.loads(l)["metrics"] for l in lines)

    # relaunching the same list does nothing; a longer list does only the new run
    assert run_campaign(specs, directory, isolate=False, log=_quiet) == {}
    assert run_campaign(specs + [_spec(100)], directory, isolate=False, log=_quiet) == {"done": 1}

    # a run cut mid-way (status left at 'running', as after a kill) is redone
    cut = directory / "runs" / specs[1].run_id / "status.json"
    cut.write_text(json.dumps({"status": "running"}))
    assert run_campaign(specs, directory, isolate=False, log=_quiet) == {"done": 1}
    print("  saved/resumed   folder + status + journal per run; relaunch redoes only the cut run")


def test_contained():
    directory = Path(tempfile.mkdtemp())
    broken = _spec(40, png="does_not_exist.png")
    counts = run_campaign([broken, _spec(40)], directory, isolate=True, timeout=120, log=_quiet)
    assert counts == {"failed": 1, "done": 1}, counts
    assert "Traceback" in read_status(directory / "runs" / broken.run_id)["error"] or \
        read_status(directory / "runs" / broken.run_id)["error"]

    hanging = _spec(10_000_000)
    counts = run_campaign([hanging], directory, isolate=True, timeout=8, log=_quiet)
    assert counts == {"timeout": 1}, counts
    print("  contained       a crash -> 'failed', a hang -> 'timeout', the campaign goes on")


def test_stopped_on_rejection():
    """Zero iterations leave f = H^T g: the realism check rejects it ('trivial')."""
    directory = Path(tempfile.mkdtemp())
    stuck = _spec(0)
    good = _spec(60)
    try:
        run_campaign([stuck, good], directory, isolate=False, log=_quiet)
        raise AssertionError("a rejected run must stop the campaign")
    except CampaignStopped as stop:
        assert stop.spec.run_id == stuck.run_id and "trivial" in stop.record["findings"]
    assert read_status(directory / "runs" / good.run_id) == {}, "nothing after the stop"

    # relaunching retries the rejected run (and stops again); accepting it moves on
    try:
        run_campaign([stuck, good], directory, isolate=False, log=_quiet)
        raise AssertionError("still rejected: must stop again")
    except CampaignStopped:
        pass
    counts = run_campaign([stuck, good], directory, isolate=False, accept_rejected=True,
                          log=_quiet)
    assert counts == {"done": 1}, counts
    assert read_status(directory / "runs" / stuck.run_id)["status"] == "rejected"
    print("  stopped         a meaningless run stops the campaign; retried, or accepted and kept")


def main():
    print("benchmarks — the campaign runner\n")
    test_saved_and_resumed()
    test_contained()
    test_stopped_on_rejection()
    print("\nA campaign survives crashes, resumes where it stopped, and stops on nonsense.")


if __name__ == "__main__":
    main()
