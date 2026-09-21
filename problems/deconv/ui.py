"""
The deconvolution interface, declared.

Compare with problems/matirf/ui.py: the same shape, far less content. Deconvolution has one
figure view instead of two, one Estimate button instead of two, and no operator-parameter
section — its only parameters live in the PSF .json. The framework does not force a simple
problem to carry machinery it has no use for.

Note what is absent and still works: `delta` is never mentioned here, because it requires
the ANISOTROPIC feature and this problem is 2D — the framework hides it with no code.
"""

from gui.reusable import FrequencyCutoffDialog
from gui.spec import Editor, Estimator, FileSlot, Preview, ProblemUI, View, image_panel
from gui.widgets import ImageAndHisto2DViewer

from .parameters import PSF_PARAMETERS_UI


# ── the preprocessing preview ─────────────────────────────────────────────────

PREVIEW_TITLES = {
    "real": ("g_raw = input image (raw measurement)", "g = preprocessed g_raw + noise"),
    "synthetic": ("f_true = input image (ground truth)", "g_synth = H * f_true (+ noise)"),
}


def _preview(config, mode):
    """The problem previews itself; see problems/matirf/ui.py for why."""
    from problems.deconv import DECONV
    return DECONV.preview(config)


def _preview_errors(config):
    from core import noise
    return noise.validate(config.get("add-noise", {}))


# ── the Estimate button ───────────────────────────────────────────────────────
# The PSF's spectrum decides which frequencies are recoverable: below a given magnitude a
# frequency is drowned in noise and dividing by it amplifies that noise without bound. The
# dialog plots the spectrum and lets the user pick the cutoff.

def _spectrum(operator, config):
    from fileio import load_png
    image = load_png(config["input-paths"]["png"])
    return operator.spectrum_magnitudes(image.shape)


ESTIMATE_LAMBDA_RR = Estimator(
    compute=_spectrum,
    tooltip="Estimate lambda_rr from the Fourier spectrum of the PSF.\n"
            "Displays the sorted |H_fft| values and lets you\n"
            "choose which frequency to use as cutoff.",
    picker=FrequencyCutoffDialog,
    title="Estimate lambda_rr from the PSF spectrum",
    info=lambda operator, config: f"PSF kernel: {tuple(operator.psf.shape)}",
)


# ── the declaration ───────────────────────────────────────────────────────────

DECONV_UI = ProblemUI(
    control_title="Deconvolution Parameter Selection",
    display_title="Display Window",
    real_mode_text="Work with real measurement",
    synthetic_mode_text="Simulate with synthetic truth",

    image_slot=FileSlot(
        toml_key="png", noun="png", dialog_filter="Image Files (*.png)",
        title_real="Path of the 2D image (measurement)",
        title_synthetic="Path of the 2D ground truth",
        preview=Preview(ImageAndHisto2DViewer, _preview, titles=PREVIEW_TITLES,
                        size=(900, 900), error_check=_preview_errors),
    ),
    json_slot=FileSlot(
        toml_key="json", noun="json", dialog_filter="Parameters Files (*.json)",
        title_real="Path of the PSF parameters",
        title_synthetic="Path of the simulation parameters",
        editor=Editor(ui=PSF_PARAMETERS_UI, title_noun="PSF Parameters"),
    ),

    views=[View("reconstruction", [image_panel(ImageAndHisto2DViewer, title="")])],

    difference_viewer=ImageAndHisto2DViewer,
    estimators={"lambda_rr": ESTIMATE_LAMBDA_RR},
)
