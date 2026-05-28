"""
Étapes de calcul du pipeline de déconvolution 2D, encapsulées en fonctions pures.

Même rôle que core/pipeline_steps.py de matirf :
    - inputs explicites en paramètres (pas de lecture de globales/state)
    - retour explicite (pas de mutation de state externe)
    - I/O fichier acceptée si elle est inhérente à l'étape (load_png, load_json)

Ces fonctions sont appelées par DeconvPipeline.setup(), mais peuvent aussi être
réutilisées par les widgets GUI qui font de l'aperçu (preview du preprocessing,
différence f_true - f).
"""

from typing import Tuple

import torch

from . import DataMode, add_noise_to_measurement
from .operations import compute_psf_from_params, apply_psf
from .metrics import compute_all_metrics
from deconv.in_out import load_png, load_json


# ---------------------------------------------------------------------------
# 1. Chargement des inputs (paramètres PSF + bruit)
# ---------------------------------------------------------------------------
def load_measurement_inputs(config: dict) -> Tuple[dict, dict]:
    """
    Charge les paramètres de PSF depuis le fichier JSON et extrait les
    paramètres d'ajout de bruit depuis la config.

    Retourne (psf_params, add_noise_params).
    """
    psf_params = load_json(config['input-paths']['json'])
    add_noise_params = config['add-noise']
    return psf_params, add_noise_params


# ---------------------------------------------------------------------------
# 2. Construction de g et H selon le mode (real vs synthetic)
# ---------------------------------------------------------------------------
def build_g_and_H_real(config: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mode REAL : on charge l'image PNG (déjà floutée par l'optique), on construit
    la PSF à partir du JSON, et on applique éventuellement du bruit en plus.

    Retourne (g, H).
    """
    psf_params, add_noise_params = load_measurement_inputs(config)
    g = load_png(config['input-paths']['png'])
    g = add_noise_to_measurement(g, add_noise_params)
    H = compute_psf_from_params(psf_params)
    return g, H


def build_g_and_H_synthetic(config: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mode SYNTHETIC : on charge f_true (image nette), on construit H, puis on
    simule la mesure floutée g = H @ f_true et on lui ajoute éventuellement
    du bruit pour reproduire des conditions réalistes.

    Retourne (g, H, f_true).
    """
    psf_params, add_noise_params = load_measurement_inputs(config)
    f_true = load_png(config['input-paths']['png'])
    H = compute_psf_from_params(psf_params)
    g = apply_psf(H, f_true)
    g = add_noise_to_measurement(g, add_noise_params)
    return g, H, f_true


# ---------------------------------------------------------------------------
# 3. Évaluation des métriques (mode synth)
# ---------------------------------------------------------------------------
def evaluate_synthetic_metrics(f: torch.Tensor, f_true: torch.Tensor) -> dict:
    """
    Calcule toutes les métriques de qualité (PSNR, SSIM, MSE).

    Pas de delta à estimer en 2D contrairement à MA-TIRF : l'image reconstruite
    et la vérité terrain sont dans le même espace (mêmes pixels, même résolution).
    """
    return compute_all_metrics(f, f_true)


# ---------------------------------------------------------------------------
# 4. Différence f_true - f (visualisation mode synth)
# ---------------------------------------------------------------------------
def compute_synthetic_difference(f: torch.Tensor, f_true: torch.Tensor) -> torch.Tensor:
    """
    Calcule la différence visualisable f_true - f. Contrairement à matirf
    on n'a pas besoin d'un facteur d'échelle alpha : f et f_true sont déjà
    dans la même échelle (toutes les deux dans [0, 1] après load_png).
    """
    return f_true - f


# ---------------------------------------------------------------------------
# 5. Aperçu du préprocessing (utilisé par la GUI de deconv)
# ---------------------------------------------------------------------------
def compute_preprocessing_preview(config: dict, mode: DataMode) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcule les deux images affichées par le viewer d'aperçu.

    Mode REAL :
        Retourne (raw_png, png_after_noise).
        L'utilisateur compare l'image PNG brute (vraie mesure floutée) avec
        sa version après ajout de bruit.

    Mode SYNTHETIC :
        Retourne (f_true, g_synth_noisy).
        L'utilisateur compare l'image nette (vérité terrain f_true) avec la
        simulation g = H @ f_true après ajout de bruit.
    """
    if mode == DataMode.REAL:
        _, add_noise_params = load_measurement_inputs(config)
        raw = load_png(config['input-paths']['png'])
        preprocessed = add_noise_to_measurement(raw, add_noise_params)
        return raw, preprocessed

    # mode == DataMode.SYNTHETIC
    g_synth, _, f_true = build_g_and_H_synthetic(config)
    return f_true, g_synth
