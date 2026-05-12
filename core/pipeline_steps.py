"""
Étapes de calcul du pipeline MA-TIRF, encapsulées en fonctions pures.

Une fonction pure ici signifie :
    - inputs explicites en paramètres (pas de lecture de globales/state)
    - retour explicite (pas de mutation de state externe)
    - I/O fichier acceptée si elle est inhérente à l'étape (load_tif, load_json)

Ces fonctions sont appelées par MaTirfPipeline.setup(), mais peuvent aussi être
réutilisées par les widgets GUI qui font de l'aperçu (TifFilePreprocessViewer,
DifferenceViewer) — c'est précisément à ça qu'elles servent : permettre au GUI
de ne plus orchestrer du calcul lui-même.
"""

from typing import Optional, Tuple

import torch

from .enums import DataMode
from .operations import (
    compute_matirf_operator_from_params,
    apply_matirf_operator,
    estimate_delta_anisotropy_from_params,
)
from .preprocess_measurement import preprocess_measurement_stack
from .reconstruction_metrics import compute_all_metrics, optimal_scale
from in_out import load_tif, load_json


# ---------------------------------------------------------------------------
# 1. Chargement des inputs
# ---------------------------------------------------------------------------
def load_measurement_inputs(config: dict) -> Tuple[dict, dict, dict]:
    """
    Charge les paramètres de mesure depuis le fichier JSON et extrait les
    sous-dictionnaires de config nécessaires au reste du pipeline.

    Retourne (measurement_params, oper_params, add_noise_params).
    """
    measurement_params = load_json(config['input-paths']['json'])
    oper_params = config['oper-params']
    add_noise_params = config['add-noise']
    return measurement_params, oper_params, add_noise_params


# ---------------------------------------------------------------------------
# 2. Construction de g et H selon le mode (real vs synthetic)
# ---------------------------------------------------------------------------
def build_g_and_H_real(config: dict) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Mode REAL : on charge la mesure expérimentale, on la préprocesse et on
    construit l'opérateur H à partir des paramètres mis à jour.

    Retourne (g_preprocessed, H, measurement_params_updated).
    """
    measurement_params, oper_params, add_noise_params = load_measurement_inputs(config)
    g_raw = load_tif(config['input-paths']['tif'])
    g, measurement_params = preprocess_measurement_stack(
        g_raw, measurement_params, add_noise_params
    )
    H = compute_matirf_operator_from_params(measurement_params, oper_params)
    return g, H, measurement_params


def build_g_and_H_synthetic(config: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mode SYNTHETIC : on charge f_true, on construit H_synth, puis on simule g par
    H_synth @ f_true et on lui applique le même preprocessing que pour des données
    réelles.

    H_synth est entièrement déterminé par les fichiers .tif (donne nz = f_true.shape[0])
    et .json (donne n_angles). On NE consomme PAS oper_params['nz'] ici : peu importe
    ce que l'utilisateur a saisi dans la GUI, on respecte la dimensionnalité réelle
    du fichier TIF synthétique. Pas d'assert possible donc, pas de risque de crash.

    Retourne (g_preprocessed, H_synth, f_true).
    """
    measurement_params, oper_params, add_noise_params = load_measurement_inputs(config)
    f_true = load_tif(config['input-paths']['tif'])
    nz_from_tif = f_true.shape[0]

    # On respecte z0 et zN de l'utilisateur (ils définissent la grille physique)
    # mais on remplace nz par celui du TIF (sinon shape mismatch H @ f_true).
    # Copie pour ne pas muter la config originale.
    synth_oper_params = dict(oper_params)
    synth_oper_params['nz'] = nz_from_tif

    H_synth = compute_matirf_operator_from_params(measurement_params, synth_oper_params)
    g = apply_matirf_operator(H_synth, f_true)
    g, _ = preprocess_measurement_stack(g, measurement_params, add_noise_params)
    return g, H_synth, f_true


# ---------------------------------------------------------------------------
# 3. Évaluation des métriques (mode synth)
# ---------------------------------------------------------------------------
def evaluate_synthetic_metrics(f: torch.Tensor, f_true: torch.Tensor,
                                config: dict) -> Tuple[dict, float]:
    """
    Calcule delta (anisotropie) et toutes les métriques de qualité.

    Retourne (metrics_dict, delta).
    """
    measurement_params, oper_params, _ = load_measurement_inputs(config)
    delta = estimate_delta_anisotropy_from_params(measurement_params, oper_params)
    metrics = compute_all_metrics(f, f_true, delta=delta)
    return metrics, delta


# ---------------------------------------------------------------------------
# 4. Différence f_true - alpha*f (visualisation mode synth)
# ---------------------------------------------------------------------------
def compute_synthetic_difference(f: torch.Tensor, f_true: torch.Tensor,
                                  config: dict) -> Tuple[torch.Tensor, float]:
    """
    Calcule la différence visualisable f_true - alpha*f où alpha minimise
    la distance entre f et f_true.

    Retourne (diff, alpha).
    """
    measurement_params, oper_params, _ = load_measurement_inputs(config)
    delta = estimate_delta_anisotropy_from_params(measurement_params, oper_params)
    alpha = optimal_scale(
        f.detach().cpu().numpy(),
        f_true.detach().cpu().numpy(),
        delta=delta,
    )
    diff = f_true - alpha * f
    return diff, alpha


# ---------------------------------------------------------------------------
# 5. Aperçu du préprocessing (utilisé par TifFilePreprocessViewer dans la GUI)
# ---------------------------------------------------------------------------
def compute_preprocessing_preview(config: dict, mode: DataMode) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calcule les deux images affichées par TifFilePreprocessViewer.

    Mode REAL :
        Retourne (raw_measurement, preprocessed_measurement).
        L'utilisateur compare le .tif brut (vraie mesure MA-TIRF) avec sa
        version après preprocessing + ajout de bruit.

    Mode SYNTHETIC :
        Retourne (f_true, g_synth_preprocessed).
        L'utilisateur compare le .tif brut (vérité terrain f_true) avec la
        simulation g_synth = H @ f_true après preprocessing + ajout de bruit.
    """
    if mode == DataMode.REAL:
        measurement_params, _, add_noise_params = load_measurement_inputs(config)
        raw = load_tif(config['input-paths']['tif'])
        preprocessed, _ = preprocess_measurement_stack(
            raw, measurement_params, add_noise_params
        )
        return raw, preprocessed

    # mode == DataMode.SYNTHETIC
    # build_g_and_H_synthetic charge f_true, calcule H et g_synth = H @ f_true,
    # puis applique le preprocessing à g_synth (preprocess + noise).
    g_synth, _, f_true = build_g_and_H_synthetic(config)
    return f_true, g_synth
