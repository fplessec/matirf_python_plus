"""
Sérialisation / désérialisation d'un ReconstructionResult sur disque.

Avant ce refactor, MaTirfPipeline.save_results() et load_results() faisaient
de l'I/O fichier directement. Or sauvegarder un résultat n'a rien à voir avec
orchestrer un calcul : ces deux préoccupations sont séparées ici.

Conventions :
    - le pipeline n'écrit JAMAIS sur disque directement (sauf via ces helpers)
    - le format d'enregistrement est :
        save_dir/
            f.TIF              # reconstruction
            config.toml        # config utilisée pour la run
            messages.txt       # log textuel
            f_true.TIF         # (synthetic only) vérité terrain
            recons_metrics.csv # (synthetic only) métriques de qualité
"""

import os
from os import makedirs
from os.path import join

from .enums import DataMode
from .reconstruction_result import ReconstructionResult
from .reconstruction_metrics import compute_all_metrics
from in_out import (
    save_tif, save_toml, save_txt, save_csv,
    load_tif, load_txt, load_csv,
)


def save_reconstruction(result: ReconstructionResult, save_dir: str,
                        config: dict, mode: DataMode) -> None:
    """
    Sauvegarde un ReconstructionResult sur disque.

    Lève ValueError si le résultat est incomplet (ex: mode synth sans f_true).
    """
    makedirs(save_dir, exist_ok=True)

    if result.f is None:
        raise ValueError("Cannot save results: 'f' is None")

    save_tif(result.f, join(save_dir, 'f.TIF'))
    save_toml(config, join(save_dir, 'config.toml'))
    save_txt(result.messages, join(save_dir, 'messages.txt'))

    if mode == DataMode.SYNTHETIC:
        if result.f_true is None:
            raise ValueError("Synthetic mode but 'f_true' is None")
        save_tif(result.f_true, join(save_dir, 'f_true.TIF'))

        # Si les métriques n'ont pas été calculées au runtime, on les calcule
        # à la sauvegarde (comportement legacy). À terme, devrait être garanti
        # par le pipeline lui-même.
        metrics = result.metrics
        if metrics is None:
            metrics = compute_all_metrics(result.f, result.f_true)
            result.metrics = metrics
        save_csv(metrics, join(save_dir, "recons_metrics.csv"))


def load_reconstruction(directory: str, mode: DataMode) -> ReconstructionResult:
    """
    Charge un ReconstructionResult depuis un dossier précédemment sauvegardé.

    Lève FileNotFoundError si un fichier obligatoire manque.
    Note : H et g ne sont pas sauvegardés (recalculables depuis la config),
    donc ils restent à None après load.
    """
    required = ["config.toml", "f.TIF", "messages.txt"]
    for f in required:
        if not os.path.exists(join(directory, f)):
            raise FileNotFoundError(f)

    result = ReconstructionResult(
        f=load_tif(join(directory, "f.TIF")),
        messages=load_txt(join(directory, "messages.txt")),
    )

    if mode == DataMode.SYNTHETIC:
        result.f_true = load_tif(join(directory, "f_true.TIF"))
        result.metrics = load_csv(join(directory, "recons_metrics.csv"))

    return result
