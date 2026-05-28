"""
Sérialisation / désérialisation d'un DeconvResult sur disque.

Même rôle que core/result_io.py de matirf, mais on sauvegarde des PNG plutôt
que des TIF, et il n'y a pas de delta à stocker.

Format d'enregistrement :
    save_dir/
        f.png              # image déconvoluée
        config.toml        # config utilisée pour la run
        messages.txt       # log textuel
        f_true.png         # (synthetic only) vérité terrain
        deconv_metrics.csv # (synthetic only) métriques de qualité
"""

import os
from os import makedirs
from os.path import join

from . import DataMode
from .deconv_result import DeconvResult
from .metrics import compute_all_metrics
from deconv.in_out import (
    save_png, save_toml, save_txt, save_csv,
    load_png, load_txt, load_csv,
)


def save_deconv(result: DeconvResult, save_dir: str,
                config: dict, mode: DataMode) -> None:
    """
    Sauvegarde un DeconvResult sur disque.

    Lève ValueError si le résultat est incomplet (ex: mode synth sans f_true).
    """
    makedirs(save_dir, exist_ok=True)

    if result.f is None:
        raise ValueError("Cannot save results: 'f' is None")

    save_png(result.f, join(save_dir, 'f.png'))
    save_toml(config, join(save_dir, 'config.toml'))
    save_txt(result.messages, join(save_dir, 'messages.txt'))

    if mode == DataMode.SYNTHETIC:
        if result.f_true is None:
            raise ValueError("Synthetic mode but 'f_true' is None")
        save_png(result.f_true, join(save_dir, 'f_true.png'))

        # Si les métriques n'ont pas été calculées au runtime, on les calcule
        # à la sauvegarde (comportement legacy). À terme, devrait être garanti
        # par le pipeline lui-même.
        metrics = result.metrics
        if metrics is None:
            metrics = compute_all_metrics(result.f, result.f_true)
            result.metrics = metrics
        save_csv(metrics, join(save_dir, "deconv_metrics.csv"))


def load_deconv(directory: str, mode: DataMode) -> DeconvResult:
    """
    Charge un DeconvResult depuis un dossier précédemment sauvegardé.

    Lève FileNotFoundError si un fichier obligatoire manque.
    Note : H et g ne sont pas sauvegardés (recalculables depuis la config),
    donc ils restent à None après load.
    """
    required = ["config.toml", "f.png", "messages.txt"]
    for f in required:
        if not os.path.exists(join(directory, f)):
            raise FileNotFoundError(f)

    result = DeconvResult(
        f=load_png(join(directory, "f.png")),
        messages=load_txt(join(directory, "messages.txt")),
    )

    if mode == DataMode.SYNTHETIC:
        result.f_true = load_png(join(directory, "f_true.png"))
        result.metrics = load_csv(join(directory, "deconv_metrics.csv"))

    return result
