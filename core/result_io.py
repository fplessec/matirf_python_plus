import os
from os import makedirs
from os.path import join

from base import DataMode
from .reconstruction_result import ReconstructionResult
from .reconstruction_metrics import compute_all_metrics
from in_out import (
    save_tif, save_toml, save_txt, save_csv,
    load_tif, load_txt, load_csv,
)


## saves a ReconstructionResult to disk (tif + config + messages + optional metrics):
def save_reconstruction(result: ReconstructionResult, save_dir: str,
                        config: dict, mode: DataMode) -> None:
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
        # compute metrics at save time if not already computed
        metrics = result.metrics
        if metrics is None:
            metrics = compute_all_metrics(result.f, result.f_true)
            result.metrics = metrics
        save_csv(metrics, join(save_dir, "recons_metrics.csv"))


## loads a ReconstructionResult from a previously saved folder:
def load_reconstruction(directory: str, mode: DataMode) -> ReconstructionResult:
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
