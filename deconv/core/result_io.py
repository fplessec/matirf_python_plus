import os
from os import makedirs
from os.path import join

from base import DataMode
from .deconv_result import DeconvResult
from .metrics import compute_all_metrics
from deconv.in_out import (
    save_png, save_toml, save_txt, save_csv,
    load_png, load_txt, load_csv,
)


## saves a DeconvResult to disk (png + config + messages + optional metrics):
def save_deconv(result: DeconvResult, save_dir: str,
                config: dict, mode: DataMode) -> None:
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
        # compute metrics at save time if not already computed
        metrics = result.metrics
        if metrics is None:
            metrics = compute_all_metrics(result.f, result.f_true)
            result.metrics = metrics
        save_csv(metrics, join(save_dir, "deconv_metrics.csv"))


## loads a DeconvResult from a previously saved folder:
def load_deconv(directory: str, mode: DataMode) -> DeconvResult:
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
