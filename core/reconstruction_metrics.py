# core/reconstruction_metrics.py

import numpy as np

METRICS = {}


def register_metric(name):
    def decorator(func):
        METRICS[name] = func
        return func
    return decorator


@register_metric("MSE")
def mse(f, f_true):
    return (f - f_true).square().mean().item()


@register_metric("MAE")
def mae(f, f_true):
    return (f - f_true).abs().mean().item()


def compute_all_metrics(f, f_true):
    results = {}

    for name, func in METRICS.items():
        try:
            results[name] = func(f, f_true)
        except Exception as e:
            results[name] = f"Error: {e}"

    return results