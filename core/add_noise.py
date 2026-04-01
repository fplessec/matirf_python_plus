import torch


def add_noise_to_measurement(g, add_noise_params):
    if add_noise_params['is_gaussian']:
        noise = torch.randn_like(g) * add_noise_params['sigma']
        g_noisy = g + noise
    else:
        raise Exception('To be implemented')
    return g_noisy
