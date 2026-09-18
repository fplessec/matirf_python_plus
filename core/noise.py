import torch


def add_noise_to_measurement(g, add_noise_params):
    """Add noise to measurement g based on the noise parameters.

    Supports three noise models:
        - Gaussian: g + sigma * N(0,1)
        - Poisson: Poisson(g)  (g must be non-negative)
        - Poisson-Gaussian: Poisson(g) + sigma * N(0,1)

    The noise type is determined by 'is_gaussian' (legacy bool) or
    'noise_type' (string matching data fidelity names).
    """
    noise_type = add_noise_params.get('noise_type', None)
    if noise_type is None:
        noise_type = 'gaussian' if add_noise_params.get('is_gaussian', True) else 'gaussian'

    sigma = add_noise_params.get('sigma', 0.0)

    if noise_type == 'gaussian':
        return g + sigma * torch.randn_like(g)
    elif noise_type == 'poisson':
        return torch.poisson(g.clamp(min=0))
    elif noise_type == 'poisson-gaussian':
        return torch.poisson(g.clamp(min=0)) + sigma * torch.randn_like(g)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
