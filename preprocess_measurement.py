import torch

from operations import compute_min_max_angles_from_params
from settings import device


def preprocess_measurement_stack(g, measurement_params, add_noise_params, normalization=1):
    angles_deg = measurement_params['angles_deg']
    n_angles, n_stacks = len(angles_deg), g.shape[0]
    assert n_angles == n_stacks, (
        f"\nWhile preprocessing the MA-TIRF measurement:\nThe number of angles ({n_angles}) and the number of stacks "
        f"({n_stacks}) do not match."
    )
    # if a stack correspond to an angle higher than the maximum angle of the TIRF microscope,
    # then the corresponding stack is an image of the background
    background_list = []  # <-to store each background stack
    g_list = []  # <-to store each non-background stacks
    _, theta_max_deg = compute_min_max_angles_from_params(measurement_params)
    for i in range(n_stacks):
        if angles_deg[i] > theta_max_deg:  # <-background
            background_list.append(g[i, :, :])
            angles_deg[i] = 'background-angle'
            print(f"stack {i} is background ({angles_deg[i]}° > {theta_max_deg}° the maximum angle)")
        else:
            g_list.append(g[i, :, :])
    if background_list:  # <-background_list is not empty: we have background stack(s)
        # we can do the mean of the stacks higher than the max angle to estimate the background:
        estimated_background = torch.stack(background_list, dim=0).to(device).mean(dim=0).unsqueeze(0)
        # then we just have to subtract the estimated background from the useful stacks:
        g = torch.stack(g_list, dim=0).to(device) - estimated_background
        angles_deg = [angle for angle in angles_deg if angle != 'background-angle']  # <-only keep the useful angles
        measurement_params['angles_deg'] = angles_deg  # <-update the measurement parameters with the non-background
                                                       # angles to compute the right the MA-TIRF operator
    # we normalize the resulting MA-TIRF stacks to complete the preprocessing:
    g = normalize_measurement(g, normalization=normalization)
    # and finally we can add noise is the user specified it:
    if add_noise_params['add_noise']:
        g = add_noise_to_measurement(g, add_noise_params)
    return g, measurement_params

def normalize_measurement(g, normalization=1):
    if normalization == 0:  # no normalisation
        pass
    elif normalization == 1:  # min=0, max=1
        g = (g - g.min()) / (g.max() - g.min())
    elif normalization == 2:  # sum of square sqrt = 1
        g = g / g.square().sum().sqrt()
    elif normalization == 3:  # mean of square sqrt =1
        g = g / g.square().mean().sqrt()
    elif normalization == 4:  # mean = 1
        g = g / g.mean()
    return g

def add_noise_to_measurement(g, add_noise_params):
    if add_noise_params['is_gaussian']:
        noise = torch.randn_like(g) * add_noise_params['sigma']
        g_noisy = g + noise
    else:
        raise Exception('To be implemented')
    return g_noisy

