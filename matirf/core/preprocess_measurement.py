import torch

from common.add_noise import add_noise_to_measurement
from .operations import compute_min_max_angles_from_params
import common.settings as settings
import matirf.settings as matirf_settings


def preprocess_measurement_stack(g, measurement_params, add_noise_params, normalization=matirf_settings.normalization):
    angles_deg = measurement_params['angles_deg']
    n_angles, n_stacks = len(angles_deg), g.shape[0]
    assert n_angles == n_stacks, (
        f"\nWhile preprocessing the MA-TIRF measurement:\nThe number of angles ({n_angles}) and the number of stacks "
        f"({n_stacks}) do not match.\n\n"
        f"Please check that the number of stacks in the .tif matches "
        f"the number of angles declared in the .json measurement parameters."
    )
    # if a stack correspond to an angle higher than the maximum angle of the TIRF microscope,
    # then the corresponding stack is an image of the background
    background_list = []  # <-to store each background stack
    g_list = []  # <-to store each non-background stacks
    _, theta_max_deg = compute_min_max_angles_from_params(measurement_params)
    background_index_list = []
    for i in range(n_stacks):
        if angles_deg[i] > theta_max_deg:  # <-background
            background_list.append(g[i, :, :])
            background_index_list.append(i)
            print(f"stack {i+1} is background ({angles_deg[i]}° > {theta_max_deg}° the maximum angle)")
            angles_deg[i] = 'background-angle'
        else:
            g_list.append(g[i, :, :])
    if background_list:  # <-background_list is not empty: we have background stack(s)
        # we can do the mean of the stacks higher than the max angle to estimate the background:
        estimated_background = torch.stack(background_list, dim=0).to(settings.device).mean(dim=0).unsqueeze(0)
        # then we just have to subtract the estimated background from the useful stacks:
        g = torch.stack(g_list, dim=0).to(settings.device) - estimated_background
        angles_deg = [angle for angle in angles_deg if angle != 'background-angle']  # <-only keep the useful angles
        measurement_params['angles_deg'] = angles_deg  # <-update the measurement parameters with the non-background
                                                       # angles to compute the right the MA-TIRF operator
        print(f"stack(s) {','.join([str(i+1) for i in background_index_list])} have been used to estimate background and have been removed.")
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
    elif normalization == 3:  # mean of square sqrt = 1
        g = g / g.square().mean().sqrt()
    elif normalization == 4:  # mean = 1
        g = g / g.mean()
    return g


