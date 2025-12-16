import torch

from operations import compute_min_max_angles_from_params
from settings import device


def preprocess_measurement_stack(g, measurement_params, normalization=4):
    angles_deg = measurement_params['angles_deg']
    g, angles_deg = check_number_of_angles_and_number_of_stacks(g, angles_deg)
    n_angles, n_stacks = len(angles_deg), g.shape[0]
    # if a stack correspond to an angle higher than the maximum angle of the TIRF microscope,
    # then the corresponding stack is an image of the background
    background_list = []  # <-to store each background stack
    g_list = []  # <-to store each non-background stacks
    _, theta_max_deg = compute_min_max_angles_from_params(measurement_params)
    for i in range(n_stacks):
        if angles_deg[i] > theta_max_deg:
            background_list.append(g[i, :, :])
            angles_deg[i] = 'background-angle'
            print(f"stack {i} is background ({angles_deg[i]}° > {theta_max_deg}° the maximum angle)")
        else:
            g_list.append(g[i, :, :])
    # we can do the mean of the stacks higher than the max angle to estimate the background:
    estimated_background = torch.stack(background_list, dim=0).to(device).mean(dim=0).unsqueeze(0)
    # then we just have to subtract the estimated background from the useful stacks:
    g = torch.stack(g_list, dim=0).to(device) - estimated_background
    angles_deg = [angle for angle in angles_deg if angle != 'background-angle']  # <-only keep the useful angles
    measurement_params['angles_deg'] = angles_deg  # <-update the measurement parameters with the non-background angles
                                                   #  so we can use the right parameters to compute the MA-TIRF operator
    ### print("angles_deg:\n", angles_deg)
    ### print("measurement_params:\n", measurement_params)
    ### print(f"estimated_background.shape : {estimated_background.shape}")
    ### print(f"g.shape : {g.shape}")
    # finally we normalize the resulting MA-TIRF stacks to complete the preprocessing:
    g = normalize_measurement(g, normalization=normalization)
    return g, measurement_params

def check_number_of_angles_and_number_of_stacks(g, angles_deg):
    n_angles = len(angles_deg)
    n_stacks = g.shape[0]
    if n_angles != n_stacks:  # the angles and the stacks do not match
        print(f"\nWarning : while preprocessing the MA-TIRF measurement stack:\n"
              f" > number of angles ({n_angles}) and number of stacks ({g.shape[0]}) do not match.")
        if n_angles > n_stacks:  # if more angles than stacks
            print(f" > removal of the {n_angles - n_stacks} last angle(s).")
            angles_deg = angles_deg[:n_stacks]
        else:  # if more stacks than angles
            print(f" > removal of the {g.shape[0] - n_angles} last stack(s).")
            g = g[:n_angles, :, :]
    return g, angles_deg

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
