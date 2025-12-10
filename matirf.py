import torch

from operations import compute_min_max_angles_from_params
from settings import device


def preprocess_matirf_images(g, matirf_params, normalisation=1):
    angles_deg = matirf_params['angles_deg']
    n_angles = len(angles_deg)
    if n_angles != g.shape[0]:
        print(f"\nWarning : while preprocessing number of angles ({n_angles}) "
              f"and number of planes ({g.shape[0]}) do not match.")
        if n_angles > g.shape[0]:
            print(f"effacement des {n_angles - g.shape[0]} derniers angles")
            matirf_params['angles_deg'] = matirf_params['angles_deg'][:g.shape[0]]
            angles_deg = matirf_params['angles_deg']
            n_angles = len(angles_deg)
        else:
            print(f"effacement des {g.shape[0] - n_angles} derniers 'plans'")
            g = g[:n_angles, :, :]
    background = torch.zeros(1, g.shape[1], g.shape[2], device=device)
    g_ = torch.tensor([], device=device)
    angles_ = []
    _, theta_max_deg = compute_min_max_angles_from_params(matirf_params)
    k, n = 0, 0
    for z in range(g.shape[0]):
        if angles_deg[z] > theta_max_deg:
            background += g[z, :, :]
            n += 1
        else:
            g_ = torch.cat((g_, g[z, :, :].unsqueeze(0)), 0)
            angles_.append(angles_deg[z])
            k += 1
    if n != 0: background /= n
    for z in range(len(g_)):
        g_[z] -= background.squeeze()
    matirf_params['angles_deg'] = angles_
    n_angles = k
    if normalisation == 0:  # no normalisation
        g = g_
    elif normalisation == 1:  # min=0, max=1
        g = (g_ - g_.min()) / (g_.max() - g_.min())
    elif normalisation == 2:  # sum of square sqrt = 1
        g = g_ / g_.square().sum().sqrt()
    elif normalisation == 3:  # mean of square sqrt =1
        g = g_ / g_.square().mean().sqrt()
    elif normalisation == 4:  # mean = 1
        g = g_ / g_.mean()
    return g