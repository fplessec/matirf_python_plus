import time
import subprocess

from os.path import join, dirname
from os import remove, getcwd
import torch
from torchvision.utils import save_image
from torchvision.io import read_image

from settings import device


def denoise_NL_BAYES(f, sigma, erase_files_after=True):

    f_denoised = torch.zeros_like(f, device=device)

    nl_bayes_dir = join(dirname(__file__), "nl_bayes_dir")
    executable = join(nl_bayes_dir, "./NL_Bayes")

    unique_name = ''.join(str(time.time()).split('.'))
    input_image = join(nl_bayes_dir, f"{unique_name}.png")
    sigma = f"{sigma}"
    add_noise = "0"
    noisy_output = join(nl_bayes_dir, f"{unique_name}ImNoisy.png")
    denoised_output = join(nl_bayes_dir, f"{unique_name}Denoised.png")
    basic_output = join(nl_bayes_dir, f"{unique_name}Basic.png")
    diff_output = join(nl_bayes_dir, f"{unique_name}Diff.png")
    bias_output = join(nl_bayes_dir, f"{unique_name}Bias.png")
    bias_basic_output = join(nl_bayes_dir, f"{unique_name}BiasBasic.png")
    diff_bias_output = join(nl_bayes_dir, f"{unique_name}DiffBias.png")
    use_area1 = "1"
    use_area2 = "0"
    compute_bias = "0"

    # NL bayes c'est du c++
    command = [
        executable,
        input_image,
        sigma,
        add_noise,
        noisy_output,
        denoised_output,
        basic_output,
        diff_output,
        bias_output,
        bias_basic_output,
        diff_bias_output,
        use_area1,
        use_area2,
        compute_bias,
    ]

    for z in range(f.shape[0]):

        f_plan = f[z, :, :]
        mean = f_plan.mean().item()
        std = f_plan.std().item()

        save_image(f_plan, input_image)
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        f_denoised_plan_from_png = read_image(denoised_output).to(f.dtype)
        mean_png = f_denoised_plan_from_png.mean().item()
        std_png = f_denoised_plan_from_png.std().item()

        # on veut conserver la distribution de l'image bruitée originale :
        f_denoised_plan = (f_denoised_plan_from_png - mean_png) / std_png * std + mean
        f_denoised[z, :, :] = f_denoised_plan

    if erase_files_after:
        remove(join(getcwd(), "measures.txt"))
        remove(input_image)
        remove(denoised_output)
        remove(basic_output)
        remove(diff_output)
        if add_noise == "1":
            remove(noisy_output)
        if compute_bias == "1":
            remove(bias_output)
            remove(bias_basic_output)
            remove(diff_bias_output)

    return f_denoised



if __name__ == "__main__":

    f = torch.rand(30, 120, 160, device=device)

    f_denoised = denoise_NL_BAYES(f, 10)