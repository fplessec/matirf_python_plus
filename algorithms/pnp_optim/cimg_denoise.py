import time
import torch

from os.path import join, dirname
from os import remove
import subprocess

from in_out import save_tif, load_tif
from settings import device


def denoise_Cimg(f, sigma, denoiser, erase_files_after=True):
    """
    denoise: denoise 2D+T images corrupted by Gaussian or Poisson noise : denoise -i InputImage.tif -o outputImage -algo AlgorithmName
 Build :  (Feb  6 2025, 15:50:08)

    -i               0                        Input image
    -o               0                        Output file
    -first           0                        Number of the first image (0: default value)
    -last            -1                       Number of the last image (depth or time: default value)

    -alpha           0                        alpha mixing of input/output images [0. - 1.] (0.: default value)
    -scale           1                        Resize the volume in the range [0.5 - 1.5] (1.: default value)
    -range           1                        Automatic intensity scaling (-1) or manual scaling

    -algo            0                        Algorithm name:
                                              - BM3D
                                              - NLBayes
                                              - NLMeans
                                              - BayesNLmeans
                                              - SAFIR
                                              - PEWA
                                              - OWF
                                              - DCT
                                              - Wiener
                                              - Bilateral
                                              - Gaussian
                                              - Median
                                              - TV
                                              - SV
                                              - HV

    Noise parameters and simulation:
    -ng              0                        Add artificial Gaussian noise before aplying the algorithm
    -np              false                    Add artificial Poisson noise before applying the algorithm
    -msg             0                        Adjust manually the assumed Gaussian noise standard deviation
    -stab            false                    Variance stabilization for Poisson noise removal

    Options and parameters of denoising algorithms:
    -patch           3                        Half size of the patch (NLMeans, PEWA, OWF, SAFIR, DCT, Wiener)
    -neigh           7                        Half size of the neighborhood (NLMeans, PEWA, OWF, SAFIR, DCT, Median, Bilateral)
    -denoisep        -1                       Denoising parameter (NLMeans: 3.5 | DCT: 3.0 | Wiener: 1.25 | Bilateral: 2.0 | Gaussian: 1.0 | TV: 6.0 | SV: 6.0 | HV: 6.0)
    -sparsep         0.6                      Sparsity parameter (SV and HV algorithms) in the range [0.1 - 0.9]
    -iter            4                        Number of iterations (NDSafir only)
    """

    cimgdenoise_dir = join(dirname(__file__), "cimgdenoise")
    executable = join(cimgdenoise_dir, "./denoise")

    unique_name = ''.join(str(time.time()).split('.'))
    input_image = join(cimgdenoise_dir, f"{unique_name}.TIF")
    denoised_output = join(cimgdenoise_dir, f"{unique_name}_denoised.TIF")
    sigma = f"{sigma}"

    # commande qui execute le code c++ compilé
    command = [
        executable,
        '-i',
        input_image,
        '-o',
        denoised_output,
        '-msg',
        sigma,
        '-algo',
        denoiser
    ]

    save_tif(f, input_image)
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    f_denoised = load_tif(denoised_output)

    if erase_files_after:
        remove(input_image)
        remove(denoised_output)

    return f_denoised



if __name__ == "__main__":

    f = torch.rand(30, 120, 160, device=device)

    f_denoised = denoise_Cimg(f, 10, 'BM3D', erase_files_after=False)