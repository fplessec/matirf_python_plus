import torch
from pathlib import Path

from algorithms.denoisers import denoise_gaussian, denoise_bilateral, denoise_wiener, denoise_dct
from algorithms.denoisers.tests import load_2d_png_image, add_noise, psnr


def main():
    torch.manual_seed(0)

    data_dir = Path(__file__).parent / "data"
    images = list(data_dir.glob("*.png"))

    sigma = 0.1

    results = {
        "gaussian": [],
        "bilateral": [],
        "wiener": [],
        "dct": []
    }

    for path in images:
        x = load_2d_png_image(path)
        y = add_noise(x, sigma)

        xg = denoise_gaussian(y, sigma)
        xb = denoise_bilateral(y, sigma)
        xw = denoise_wiener(y, sigma)
        xd = denoise_dct(y, sigma)

        results["gaussian"].append(psnr(x, xg).item())
        results["bilateral"].append(psnr(x, xb).item())
        results["wiener"].append(psnr(x, xw).item())
        results["dct"].append(psnr(x, xd).item())

    print(f"\nResults (sigma={sigma}):")
    for k, v in results.items():
        print(f"{k:10s}: {sum(v)/len(v):.2f} dB")


if __name__ == "__main__":
    main()