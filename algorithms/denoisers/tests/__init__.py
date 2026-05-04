from PIL import Image
import torch
import numpy as np


def load_2d_png_image(path):
    img = Image.open(path).convert("L")  # grayscale
    x = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    return x.unsqueeze(0)  # (1, H, W)

def add_noise(x, sigma):
    return x + sigma * torch.randn_like(x)

def psnr(x, x_hat):
    mse = torch.mean((x - x_hat) ** 2)
    return 10 * torch.log10(1.0 / mse)


