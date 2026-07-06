import numpy as np
import torch
from PIL import Image

from common.in_out.utils import _fetch
import common.settings as settings


def load_png(filepath) -> torch.Tensor:
    """
    Load an 2D grayscale image .png file (located at filepath) and returns a pytorch tensor.
    """
    img = Image.open(str(_fetch(filepath))).convert('L')  # 'L' = 8-bit grayscale
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize in [0, 1]
    tensor = torch.tensor(arr, dtype=settings.dtype, device=settings.device)
    return tensor


def save_png(image: torch.Tensor, filepath) -> None:
    """
    Save a 2D or (3D image 1xHxW), named and located after filepath.
    """
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.ndim != 2:
        raise ValueError(f"save_png needs a 2d image, whereas the received input image shape  is{tuple(image.shape)}")
    arr = image.detach().to('cpu').clamp(0.0, 1.0).numpy()
    arr_u8 = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(arr_u8, mode='L').save(str(filepath))