import torch
import tifffile

from .utils import _fetch
from settings import device, dtype


def load_tif(filepath) -> torch.tensor:
    """Load an .tif file (located at filepath) and returns a pytorch tensor."""
    image = tifffile.imread(_fetch(filepath))
    tensor = torch.tensor(image, dtype=dtype, device=device)
    return tensor


def save_tif(image: torch.tensor, filepath):
    """Save a 1, 2 or 3D image (named and located after filepath). Ensures a 'ZYX' (or 'YX'/'X') format."""
    if image.dtype != dtype: image = image.to(dtype)
    axes_imagej = 'ZYX' if len(image.shape) == 3 else 'YX' if len(image.shape) == 2 else 'X'
    tifffile.imwrite(filepath, image.detach().to('cpu').numpy(),
                     imagej=True, metadata={'axes': axes_imagej})
