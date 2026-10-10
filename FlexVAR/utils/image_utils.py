import math
import os

import numpy as np
import torch
from PIL import Image


def _normalize_to_uint8(image: torch.Tensor, value_range=(-1, 1)) -> torch.Tensor:
    image = image.detach().float().cpu()
    if value_range is not None:
        lo, hi = value_range
        if hi <= lo:
            raise ValueError(f'invalid value_range={value_range!r}')
        image = image.clamp(lo, hi)
        image = (image - lo) / (hi - lo)
    else:
        image = image.clamp(0, 1)
    return (image * 255.0).round().to(torch.uint8)


def save_tensor_image(image: torch.Tensor, path: str, value_range=(-1, 1)):
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f'Expected CHW image tensor, got shape={tuple(image.shape)}')
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = _normalize_to_uint8(image, value_range=value_range).permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


def save_tensor_grid(images: torch.Tensor, path: str, nrow: int = 4, value_range=(-1, 1)):
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(f'Expected BCHW tensor, got shape={tuple(images.shape)}')

    images = images.detach().float().cpu()
    if images.shape[1] == 1:
        images = images.expand(-1, 3, -1, -1)

    n = int(images.shape[0])
    nrow = max(1, min(int(nrow), n))
    ncol = math.ceil(n / nrow)
    c, h, w = images.shape[1:]
    grid = torch.zeros((c, ncol * h, nrow * w), dtype=torch.uint8)
    scaled = _normalize_to_uint8(images, value_range=value_range)

    for idx in range(n):
        row, col = divmod(idx, nrow)
        grid[:, row * h:(row + 1) * h, col * w:(col + 1) * w] = scaled[idx]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = grid.permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)

