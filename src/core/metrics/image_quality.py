from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from PIL import Image


@dataclass(slots=True)
class ImageQuality:
    mse: float
    psnr: float
    ssim: float


def _to_gray_float(img: Image.Image) -> np.ndarray:
    """
    Переводим в оттенки серого и float32 [0..255].
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    # luminance approx
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return gray.astype(np.float32)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.mean(diff * diff))


def _psnr(mse: float, max_val: float = 255.0) -> float:
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((max_val * max_val) / mse)


def _ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    """
    Упрощённый SSIM по всему изображению (global SSIM).
    Для исследовательских метрик обычно достаточно и стабильно.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    mu_x = float(np.mean(a))
    mu_y = float(np.mean(b))

    sigma_x2 = float(np.var(a))
    sigma_y2 = float(np.var(b))
    sigma_xy = float(np.mean((a - mu_x) * (b - mu_y)))

    L = 255.0
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)
    if den == 0:
        return 0.0
    return float(num / den)


def compare_images(original: Image.Image, modified: Image.Image) -> ImageQuality:
    """
    Возвращает MSE/PSNR/SSIM между исходной и модифицированной страницей.
    Если размеры разные — приводим modified к размеру original (мягко, чтобы не падало).
    """
    if original.size != modified.size:
        modified = modified.resize(original.size)

    a = _to_gray_float(original)
    b = _to_gray_float(modified)

    mse = _mse(a, b)
    psnr = _psnr(mse)
    ssim = _ssim_global(a, b)
    return ImageQuality(mse=mse, psnr=psnr, ssim=ssim)