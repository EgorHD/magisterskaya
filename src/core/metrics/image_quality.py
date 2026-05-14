from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(slots=True)
class ImageQuality:
    # Среднеквадратичная ошибка
    mse: float

    # Пиковое отношение сигнал/шум
    psnr: float

    # Структурное сходство
    ssim: float


# Перевод изображения в grayscale float32
def _to_gray_float(img: Image.Image) -> np.ndarray:
    # Нормализация в RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Преобразование в массив
    arr = np.asarray(img, dtype=np.float32)

    # Приближённый расчёт яркости
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    return gray.astype(np.float32)


# Расчёт MSE
def _mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.mean(diff * diff))


# Расчёт PSNR
def _psnr(mse: float, max_val: float = 255.0) -> float:
    if mse <= 0.0:
        return float("inf")

    return 10.0 * math.log10((max_val * max_val) / mse)


# Упрощённый глобальный SSIM
def _ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    # Средние значения
    mu_x = float(np.mean(a))
    mu_y = float(np.mean(b))

    # Дисперсии и ковариация
    sigma_x2 = float(np.var(a))
    sigma_y2 = float(np.var(b))
    sigma_xy = float(np.mean((a - mu_x) * (b - mu_y)))

    # Константы SSIM
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


# Сравнение двух изображений
def compare_images(original: Image.Image, modified: Image.Image) -> ImageQuality:
    # Если размеры не совпадают, подгоняем modified
    if original.size != modified.size:
        modified = modified.resize(original.size)

    # Перевод в grayscale
    a = _to_gray_float(original)
    b = _to_gray_float(modified)

    # Расчёт метрик
    mse = _mse(a, b)
    psnr = _psnr(mse)
    ssim = _ssim_global(a, b)

    return ImageQuality(mse=mse, psnr=psnr, ssim=ssim)