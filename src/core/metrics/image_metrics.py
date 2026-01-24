from __future__ import annotations

import numpy as np
import cv2
from PIL import Image


def psnr(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(img1.convert("RGB"), dtype=np.float32)
    b = np.array(img2.convert("RGB"), dtype=np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Упрощённый SSIM по яркости (Y), окно Gaussian (11x11).
    """
    a = np.array(img1.convert("RGB"), dtype=np.uint8)
    b = np.array(img2.convert("RGB"), dtype=np.uint8)

    ay = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    by = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(ay, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(by, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(ay * ay, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(by * by, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(ay * by, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))
