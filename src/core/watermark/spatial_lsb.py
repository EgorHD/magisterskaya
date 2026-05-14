from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

from core.models.ocr import OCRPageResult
from core.watermark.codec import bits_to_bytes, bytes_to_bits


@dataclass(slots=True)
class SpatialConfig:
    # Число бит на один span
    bits_per_span: int = 32

    # Канал для записи: 0=R 1=G 2=B
    channel: int = 2

    # Ключ для генерации seed
    seed_key: str = "HWM-SPATIAL"


# Генерация seed для конкретного span
def _span_seed(cfg: SpatialConfig, page_index: int, span_index: int) -> int:
    s = f"{cfg.seed_key}|p={page_index}|i={span_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")


# Встраивание payload в области OCR-слов
def embed_spatial(
    image: Image.Image,
    ocr_page: OCRPageResult,
    payload: bytes,
    cfg: SpatialConfig,
) -> Image.Image:
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    # Перевод payload в биты
    bits = bytes_to_bits(payload)

    # Сколько spans нужно для записи
    need_spans = (len(bits) + cfg.bits_per_span - 1) // cfg.bits_per_span
    spans = ocr_page.spans[:need_spans]

    bit_pos = 0

    for si, sp in enumerate(spans):
        if bit_pos >= len(bits):
            break

        x1, y1, x2, y2 = sp.bbox()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(arr.shape[1], x2), min(arr.shape[0], y2)

        # Слишком маленькие области пропускаем
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        seed = _span_seed(cfg, ocr_page.page_index, si)
        rng = np.random.default_rng(seed)

        # Количество бит для текущего span
        count = min(cfg.bits_per_span, len(bits) - bit_pos)

        # Случайные точки внутри bbox
        ys = rng.integers(y1, y2, size=count)
        xs = rng.integers(x1, x2, size=count)

        ch = cfg.channel
        for k in range(count):
            b = bits[bit_pos]
            px = arr[ys[k], xs[k], ch]
            arr[ys[k], xs[k], ch] = (px & 0xFE) | b
            bit_pos += 1

    return Image.fromarray(arr, mode="RGB")


# Извлечение payload из областей OCR-слов
def extract_spatial(
    image: Image.Image,
    ocr_page: OCRPageResult,
    payload_bits_len: int,
    cfg: SpatialConfig,
) -> bytes:
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    bits_out: List[int] = []

    # Сколько spans нужно для чтения
    spans_need = (payload_bits_len + cfg.bits_per_span - 1) // cfg.bits_per_span
    spans = ocr_page.spans[:spans_need]

    for si, sp in enumerate(spans):
        if len(bits_out) >= payload_bits_len:
            break

        x1, y1, x2, y2 = sp.bbox()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(arr.shape[1], x2), min(arr.shape[0], y2)

        # Слишком маленькие области пропускаем
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        seed = _span_seed(cfg, ocr_page.page_index, si)
        rng = np.random.default_rng(seed)

        remaining = payload_bits_len - len(bits_out)
        count = min(cfg.bits_per_span, remaining)

        # Те же случайные точки внутри bbox
        ys = rng.integers(y1, y2, size=count)
        xs = rng.integers(x1, x2, size=count)

        ch = cfg.channel
        for k in range(count):
            bits_out.append(int(arr[ys[k], xs[k], ch] & 1))

    # Добивка до кратности 8
    while len(bits_out) % 8 != 0:
        bits_out.append(0)

    return bits_to_bytes(bits_out[: ((payload_bits_len + 7) // 8) * 8])