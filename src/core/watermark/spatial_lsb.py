from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from core.models.ocr import OCRPageResult, OCRSpan
from core.watermark.codec import bytes_to_bits, bits_to_bytes


@dataclass(slots=True)
class SpatialConfig:
    bits_per_span: int = 32        # сколько бит на одно слово (span)
    channel: int = 2               # 0=R 1=G 2=B
    seed_key: str = "HWM-SPATIAL"  # ключ PRNG


def _span_seed(cfg: SpatialConfig, page_index: int, span_index: int) -> int:
    s = f"{cfg.seed_key}|p={page_index}|i={span_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")


def embed_spatial(image: Image.Image, ocr_page: OCRPageResult, payload: bytes, cfg: SpatialConfig) -> Image.Image:
    """
    Встраивает payload в области слов, распределяя биты по spans.
    Если payload не влезает, лишнее обрезается (для прототипа).
    """
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    bits = bytes_to_bits(payload)
    need_spans = (len(bits) + cfg.bits_per_span - 1) // cfg.bits_per_span

    spans = ocr_page.spans[:need_spans]
    bit_pos = 0

    for si, sp in enumerate(spans):
        if bit_pos >= len(bits):
            break
        x1, y1, x2, y2 = sp.bbox()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(arr.shape[1], x2), min(arr.shape[0], y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        seed = _span_seed(cfg, ocr_page.page_index, si)
        rng = np.random.default_rng(seed)

        # Сэмплируем точки внутри bbox
        count = min(cfg.bits_per_span, len(bits) - bit_pos)
        ys = rng.integers(y1, y2, size=count)
        xs = rng.integers(x1, x2, size=count)

        ch = cfg.channel
        for k in range(count):
            b = bits[bit_pos]
            px = arr[ys[k], xs[k], ch]
            arr[ys[k], xs[k], ch] = (px & 0xFE) | b
            bit_pos += 1

    return Image.fromarray(arr, mode="RGB")


def extract_spatial(image: Image.Image, ocr_page: OCRPageResult, payload_bits_len: int, cfg: SpatialConfig) -> bytes:
    """
    Извлекает payload_bits_len бит из областей слов.
    """
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    bits_out: List[int] = []
    spans_need = (payload_bits_len + cfg.bits_per_span - 1) // cfg.bits_per_span
    spans = ocr_page.spans[:spans_need]

    for si, sp in enumerate(spans):
        if len(bits_out) >= payload_bits_len:
            break
        x1, y1, x2, y2 = sp.bbox()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(arr.shape[1], x2), min(arr.shape[0], y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue

        seed = _span_seed(cfg, ocr_page.page_index, si)
        rng = np.random.default_rng(seed)

        remaining = payload_bits_len - len(bits_out)
        count = min(cfg.bits_per_span, remaining)

        ys = rng.integers(y1, y2, size=count)
        xs = rng.integers(x1, x2, size=count)

        ch = cfg.channel
        for k in range(count):
            bits_out.append(int(arr[ys[k], xs[k], ch] & 1))

    # добиваем нулями до кратности 8
    while len(bits_out) % 8 != 0:
        bits_out.append(0)

    return bits_to_bytes(bits_out[: ((payload_bits_len + 7)//8)*8])
