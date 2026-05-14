from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from core.models.ocr import OCRSpan


@dataclass(slots=True)
class SpatialWordLSBConfig:
    # какой канал менять: 0=R,1=G,2=B
    channel: int = 2
    # сколько LSB использовать (1 = самый незаметный/хрупкий)
    lsb_bits: int = 1
    # отступ внутрь bbox, чтобы не трогать границы
    inset: int = 1
    # максимум пикселей на слово (ограничение для скорости)
    max_pixels_per_span: int = 40_000


def _bbox_safe(span: OCRSpan) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = span.bbox()
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return int(x1), int(y1), int(x2), int(y2)


def _bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.uint8)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8)
    if len(bits) % 8 != 0:
        bits = np.pad(bits, (0, 8 - (len(bits) % 8)), constant_values=0)
    arr = np.packbits(bits)
    return arr.tobytes()


def embed_hashes_in_words(
    image: Image.Image,
    spans_sorted: List[OCRSpan],
    hashes: List[bytes],
    cfg: SpatialWordLSBConfig,
) -> Image.Image:
    """
    Встраиваем для каждого span фиксированное число байт (hashes[i]) в пиксели внутри bbox.
    Хрупкий слой: любое изменение в области слов -> обнаружение.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    np_img = np.array(image, dtype=np.uint8)
    H, W, _ = np_img.shape

    if len(spans_sorted) != len(hashes):
        raise ValueError("spans_sorted и hashes должны быть одинаковой длины")

    ch = int(cfg.channel)
    lsb_bits = int(cfg.lsb_bits)
    if ch < 0 or ch > 2:
        raise ValueError("channel должен быть 0..2")
    if lsb_bits < 1 or lsb_bits > 2:
        # 1-2 достаточно, дальше качество страдает и смысла мало
        raise ValueError("lsb_bits должен быть 1 или 2")

    for sp, hb in zip(spans_sorted, hashes):
        x1, y1, x2, y2 = _bbox_safe(sp)

        # inset + clip
        x1 = max(0, min(W - 1, x1 + cfg.inset))
        y1 = max(0, min(H - 1, y1 + cfg.inset))
        x2 = max(0, min(W, x2 - cfg.inset))
        y2 = max(0, min(H, y2 - cfg.inset))

        if x2 <= x1 or y2 <= y1:
            continue

        region = np_img[y1:y2, x1:x2, ch]
        flat_u8 = region.reshape(-1)

        if flat_u8.size == 0:
            continue

        n = int(min(flat_u8.size, cfg.max_pixels_per_span))
        bits = _bytes_to_bits(hb)
        need = int(bits.size)
        cap = n * lsb_bits
        if need > cap:
            # bbox мал — пропускаем
            continue

        # ВАЖНО: работаем в int, потом клип + uint8
        flat = flat_u8[:n].astype(np.int32, copy=True)

        idx = 0
        for i in range(n):
            v = int(flat[i])
            for b in range(lsb_bits):
                bit = int(bits[idx])
                idx += 1
                v = (v & ~(1 << b)) | (bit << b)
                if idx >= need:
                    break
            flat[i] = v
            if idx >= need:
                break

        # безопасно возвращаем в uint8
        flat = np.clip(flat, 0, 255).astype(np.uint8)

        # пишем обратно в region (только первые n пикселей)
        out_region = region.reshape(-1)
        out_region[:n] = flat
        np_img[y1:y2, x1:x2, ch] = out_region.reshape(region.shape)

    return Image.fromarray(np_img, mode="RGB")


def extract_hashes_from_words(
    image: Image.Image,
    spans_sorted: List[OCRSpan],
    bytes_per_span: int,
    cfg: SpatialWordLSBConfig,
) -> List[bytes]:
    """
    Извлекаем bytes_per_span байт из bbox каждого span в том же порядке.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    np_img = np.array(image, dtype=np.uint8)
    H, W, _ = np_img.shape

    ch = int(cfg.channel)
    lsb_bits = int(cfg.lsb_bits)

    out: List[bytes] = []
    bits_per_span = int(bytes_per_span) * 8

    for sp in spans_sorted:
        x1, y1, x2, y2 = _bbox_safe(sp)

        x1 = max(0, min(W - 1, x1 + cfg.inset))
        y1 = max(0, min(H - 1, y1 + cfg.inset))
        x2 = max(0, min(W, x2 - cfg.inset))
        y2 = max(0, min(H, y2 - cfg.inset))

        if x2 <= x1 or y2 <= y1:
            out.append(b"")
            continue

        region = np_img[y1:y2, x1:x2, ch]
        flat_u8 = region.reshape(-1)
        if flat_u8.size == 0:
            out.append(b"")
            continue

        n = int(min(flat_u8.size, cfg.max_pixels_per_span))
        cap = n * lsb_bits
        if bits_per_span > cap:
            out.append(b"")
            continue

        bits = np.zeros(bits_per_span, dtype=np.uint8)
        idx = 0
        # тоже работаем через int
        flat = flat_u8[:n].astype(np.int32, copy=False)

        for i in range(n):
            v = int(flat[i])
            for b in range(lsb_bits):
                bits[idx] = (v >> b) & 1
                idx += 1
                if idx >= bits_per_span:
                    break
            if idx >= bits_per_span:
                break

        out.append(_bits_to_bytes(bits)[:bytes_per_span])

    return out