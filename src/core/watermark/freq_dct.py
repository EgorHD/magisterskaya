from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import zlib

import numpy as np
from PIL import Image
import cv2


@dataclass(slots=True)
class DCTConfig:
    block: int = 8
    # коэффициенты для сравнения (mid-frequency)
    c1: Tuple[int, int] = (2, 3)
    c2: Tuple[int, int] = (3, 2)
    # "сила" встраивания
    delta: float = 12.0
    # поля (px) — где именно внедряем (верх/низ/лево/право)
    margin_px: int = 120
    # повторение бит для устойчивости
    repetition: int = 3


def _bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr).astype(np.uint8)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if len(bits) % 8 != 0:
        bits = np.pad(bits, (0, 8 - len(bits) % 8), constant_values=0)
    return np.packbits(bits.astype(np.uint8)).tobytes()


def _with_header(payload: bytes) -> bytes:
    # header: len(4) + crc32(4) + payload
    ln = len(payload).to_bytes(4, "big")
    crc = zlib.crc32(payload).to_bytes(4, "big")
    return ln + crc + payload


def _parse_header(data: bytes) -> bytes:
    if len(data) < 8:
        raise ValueError("Too short")
    ln = int.from_bytes(data[:4], "big")
    crc = int.from_bytes(data[4:8], "big")
    payload = data[8:8 + ln]
    if len(payload) != ln:
        raise ValueError("Bad length")
    if zlib.crc32(payload) != crc:
        raise ValueError("Bad CRC32")
    return payload


def _get_embed_mask(h: int, w: int, m: int) -> np.ndarray:
    """
    Маска областей полей: верх/низ/лево/право толщиной m.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    m = max(0, min(m, min(h // 2, w // 2)))
    if m == 0:
        mask[:, :] = 1
        return mask
    mask[:m, :] = 1
    mask[h - m:, :] = 1
    mask[:, :m] = 1
    mask[:, w - m:] = 1
    return mask


def embed_bytes_dct(image: Image.Image, payload: bytes, cfg: DCTConfig) -> Image.Image:
    """
    Встраивание в DCT по блокам 8x8 в полях.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    np_img = np.array(image, dtype=np.uint8)

    # работаем в YCrCb, встраиваем в Y (яркость)
    ycc = cv2.cvtColor(np_img, cv2.COLOR_RGB2YCrCb)
    Y = ycc[:, :, 0].astype(np.float32)

    h, w = Y.shape
    b = cfg.block
    mask = _get_embed_mask(h, w, cfg.margin_px)

    data = _with_header(payload)
    bits = _bytes_to_bits(data)
    if cfg.repetition > 1:
        bits = np.repeat(bits, cfg.repetition)

    # доступные блоки
    blocks: List[Tuple[int, int]] = []
    for y in range(0, h - b + 1, b):
        for x in range(0, w - b + 1, b):
            # блок подходит, если большинство пикселей в маске
            if mask[y:y + b, x:x + b].mean() > 0.7:
                blocks.append((y, x))

    if len(bits) > len(blocks):
        raise ValueError(f"DCT capacity too small: need {len(bits)} bits, have {len(blocks)} bits")

    (u1, v1) = cfg.c1
    (u2, v2) = cfg.c2
    delta = float(cfg.delta)

    for i, bit in enumerate(bits):
        y0, x0 = blocks[i]
        block = Y[y0:y0 + b, x0:x0 + b]
        dct = cv2.dct(block)

        a = dct[u1, v1]
        c = dct[u2, v2]

        # enforce relation depending on bit
        if bit == 1:
            # want a > c + delta
            if a <= c + delta:
                mid = (a + c) / 2.0
                dct[u1, v1] = mid + delta / 2.0
                dct[u2, v2] = mid - delta / 2.0
        else:
            # want c > a + delta
            if c <= a + delta:
                mid = (a + c) / 2.0
                dct[u1, v1] = mid - delta / 2.0
                dct[u2, v2] = mid + delta / 2.0

        Y[y0:y0 + b, x0:x0 + b] = cv2.idct(dct)

    # обратно в uint8
    ycc[:, :, 0] = np.clip(Y, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(out, mode="RGB")


def extract_bytes_dct(image: Image.Image, cfg: DCTConfig, *, max_bytes: int = 200_000) -> bytes:
    """
    Извлекаем payload (с CRC) из DCT.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    np_img = np.array(image, dtype=np.uint8)

    ycc = cv2.cvtColor(np_img, cv2.COLOR_RGB2YCrCb)
    Y = ycc[:, :, 0].astype(np.float32)
    h, w = Y.shape
    b = cfg.block

    mask = _get_embed_mask(h, w, cfg.margin_px)

    blocks: List[Tuple[int, int]] = []
    for y in range(0, h - b + 1, b):
        for x in range(0, w - b + 1, b):
            if mask[y:y + b, x:x + b].mean() > 0.7:
                blocks.append((y, x))

    (u1, v1) = cfg.c1
    (u2, v2) = cfg.c2

    # сначала читаем с запасом: header(8 bytes) => 64 bits * repetition
    # потом узнаём длину и дочитываем
    def read_bits(nbits: int) -> np.ndarray:
        if nbits > len(blocks):
            raise ValueError("Not enough DCT blocks")
        bits = np.zeros(nbits, dtype=np.uint8)
        for i in range(nbits):
            y0, x0 = blocks[i]
            block = Y[y0:y0 + b, x0:x0 + b]
            dct = cv2.dct(block)
            a = dct[u1, v1]
            c = dct[u2, v2]
            bits[i] = 1 if a > c else 0
        return bits

    rep = max(1, int(cfg.repetition))

    # читаем первые 8 байт (header)
    header_bits = read_bits(8 * 8 * rep)
    if rep > 1:
        header_bits = header_bits.reshape(-1, rep)
        header_bits = (header_bits.mean(axis=1) >= 0.5).astype(np.uint8)

    header_bytes = _bits_to_bytes(header_bits)[:8]
    ln = int.from_bytes(header_bytes[:4], "big")
    if ln < 0 or ln > max_bytes:
        raise ValueError("Bad length in header")

    total_bytes = 8 + ln
    total_bits = total_bytes * 8
    need = total_bits * rep

    all_bits = read_bits(need)
    if rep > 1:
        all_bits = all_bits.reshape(-1, rep)
        all_bits = (all_bits.mean(axis=1) >= 0.5).astype(np.uint8)

    data = _bits_to_bytes(all_bits)[:total_bytes]
    return _parse_header(data)
