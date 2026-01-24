from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from PIL import Image


@dataclass(slots=True)
class MarginLSBConfig:
    margin: int = 80          # ширина полей в пикселях
    channel: int = 2          # 0=R 1=G 2=B
    seed_key: str = "HWM-MARGIN-LSB"


def _mask(h: int, w: int, m: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    m = max(1, min(m, min(h, w) // 2))
    mask[:m, :] = 1
    mask[h - m:, :] = 1
    mask[:, :m] = 1
    mask[:, w - m:] = 1
    return mask


def _seed(cfg: MarginLSBConfig, page_index: int) -> int:
    s = f"{cfg.seed_key}|p={page_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")


def _bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)  # MSB->LSB
    return bits.astype(np.uint8)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8)
    if bits.size % 8 != 0:
        pad = 8 - (bits.size % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    b = np.packbits(bits)
    return b.tobytes()


def capacity_bytes(image: Image.Image, cfg: MarginLSBConfig) -> int:
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    coords = np.argwhere(_mask(h, w, cfg.margin) == 1)
    return coords.shape[0] // 8  # 1 бит на пиксель (один канал)


def embed_bytes(image: Image.Image, page_index: int, payload: bytes, cfg: MarginLSBConfig) -> Image.Image:
    """
    Пишем в поля: [len(payload) 4 байта big-endian] + payload
    в LSB выбранного канала по детерминированной перестановке координат.
    """
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]

    header = len(payload).to_bytes(4, "big")
    blob = header + payload
    bits = _bytes_to_bits(blob)

    coords = np.argwhere(_mask(h, w, cfg.margin) == 1)
    if coords.shape[0] < bits.size:
        raise ValueError(
            f"Недостаточно ёмкости полей: нужно {bits.size // 8}B, есть {coords.shape[0] // 8}B"
        )

    rng = np.random.default_rng(_seed(cfg, page_index))
    perm = rng.permutation(coords.shape[0])
    sel = coords[perm[:bits.size]]

    ch = cfg.channel
    ys = sel[:, 0]
    xs = sel[:, 1]

    arr[ys, xs, ch] = (arr[ys, xs, ch] & 0xFE) | bits

    return Image.fromarray(arr, mode="RGB")


def extract_bytes(image: Image.Image, page_index: int, cfg: MarginLSBConfig) -> bytes:
    """
    Читаем из полей: сначала 4 байта длины, затем payload.
    Важно: используем ту же перестановку, что и при встраивании.
    """
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]

    coords = np.argwhere(_mask(h, w, cfg.margin) == 1)
    if coords.shape[0] < 32:
        raise ValueError("Слишком маленькое изображение/поля для чтения заголовка")

    rng = np.random.default_rng(_seed(cfg, page_index))
    perm = rng.permutation(coords.shape[0])

    # 1) длина: первые 32 бита (4 байта)
    sel_len = coords[perm[:32]]
    bits_len = (arr[sel_len[:, 0], sel_len[:, 1], cfg.channel] & 1).astype(np.uint8)
    header = _bits_to_bytes(bits_len)[:4]
    L = int.from_bytes(header, "big")

    cap = coords.shape[0] // 8
    if L <= 0 or L > cap:
        raise ValueError(f"Некорректная длина payload: {L} (cap={cap})")

    total_bits = (4 + L) * 8
    if coords.shape[0] < total_bits:
        raise ValueError(f"Недостаточно ёмкости для payload: нужно {(4 + L)}B, есть {cap}B")

    # 2) читаем весь blob (заголовок+payload) по тем же координатам
    sel = coords[perm[:total_bits]]
    bits = (arr[sel[:, 0], sel[:, 1], cfg.channel] & 1).astype(np.uint8)
    blob = _bits_to_bytes(bits)[: (4 + L)]

    return blob[4:]  # возвращаем только payload
