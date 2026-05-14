from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(slots=True)
class MarginLSBConfig:
    # Ширина полей в пикселях
    margin: int = 80

    # Канал для записи: 0=R 1=G 2=B
    channel: int = 2

    # Ключ для генерации детерминированной перестановки
    seed_key: str = "HWM-MARGIN-LSB"


# Маска полей страницы
def _mask(h: int, w: int, m: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    m = max(1, min(m, min(h, w) // 2))

    mask[:m, :] = 1
    mask[h - m:, :] = 1
    mask[:, :m] = 1
    mask[:, w - m:] = 1

    return mask


# Генерация seed для страницы
def _seed(cfg: MarginLSBConfig, page_index: int) -> int:
    s = f"{cfg.seed_key}|p={page_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big")


# Перевод байтов в массив битов
def _bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr)
    return bits.astype(np.uint8)


# Перевод массива битов в байты
def _bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8)

    if bits.size % 8 != 0:
        pad = 8 - (bits.size % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    b = np.packbits(bits)
    return b.tobytes()


# Оценка вместимости полей в байтах
def capacity_bytes(image: Image.Image, cfg: MarginLSBConfig) -> int:
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]

    coords = np.argwhere(_mask(h, w, cfg.margin) == 1)

    # Один пиксель хранит один бит в выбранном канале
    return coords.shape[0] // 8


# Встраивание payload в поля страницы
def embed_bytes(
    image: Image.Image,
    page_index: int,
    payload: bytes,
    cfg: MarginLSBConfig,
) -> Image.Image:
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]

    # Заголовок: длина payload в 4 байтах
    header = len(payload).to_bytes(4, "big")
    blob = header + payload
    bits = _bytes_to_bits(blob)

    coords = np.argwhere(_mask(h, w, cfg.margin) == 1)

    if coords.shape[0] < bits.size:
        raise ValueError(
            f"Недостаточно ёмкости полей: нужно {bits.size // 8}B, есть {coords.shape[0] // 8}B"
        )

    # Детерминированная перестановка координат
    rng = np.random.default_rng(_seed(cfg, page_index))
    perm = rng.permutation(coords.shape[0])
    sel = coords[perm[:bits.size]]

    ch = cfg.channel
    ys = sel[:, 0]
    xs = sel[:, 1]

    # Запись битов в LSB выбранного канала
    arr[ys, xs, ch] = (arr[ys, xs, ch] & 0xFE) | bits

    return Image.fromarray(arr, mode="RGB")


# Извлечение payload из полей страницы
def extract_bytes(image: Image.Image, page_index: int, cfg: MarginLSBConfig) -> bytes:
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape[:2]

    coords = np.argwhere(_mask(h, w, cfg.margin) == 1)
    if coords.shape[0] < 32:
        raise ValueError("Слишком маленькое изображение/поля для чтения заголовка")

    # Та же детерминированная перестановка координат
    rng = np.random.default_rng(_seed(cfg, page_index))
    perm = rng.permutation(coords.shape[0])

    # Чтение длины payload из первых 32 бит
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

    # Чтение всего blob: заголовок + payload
    sel = coords[perm[:total_bits]]
    bits = (arr[sel[:, 0], sel[:, 1], cfg.channel] & 1).astype(np.uint8)
    blob = _bits_to_bytes(bits)[: (4 + L)]

    # Возвращаем только payload
    return blob[4:]