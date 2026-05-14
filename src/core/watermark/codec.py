from __future__ import annotations

import hashlib
import struct
import zlib


# Сигнатура payload гибридного ЦВЗ
MAGIC = b"HWM1"


# Упаковка primary и redundant в единый payload
def pack_payload(primary_bytes: bytes, redundant_bytes: bytes) -> bytes:
    ver = 1

    # CRC для primary-слоя
    crc_p = zlib.crc32(primary_bytes) & 0xFFFFFFFF

    # CRC для redundant-слоя
    crc_r = zlib.crc32(redundant_bytes) & 0xFFFFFFFF

    # Заголовок payload
    header = struct.pack(
        ">4sBIIII",
        MAGIC,
        ver,
        len(primary_bytes),
        len(redundant_bytes),
        crc_p,
        crc_r,
    )

    return header + primary_bytes + redundant_bytes


# Распаковка payload на primary и redundant
def unpack_payload(blob: bytes) -> tuple[bytes, bytes]:
    # Проверка минимального размера payload
    if len(blob) < 4 + 1 + 4 + 4 + 4 + 4:
        raise ValueError("Payload too small")

    # Чтение заголовка
    magic, ver, lp, lr, crc_p, crc_r = struct.unpack(">4sBIIII", blob[:21])

    # Проверка сигнатуры и версии
    if magic != MAGIC or ver != 1:
        raise ValueError("Bad payload header")

    # Проверка полной длины payload
    need = 21 + lp + lr
    if len(blob) < need:
        raise ValueError("Payload truncated")

    # Извлечение primary и redundant
    p = blob[21:21 + lp]
    r = blob[21 + lp:21 + lp + lr]

    # Проверка CRC primary
    if (zlib.crc32(p) & 0xFFFFFFFF) != crc_p:
        raise ValueError("Primary CRC mismatch")

    # Проверка CRC redundant
    if (zlib.crc32(r) & 0xFFFFFFFF) != crc_r:
        raise ValueError("Redundant CRC mismatch")

    return p, r


# Преобразование байтов в список битов
def bytes_to_bits(data: bytes) -> list[int]:
    out: list[int] = []

    for b in data:
        for i in range(7, -1, -1):
            out.append((b >> i) & 1)

    return out


# Преобразование списка битов в байты
def bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("bits length must be multiple of 8")

    bb = bytearray()

    for i in range(0, len(bits), 8):
        v = 0
        for j in range(8):
            v = (v << 1) | (bits[i + j] & 1)
        bb.append(v)

    return bytes(bb)


# Усечённый SHA-256 для текста
def sha256_trunc(text: str, nbytes: int = 8) -> bytes:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    return h[:nbytes]


# Сжатие текста
def compress_text(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8", errors="ignore"), level=9)


# Распаковка текста
def decompress_text(blob: bytes) -> str:
    return zlib.decompress(blob).decode("utf-8", errors="ignore")