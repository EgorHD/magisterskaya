from __future__ import annotations

import zlib
import struct
import hashlib


MAGIC = b"HWM1"  # Hybrid WaterMark v1


def pack_payload(primary_bytes: bytes, redundant_bytes: bytes) -> bytes:
    """
    Формат:
    MAGIC(4) | ver(1)=1 | lenP(4) | lenR(4) | crcP(4) | crcR(4) | P | R
    """
    ver = 1
    crc_p = zlib.crc32(primary_bytes) & 0xFFFFFFFF
    crc_r = zlib.crc32(redundant_bytes) & 0xFFFFFFFF
    header = struct.pack(">4sBIIII", MAGIC, ver, len(primary_bytes), len(redundant_bytes), crc_p, crc_r)
    return header + primary_bytes + redundant_bytes


def unpack_payload(blob: bytes) -> tuple[bytes, bytes]:
    if len(blob) < 4 + 1 + 4 + 4 + 4 + 4:
        raise ValueError("Payload too small")

    magic, ver, lp, lr, crc_p, crc_r = struct.unpack(">4sBIIII", blob[:21])
    if magic != MAGIC or ver != 1:
        raise ValueError("Bad payload header")

    need = 21 + lp + lr
    if len(blob) < need:
        raise ValueError("Payload truncated")

    p = blob[21:21 + lp]
    r = blob[21 + lp:21 + lp + lr]

    if (zlib.crc32(p) & 0xFFFFFFFF) != crc_p:
        raise ValueError("Primary CRC mismatch")
    if (zlib.crc32(r) & 0xFFFFFFFF) != crc_r:
        raise ValueError("Redundant CRC mismatch")

    return p, r


def bytes_to_bits(data: bytes) -> list[int]:
    out: list[int] = []
    for b in data:
        for i in range(7, -1, -1):
            out.append((b >> i) & 1)
    return out


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


def sha256_trunc(text: str, nbytes: int = 8) -> bytes:
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    return h[:nbytes]


def compress_text(text: str) -> bytes:
    return zlib.compress(text.encode("utf-8", errors="ignore"), level=9)


def decompress_text(blob: bytes) -> str:
    return zlib.decompress(blob).decode("utf-8", errors="ignore")
