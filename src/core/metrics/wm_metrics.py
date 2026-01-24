from __future__ import annotations

def ber(bits_a: bytes, bits_b: bytes) -> float:
    n = min(len(bits_a), len(bits_b))
    if n == 0:
        return 0.0
    err = 0
    total = n * 8
    for i in range(n):
        x = bits_a[i] ^ bits_b[i]
        err += _popcount8(x)
    return err / total


def _popcount8(x: int) -> int:
    return bin(x & 0xFF).count("1")
