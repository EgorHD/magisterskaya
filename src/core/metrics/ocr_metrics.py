from __future__ import annotations

import re


def _levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    # работает и для строк (по символам), и для списков (по токенам)
    if isinstance(a, str):
        a = list(a)
    if isinstance(b, str):
        b = list(b)

    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def cer(ref: str, hyp: str) -> float:
    ref = ref or ""
    hyp = hyp or ""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return _levenshtein(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    ref_toks = re.findall(r"\S+", ref or "")
    hyp_toks = re.findall(r"\S+", hyp or "")
    if len(ref_toks) == 0:
        return 0.0 if len(hyp_toks) == 0 else 1.0
    return _levenshtein(ref_toks, hyp_toks) / max(1, len(ref_toks))
