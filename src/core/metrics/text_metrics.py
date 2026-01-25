from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


@dataclass(slots=True)
class TextSimilarity:
    cer: float  # Character Error Rate
    wer: float  # Word Error Rate
    char_distance: int
    word_distance: int
    ref_chars: int
    ref_words: int


_whitespace_re = re.compile(r"\s+", re.UNICODE)
_word_re = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", re.UNICODE)


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = _whitespace_re.sub(" ", s)
    return s


def tokenize_words(s: str) -> list[str]:
    s = normalize_text(s).lower()
    return _word_re.findall(s)


def _levenshtein_distance_seq(a: Sequence, b: Sequence) -> int:
    """
    Левенштейн с памятью O(min(n,m)).
    """
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    # делаем b короче для экономии памяти
    if m > n:
        a, b = b, a
        n, m = m, n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur0 = i
        prev_diag = i - 1
        for j in range(1, m + 1):
            temp = prev[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur = min(
                prev[j] + 1,      # deletion
                cur0 + 1,         # insertion
                prev_diag + cost  # substitution
            )
            prev_diag = temp
            prev[j] = cur
            cur0 = cur
        prev[0] = i
    return prev[m]


def compute_text_similarity(ocr_text: str, ref_text: str) -> TextSimilarity:
    """
    CER/WER относительно ref_text (эталон из ЦВЗ).
    """
    ocr_n = normalize_text(ocr_text)
    ref_n = normalize_text(ref_text)

    # CER
    char_dist = _levenshtein_distance_seq(ocr_n, ref_n)
    ref_chars = max(1, len(ref_n))
    cer = char_dist / ref_chars

    # WER
    o_words = tokenize_words(ocr_text)
    r_words = tokenize_words(ref_text)
    word_dist = _levenshtein_distance_seq(o_words, r_words)
    ref_words = max(1, len(r_words))
    wer = word_dist / ref_words

    return TextSimilarity(
        cer=float(cer),
        wer=float(wer),
        char_distance=int(char_dist),
        word_distance=int(word_dist),
        ref_chars=int(len(ref_n)),
        ref_words=int(len(r_words)),
    )
