from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import re
import difflib


_word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    # слова + знаки препинания отдельными токенами
    return _word_re.findall(text or "")


@dataclass(slots=True)
class WordDiff:
    """
    Отличие на уровне токенов.
    op: 'insert'/'delete'/'replace'
    a: что в OCR (текущее)
    b: что должно быть (из ЦВЗ)
    pos_a: индекс токена в OCR (примерно)
    pos_b: индекс токена в эталоне
    context_a/context_b: контекст вокруг отличия
    """
    op: str
    a: str
    b: str
    pos_a: int
    pos_b: int
    context_a: str
    context_b: str


def _context(tokens: List[str], i: int, j: int, window: int = 6) -> str:
    left = max(0, i - window)
    right = min(len(tokens), j + window)
    return " ".join(tokens[left:right])


def diff_words(ocr_text: str, ref_text: str, *, max_items: int = 50) -> List[WordDiff]:
    a = tokenize(ocr_text)
    b = tokenize(ref_text)

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out: List[WordDiff] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue

        # собираем фрагменты
        frag_a = " ".join(a[i1:i2]).strip()
        frag_b = " ".join(b[j1:j2]).strip()

        if tag == "replace":
            op = "replace"
        elif tag == "delete":
            op = "delete"
        elif tag == "insert":
            op = "insert"
        else:
            op = tag

        out.append(
            WordDiff(
                op=op,
                a=frag_a,
                b=frag_b,
                pos_a=i1,
                pos_b=j1,
                context_a=_context(a, i1, i2),
                context_b=_context(b, j1, j2),
            )
        )

        if len(out) >= max_items:
            break

    return out


def format_diffs(diffs: List[WordDiff]) -> str:
    if not diffs:
        return "Отличий не найдено."

    lines: List[str] = []
    for k, d in enumerate(diffs, start=1):
        lines.append(f"#{k} [{d.op}]")
        lines.append(f"OCR (факт): {d.a or '(пусто)'}")
        lines.append(f"Эталон:     {d.b or '(пусто)'}")
        lines.append(f"Контекст OCR:    {d.context_a}")
        lines.append(f"Контекст эталон: {d.context_b}")
        lines.append("")
    return "\n".join(lines).strip()
