from __future__ import annotations
import difflib
import re
from dataclasses import dataclass
from typing import List


# Токены: слова и знаки препинания
_word_re = re.compile(r"\w+|[^\w\s]", re.UNICODE)


# Разбиение текста на токены
def tokenize(text: str) -> List[str]:
    return _word_re.findall(text or "")


@dataclass(slots=True)
class WordDiff:
    # Тип отличия: insert / delete / replace
    op: str

    # Фрагмент из OCR-текста
    a: str

    # Эталонный фрагмент
    b: str

    # Позиция в OCR-тексте
    pos_a: int

    # Позиция в эталонном тексте
    pos_b: int

    # Контекст вокруг отличия в OCR
    context_a: str

    # Контекст вокруг отличия в эталоне
    context_b: str


# Получение контекста вокруг отличия
def _context(tokens: List[str], i: int, j: int, window: int = 6) -> str:
    left = max(0, i - window)
    right = min(len(tokens), j + window)
    return " ".join(tokens[left:right])


# Поиск отличий между OCR-текстом и эталоном
def diff_words(ocr_text: str, ref_text: str, *, max_items: int = 50) -> List[WordDiff]:
    # Токенизация текстов
    a = tokenize(ocr_text)
    b = tokenize(ref_text)

    # Поиск различий между последовательностями
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out: List[WordDiff] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        # Равные участки пропускаем
        if tag == "equal":
            continue

        # Фрагмент из OCR-текста
        frag_a = " ".join(a[i1:i2]).strip()

        # Фрагмент из эталонного текста
        frag_b = " ".join(b[j1:j2]).strip()

        # Нормализация типа операции
        if tag == "replace":
            op = "replace"
        elif tag == "delete":
            op = "delete"
        elif tag == "insert":
            op = "insert"
        else:
            op = tag

        # Сохранение найденного отличия
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

        # Ограничение числа отличий в отчёте
        if len(out) >= max_items:
            break

    return out


# Форматирование отличий в текстовый отчёт
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