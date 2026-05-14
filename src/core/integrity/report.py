from __future__ import annotations
from dataclasses import dataclass


@dataclass(slots=True)
class IntegrityReport:
    # Общий результат проверки
    ok: bool

    # Номера страниц с нарушениями
    broken_pages: list[int]

    # Индексы повреждённых фрагментов по страницам
    # Формат: page_index -> list of span indices
    broken_spans: dict[int, list[int]]

    # Текстовое описание результата
    message: str