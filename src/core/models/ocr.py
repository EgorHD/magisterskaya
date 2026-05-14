from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# Точка на плоскости
Point = tuple[float, float]

# Четырёхугольник области текста
Quad = tuple[Point, Point, Point, Point]


@dataclass(frozen=True, slots=True)
class OCRSpan:
    # Координаты области текста
    quad: Quad

    # Распознанный текст
    text: str

    # Уверенность OCR
    confidence: float

    # Ограничивающий прямоугольник области
    def bbox(self) -> tuple[int, int, int, int]:
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


@dataclass(slots=True)
class OCRPageResult:
    # Индекс страницы
    page_index: int

    # OCR-спаны страницы
    spans: list[OCRSpan]

    # Текст страницы
    def text(self) -> str:
        return "\n".join(s.text for s in self.spans if s.text)


@dataclass(slots=True)
class OCRResult:
    # Результаты OCR по страницам
    pages: list[OCRPageResult]

    # Общий текст документа
    def text(self) -> str:
        chunks: list[str] = []

        for p in self.pages:
            t = p.text().strip()
            if t:
                chunks.append(f"--- Страница {p.page_index + 1} ---\n{t}")

        return "\n\n".join(chunks)

    # Итератор по всем OCR-спанам документа
    def iter_spans(self) -> Iterable[OCRSpan]:
        for p in self.pages:
            yield from p.spans