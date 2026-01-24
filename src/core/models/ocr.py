from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


@dataclass(frozen=True, slots=True)
class OCRSpan:
    """
    Один распознанный фрагмент текста (обычно строка/часть строки),
    который возвращает OCR вместе с четырёхугольником области.
    """
    quad: Quad
    text: str
    confidence: float

    def bbox(self) -> tuple[int, int, int, int]:
        """Ограничивающий прямоугольник (x_min, y_min, x_max, y_max)."""
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


@dataclass(slots=True)
class OCRPageResult:
    page_index: int
    spans: list[OCRSpan]

    def text(self) -> str:
        return "\n".join(s.text for s in self.spans if s.text)


@dataclass(slots=True)
class OCRResult:
    pages: list[OCRPageResult]

    def text(self) -> str:
        chunks: list[str] = []
        for p in self.pages:
            t = p.text().strip()
            if t:
                chunks.append(f"--- Страница {p.page_index + 1} ---\n{t}")
        return "\n\n".join(chunks)

    def iter_spans(self) -> Iterable[OCRSpan]:
        for p in self.pages:
            yield from p.spans