from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class DocumentFormat(str, Enum):
    PDF = "pdf"
    TIFF = "tiff"
    JPEG = "jpeg"
    PNG = "png"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Page:
    """
    Одна страница документа.

    На ранних этапах мы можем хранить:
    - изображение страницы (когда подключим Pillow/OpenCV) в поле image
    - либо путь к временному файлу/кэшу (если решим кэшировать рендер)
    """
    index: int
    width: int | None = None
    height: int | None = None

    # Позже сюда положим PIL.Image или numpy.ndarray.
    image: Any | None = None

    # Если страница была получена из временного файла (например, рендер PDF -> PNG)
    image_path: Optional[Path] = None

    # Текст распознавания (пока строкой; позже будет OCRResult со словами и bbox)
    ocr_text: str = ""
    ocr_result: Any | None = None  # позже: OCRPageResult


@dataclass(slots=True)
class Document:
    """
    Единая модель "электронного образа документа" для всех форматов.
    """
    source_path: Path
    doc_format: DocumentFormat = DocumentFormat.UNKNOWN
    pages: list[Page] = field(default_factory=list)

    # Метаданные (пригодится для PDF: dpi, размер страницы и т.п.)
    meta: dict[str, Any] = field(default_factory=dict)

    def page_count(self) -> int:
        return len(self.pages)

    def is_multipage(self) -> bool:
        return len(self.pages) > 1

    @property
    def name(self) -> str:
        return self.source_path.name

    @property
    def suffix(self) -> str:
        return self.source_path.suffix.lower().lstrip(".")

    @staticmethod
    def detect_format(path: Path) -> DocumentFormat:
        ext = path.suffix.lower().lstrip(".")
        if ext == "pdf":
            return DocumentFormat.PDF
        if ext in ("tif", "tiff"):
            return DocumentFormat.TIFF
        if ext in ("jpg", "jpeg"):
            return DocumentFormat.JPEG
        if ext == "png":
            return DocumentFormat.PNG
        return DocumentFormat.UNKNOWN