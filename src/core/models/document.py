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
    # Индекс страницы в документе
    index: int

    # Размеры страницы
    width: int | None = None
    height: int | None = None

    # Изображение страницы
    image: Any | None = None

    # Путь к временному файлу страницы
    image_path: Optional[Path] = None

    # Распознанный текст страницы
    ocr_text: str = ""

    # Результат OCR со спанами и bbox
    ocr_result: Any | None = None


@dataclass(slots=True)
class Document:
    # Исходный путь к документу
    source_path: Path

    # Формат документа
    doc_format: DocumentFormat = DocumentFormat.UNKNOWN

    # Страницы документа
    pages: list[Page] = field(default_factory=list)

    # Служебные метаданные
    meta: dict[str, Any] = field(default_factory=dict)

    # Количество страниц
    def page_count(self) -> int:
        return len(self.pages)

    # Проверка на многостраничный документ
    def is_multipage(self) -> bool:
        return len(self.pages) > 1

    # Имя файла документа
    @property
    def name(self) -> str:
        return self.source_path.name

    # Расширение файла без точки
    @property
    def suffix(self) -> str:
        return self.source_path.suffix.lower().lstrip(".")

    # Определение формата по расширению
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