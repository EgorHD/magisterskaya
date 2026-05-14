from __future__ import annotations

from pathlib import Path

from core.io.errors import DocumentLoadError
from core.io.image_backend import load_raster_image
from core.io.pdf_backend import load_pdf
from core.io.tiff_backend import load_tiff
from core.models.document import Document, DocumentFormat


# Загрузка документа в единую модель Document
def load_document(path: str | Path, *, pdf_dpi: int = 200) -> Document:
    p = Path(path)

    # Проверка существования файла
    if not p.exists() or not p.is_file():
        raise DocumentLoadError(f"Файл не найден: {p}")

    # Определение формата документа
    doc_format = Document.detect_format(p)

    # Создание объекта документа
    doc = Document(source_path=p, doc_format=doc_format)

    # Загрузка JPEG и PNG
    if doc_format in (DocumentFormat.JPEG, DocumentFormat.PNG):
        doc.pages = load_raster_image(p)
        return doc

    # Загрузка TIFF
    if doc_format == DocumentFormat.TIFF:
        doc.pages = load_tiff(p)
        return doc

    # Загрузка PDF
    if doc_format == DocumentFormat.PDF:
        doc.meta["dpi"] = pdf_dpi
        doc.pages = load_pdf(p, dpi=pdf_dpi)
        return doc

    # Ошибка для неподдерживаемого формата
    raise DocumentLoadError(f"Неподдерживаемый формат: .{doc.suffix}")