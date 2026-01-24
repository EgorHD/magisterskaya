from __future__ import annotations

from pathlib import Path

from core.models.document import Document, DocumentFormat
from core.io.errors import DocumentLoadError
from core.io.image_backend import load_raster_image
from core.io.tiff_backend import load_tiff
from core.io.pdf_backend import load_pdf


def load_document(path: str | Path, *, pdf_dpi: int = 200) -> Document:
    """
    Загружает ЭОД в единую модель Document.
    Поддержка: JPEG/PNG/TIFF/PDF.
    """
    p = Path(path)

    if not p.exists() or not p.is_file():
        raise DocumentLoadError(f"Файл не найден: {p}")

    doc_format = Document.detect_format(p)
    doc = Document(source_path=p, doc_format=doc_format)

    if doc_format in (DocumentFormat.JPEG, DocumentFormat.PNG):
        doc.pages = load_raster_image(p)
        return doc

    if doc_format == DocumentFormat.TIFF:
        doc.pages = load_tiff(p)
        return doc

    if doc_format == DocumentFormat.PDF:
        doc.meta["dpi"] = pdf_dpi
        doc.pages = load_pdf(p, dpi=pdf_dpi)
        return doc

    raise DocumentLoadError(f"Неподдерживаемый формат: .{doc.suffix}")