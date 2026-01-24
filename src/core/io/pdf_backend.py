from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image

from core.models.document import Page
from core.io.errors import DocumentLoadError


def load_pdf(path: Path, dpi: int = 200) -> List[Page]:
    """
    Рендерит PDF в список страниц (PIL.Image).
    dpi влияет на качество и скорость (200 — разумный компромисс).
    """
    try:
        pdf = fitz.open(path)
    except Exception as e:
        raise DocumentLoadError(f"Не удалось открыть PDF: {path}") from e

    if pdf.needs_pass:
        pdf.close()
        raise DocumentLoadError("PDF защищён паролем и не может быть обработан без пароля.")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pages: List[Page] = []
    try:
        for i in range(pdf.page_count):
            page = pdf.load_page(i)

            # Принудительно в RGB без alpha (стабильнее для дальнейшего ЦВЗ и метрик)
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)

            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            pages.append(
                Page(
                    index=i,
                    width=pix.width,
                    height=pix.height,
                    image=img,
                    image_path=None,
                    ocr_text="",
                )
            )
    finally:
        pdf.close()

    if not pages:
        raise DocumentLoadError(f"PDF не содержит страниц: {path}")

    return pages