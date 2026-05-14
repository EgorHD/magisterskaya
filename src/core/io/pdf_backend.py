from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image

from core.io.errors import DocumentLoadError
from core.models.document import Page


# Загрузка PDF в список страниц
def load_pdf(path: Path, dpi: int = 200) -> List[Page]:
    try:
        pdf = fitz.open(path)
    except Exception as e:
        raise DocumentLoadError(f"Не удалось открыть PDF: {path}") from e

    # Проверка защиты паролем
    if pdf.needs_pass:
        pdf.close()
        raise DocumentLoadError("PDF защищён паролем и не может быть обработан без пароля.")

    # Коэффициент масштабирования для рендеринга
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    pages: List[Page] = []

    try:
        for i in range(pdf.page_count):
            page = pdf.load_page(i)

            # Рендер страницы в RGB без alpha
            pix = page.get_pixmap(
                matrix=matrix,
                colorspace=fitz.csRGB,
                alpha=False,
            )

            # Перевод в PIL.Image
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            # Добавление страницы в модель документа
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
        # Закрытие PDF-документа
        pdf.close()

    # Проверка, что в PDF есть страницы
    if not pages:
        raise DocumentLoadError(f"PDF не содержит страниц: {path}")

    return pages