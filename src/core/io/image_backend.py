from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, UnidentifiedImageError

from core.models.document import Page
from core.io.errors import DocumentLoadError


def load_raster_image(path: Path) -> List[Page]:
    """
    Загружает одностраничные растровые изображения (JPEG/PNG и т.д.).
    Возвращает список страниц (обычно 1 Page).
    """
    try:
        img = Image.open(path)
    except (UnidentifiedImageError, OSError) as e:
        raise DocumentLoadError(f"Не удалось открыть изображение: {path}") from e

    # Нормализуем в RGB (для стабильной дальнейшей обработки)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    w, h = img.size
    page = Page(index=0, width=w, height=h, image=img, image_path=None, ocr_text="")
    return [page]