from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, UnidentifiedImageError

from core.io.errors import DocumentLoadError
from core.models.document import Page


# Загрузка одностраничного растрового изображения
def load_raster_image(path: Path) -> List[Page]:
    try:
        img = Image.open(path)
    except (UnidentifiedImageError, OSError) as e:
        raise DocumentLoadError(f"Не удалось открыть изображение: {path}") from e

    # Нормализация изображения для дальнейшей обработки
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Получение размеров изображения
    w, h = img.size

    # Формирование одной страницы документа
    page = Page(
        index=0,
        width=w,
        height=h,
        image=img,
        image_path=None,
        ocr_text="",
    )

    return [page]