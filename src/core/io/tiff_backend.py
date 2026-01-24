from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageSequence, UnidentifiedImageError

from core.models.document import Page
from core.io.errors import DocumentLoadError


def load_tiff(path: Path) -> List[Page]:
    """
    Загружает TIFF (включая многокадровые TIFF).
    Каждую "рамку" TIFF считаем страницей.
    """
    try:
        img = Image.open(path)
    except (UnidentifiedImageError, OSError) as e:
        raise DocumentLoadError(f"Не удалось открыть TIFF: {path}") from e

    pages: List[Page] = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        # Важно: frame может быть ленивым объектом, сделаем копию
        frame_copy = frame.copy()
        if frame_copy.mode not in ("RGB", "L"):
            frame_copy = frame_copy.convert("RGB")

        w, h = frame_copy.size
        pages.append(Page(index=i, width=w, height=h, image=frame_copy, image_path=None, ocr_text=""))

    if not pages:
        raise DocumentLoadError(f"TIFF не содержит кадров/страниц: {path}")

    return pages