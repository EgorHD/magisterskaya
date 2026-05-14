from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from core.io.errors import DocumentLoadError
from core.models.document import Document, DocumentFormat


# Сохранение документа в выходной файл
def save_document(doc: Document, out_path: str | Path) -> None:
    out = Path(out_path)

    # Создание выходной папки
    out.parent.mkdir(parents=True, exist_ok=True)

    # Определение формата выходного файла
    fmt = Document.detect_format(out)
    if fmt == DocumentFormat.UNKNOWN:
        raise DocumentLoadError(f"Неподдерживаемое расширение выходного файла: {out.suffix}")

    # Проверка наличия изображений страниц
    if not doc.pages or any(p.image is None for p in doc.pages):
        raise DocumentLoadError("Нечего сохранять: документ не содержит изображений страниц.")

    # Сохранение одностраничного JPEG или PNG
    if fmt in (DocumentFormat.JPEG, DocumentFormat.PNG):
        img: Image.Image = doc.pages[0].image  # type: ignore
        img = _norm_img(img, fmt)
        img.save(out)
        return

    # Сохранение многостраничного TIFF
    if fmt == DocumentFormat.TIFF:
        imgs: List[Image.Image] = [_norm_img(p.image, fmt) for p in doc.pages]  # type: ignore
        first, rest = imgs[0], imgs[1:]
        first.save(out, save_all=True, append_images=rest, compression="tiff_deflate")
        return

    # Сохранение PDF из изображений
    if fmt == DocumentFormat.PDF:
        imgs: List[Image.Image] = [_norm_img(p.image, fmt) for p in doc.pages]  # type: ignore

        # DPI можно взять из метаданных документа
        dpi = int(doc.meta.get("dpi", 300))

        # Pillow собирает PDF из списка изображений
        first, rest = imgs[0], imgs[1:]
        first.save(
            out,
            "PDF",
            save_all=True,
            append_images=rest,
            resolution=dpi,
        )
        return

    # Ошибка для неподдерживаемого формата
    raise DocumentLoadError(f"Неподдерживаемый формат сохранения: {fmt}")


# Нормализация изображения перед сохранением
def _norm_img(img: Image.Image, fmt: DocumentFormat) -> Image.Image:
    # Приведение к RGB при нестандартном режиме
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # JPEG сохраняем только в RGB
    if fmt == DocumentFormat.JPEG and img.mode != "RGB":
        img = img.convert("RGB")

    return img