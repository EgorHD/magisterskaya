from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from core.models.document import Document, DocumentFormat
from core.io.errors import DocumentLoadError


def save_document(doc: Document, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fmt = Document.detect_format(out)
    if fmt == DocumentFormat.UNKNOWN:
        raise DocumentLoadError(f"Неподдерживаемое расширение выходного файла: {out.suffix}")

    if not doc.pages or any(p.image is None for p in doc.pages):
        raise DocumentLoadError("Нечего сохранять: документ не содержит изображений страниц.")

    if fmt in (DocumentFormat.JPEG, DocumentFormat.PNG):
        img: Image.Image = doc.pages[0].image  # type: ignore
        img = _norm_img(img, fmt)
        img.save(out)
        return

    if fmt == DocumentFormat.TIFF:
        imgs: List[Image.Image] = [_norm_img(p.image, fmt) for p in doc.pages]  # type: ignore
        first, rest = imgs[0], imgs[1:]
        first.save(out, save_all=True, append_images=rest, compression="tiff_deflate")
        return

    if fmt == DocumentFormat.PDF:
        imgs: List[Image.Image] = [_norm_img(p.image, fmt) for p in doc.pages]  # type: ignore

        # DPI можно взять из doc.meta, если рендерили PDF
        dpi = int(doc.meta.get("dpi", 300))

        # Pillow умеет собирать PDF из изображений
        first, rest = imgs[0], imgs[1:]
        first.save(
            out,
            "PDF",
            save_all=True,
            append_images=rest,
            resolution=dpi
        )
        return

    raise DocumentLoadError(f"Неподдерживаемый формат сохранения: {fmt}")


def _norm_img(img: Image.Image, fmt: DocumentFormat) -> Image.Image:
    # JPEG не любит alpha, PDF/TIFF тоже лучше без неожиданностей
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    # JPEG строго RGB/L
    if fmt == DocumentFormat.JPEG and img.mode != "RGB":
        img = img.convert("RGB")
    return img
