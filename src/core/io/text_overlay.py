from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass(slots=True)
class TextPageConfig:
    margin_px: int = 80
    font_size: int = 28
    line_spacing: int = 8
    bg: Tuple[int, int, int] = (255, 255, 255)
    fg: Tuple[int, int, int] = (0, 0, 0)
    title: str = "Восстановленный текст (из резервного слоя ЦВЗ)"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Пытаемся загрузить шрифт с поддержкой кириллицы.
    Pillow обычно содержит DejaVuSans.ttf.
    На Windows часто есть Arial.
    """
    candidates = [
        "DejaVuSans.ttf",
        "Arial.ttf",
        "arial.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font, max_width_px: int) -> List[str]:
    # грубая обёртка: подбираем примерное число символов на строку
    # затем уточняем измерением ширины
    if max_width_px <= 50:
        return [text]

    # начальная оценка
    avg_char_px = max(8, font.size // 2) if hasattr(font, "size") else 10
    est_chars = max(20, max_width_px // avg_char_px)

    lines: List[str] = []
    for para in (text or "").splitlines():
        if not para.strip():
            lines.append("")
            continue
        chunks = textwrap.wrap(para, width=est_chars, break_long_words=True, break_on_hyphens=False)
        if not chunks:
            lines.append("")
            continue

        # уточняем: если строка слишком длинная — дробим ещё
        for ch in chunks:
            cur = ch
            while True:
                w = draw.textlength(cur, font=font)
                if w <= max_width_px or len(cur) <= 5:
                    lines.append(cur)
                    break
                # отрезаем часть
                cut = max(5, int(len(cur) * (max_width_px / max(w, 1))))
                lines.append(cur[:cut])
                cur = cur[cut:]
    return lines


def make_text_pages(
    restored_text: str,
    *,
    page_size: Tuple[int, int],
    cfg: TextPageConfig | None = None,
) -> List[Image.Image]:
    """
    Делает одну или несколько страниц-изображений с текстом.
    Размер страниц берём как у документа (page_size).
    """
    cfg = cfg or TextPageConfig()
    w, h = page_size
    img_pages: List[Image.Image] = []

    font_title = _load_font(cfg.font_size + 6)
    font_body = _load_font(cfg.font_size)

    # создаём временный canvas для измерений
    tmp = Image.new("RGB", (w, h), cfg.bg)
    dtmp = ImageDraw.Draw(tmp)

    inner_w = w - 2 * cfg.margin_px
    inner_h = h - 2 * cfg.margin_px

    title_lines = [cfg.title]
    body_lines = _wrap_text_to_width(dtmp, restored_text, font_body, inner_w)

    # высоты строк
    title_h = (font_title.size + cfg.line_spacing) if hasattr(font_title, "size") else (cfg.font_size + 6 + cfg.line_spacing)
    body_h = (font_body.size + cfg.line_spacing) if hasattr(font_body, "size") else (cfg.font_size + cfg.line_spacing)

    # сколько строк тела помещается на странице (учитывая заголовок)
    header_block = title_h * len(title_lines) + cfg.line_spacing * 2
    max_body_lines_per_page = max(1, (inner_h - header_block) // body_h)

    # делим на страницы
    i = 0
    while i < len(body_lines):
        page = Image.new("RGB", (w, h), cfg.bg)
        draw = ImageDraw.Draw(page)

        x = cfg.margin_px
        y = cfg.margin_px

        # заголовок
        for tl in title_lines:
            draw.text((x, y), tl, font=font_title, fill=cfg.fg)
            y += title_h
        y += cfg.line_spacing * 2

        # тело
        chunk = body_lines[i:i + max_body_lines_per_page]
        for line in chunk:
            draw.text((x, y), line, font=font_body, fill=cfg.fg)
            y += body_h

        img_pages.append(page)
        i += max_body_lines_per_page

    return img_pages


def save_images_as_document(
    images: List[Image.Image],
    out_path: str | Path,
    *,
    dpi: int = 300,
) -> None:
    """
    Сохраняет список изображений как TIFF или PDF (по расширению).
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not images:
        raise ValueError("Нет изображений для сохранения.")

    ext = out.suffix.lower()
    imgs = [im.convert("RGB") if im.mode != "RGB" else im for im in images]
    first, rest = imgs[0], imgs[1:]

    if ext in (".tif", ".tiff"):
        first.save(out, save_all=True, append_images=rest, compression="tiff_deflate", dpi=(dpi, dpi))
        return

    if ext == ".pdf":
        first.save(out, "PDF", save_all=True, append_images=rest, resolution=dpi)
        return

    raise ValueError(f"Неподдерживаемый формат сохранения: {ext} (нужно .tif/.tiff/.pdf)")
