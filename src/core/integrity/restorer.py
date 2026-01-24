from __future__ import annotations

from core.models.document import Document
from core.watermark.hybrid import HybridConfig, extract_from_page
from core.watermark.codec import decompress_text


def restore_text_from_watermark(doc: Document, cfg: HybridConfig) -> tuple[str, str]:
    """
    Возвращает (restored_text, debug_info)
    """
    for i in range(len(doc.pages)):
        ext = extract_from_page(doc, i, cfg)
        if not ext.ok or not ext.redundant:
            continue
        try:
            text = decompress_text(ext.redundant)
            return text, f"OK: extracted redundant from page {i+1}, bytes={len(ext.redundant)}"
        except Exception as e:
            # пробуем следующую страницу
            continue

    return "", "FAIL: could not extract/decompress redundant from any page"
