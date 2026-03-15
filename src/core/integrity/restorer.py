# src/core/integrity/restorer.py
from __future__ import annotations

from core.models.document import Document
from core.watermark.hybrid import HybridConfig, extract_redundant_text


def restore_text_from_watermark(doc: Document, cfg: HybridConfig) -> tuple[str, str]:
    
    """
    Возвращает (restored_text, debug_info).

    Поддерживает:
    - стандартный режим (payload целиком на странице)
    - лёгкий режим (payload chunked по страницам)
    """
    text, dbg = extract_redundant_text(doc, cfg)
    return text, dbg