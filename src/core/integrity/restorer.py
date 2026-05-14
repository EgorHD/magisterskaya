# src/core/integrity/restorer.py
from __future__ import annotations
from core.models.document import Document
from core.watermark.hybrid import HybridConfig, extract_redundant_text


# Восстановление текста из резервного слоя ЦВЗ
def restore_text_from_watermark(doc: Document, cfg: HybridConfig) -> tuple[str, str]:
    # Возвращает восстановленный текст и отладочную информацию
    text, dbg = extract_redundant_text(doc, cfg)
    return text, dbg