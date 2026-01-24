from __future__ import annotations

from dataclasses import dataclass, field

from core.models.document import Document
from core.watermark.codec import compress_text, decompress_text, sha256_trunc

from core.watermark.spatial_word_lsb import (
    SpatialWordLSBConfig,
    embed_hashes_in_words,
    extract_hashes_from_words,
)

from core.watermark.freq_dct import (
    DCTConfig,
    embed_bytes_dct,
    extract_bytes_dct,
)


@dataclass(slots=True)
class HybridConfig:
    """
    Реально гибридный ЦВЗ:
    - spatial: LSB в bbox слов (хрупкий слой для локализации подмены)
    - freq: DCT в полях/фоне (резервный слой для восстановления)
    """
    spatial: SpatialWordLSBConfig = field(default_factory=SpatialWordLSBConfig)
    freq: DCTConfig = field(default_factory=DCTConfig)

    # сколько байт хеша на слово/спан
    word_hash_bytes: int = 4

    # лимит резервного текста (после zlib)
    max_redundant_bytes: int = 120_000


@dataclass(slots=True)
class HybridExtractResult:
    """
    Совместимость со старым интерфейсом:
    primary — не используем (оставляем пустым),
    redundant — байты резервного текста (из частотного слоя)
    """
    ok: bool
    primary: bytes
    redundant: bytes
    error: str | None = None


@dataclass(slots=True)
class HybridVerifyPage:
    ok: bool
    changed_indices: list[int]
    error: str | None = None


def _sorted_spans(page) -> list:
    spans = list(page.ocr_result.spans)
    spans.sort(key=lambda sp: (sp.bbox()[1], sp.bbox()[0]))
    return spans


def embed_document(doc: Document, cfg: HybridConfig) -> Document:
    """
    Гибрид:
    - Spatial layer: хеши слов в bbox слов (LSB в области текста) — хрупко, локализация подмены
    - Frequency layer: резервный текст в DCT в полях/фоне — для восстановления
    """
    # резервный текст (весь документ)
    full_text = "\n".join((p.ocr_text or "").strip() for p in doc.pages)
    redundant = compress_text(full_text)
    if len(redundant) > cfg.max_redundant_bytes:
        redundant = redundant[: cfg.max_redundant_bytes]

    for page in doc.pages:
        if page.image is None or page.ocr_result is None:
            continue

        spans = _sorted_spans(page)

        # 1) spatial: для каждого span хеш
        hashes = [sha256_trunc(sp.text, nbytes=cfg.word_hash_bytes) for sp in spans]
        page.image = embed_hashes_in_words(page.image, spans, hashes, cfg.spatial)

        # 2) freq: встраиваем redundant в поля страницы
        page.image = embed_bytes_dct(page.image, redundant, cfg.freq)

    return doc


def verify_page(doc: Document, page_index: int, cfg: HybridConfig) -> HybridVerifyPage:
    """
    Проверка целостности по spatial-слою (bbox слов).
    Возвращает индексы span'ов (в отсортированном порядке), где хеш не совпал.
    """
    page = doc.pages[page_index]
    if page.image is None or page.ocr_result is None:
        return HybridVerifyPage(False, [], "No image or OCR")

    spans = _sorted_spans(page)
    extracted = extract_hashes_from_words(page.image, spans, cfg.word_hash_bytes, cfg.spatial)

    changed: list[int] = []
    for i, sp in enumerate(spans):
        exp = sha256_trunc(sp.text, nbytes=cfg.word_hash_bytes)
        got = extracted[i] if i < len(extracted) else b""
        if (not got) or (got != exp):
            changed.append(i)

    return HybridVerifyPage(ok=(len(changed) == 0), changed_indices=changed, error=None)


def extract_from_page(doc: Document, page_index: int, cfg: HybridConfig) -> HybridExtractResult:
    """
    СОВМЕСТИМОСТЬ ДЛЯ UI:
    Извлекаем резервный слой (частотный DCT) с конкретной страницы.
    """
    page = doc.pages[page_index]
    if page.image is None:
        return HybridExtractResult(False, b"", b"", "No image")

    try:
        redundant = extract_bytes_dct(page.image, cfg.freq)
        return HybridExtractResult(True, b"", redundant, None)
    except Exception as e:
        return HybridExtractResult(False, b"", b"", f"Extract error: {type(e).__name__}: {e}")


def extract_redundant_text_from_any_page(doc: Document, cfg: HybridConfig) -> str:
    """
    Резервный текст читаем из частотного слоя (DCT) — пытаемся с любой страницы.
    """
    for i in range(len(doc.pages)):
        ext = extract_from_page(doc, i, cfg)
        if not ext.ok or not ext.redundant:
            continue
        try:
            return decompress_text(ext.redundant)
        except Exception:
            continue
    return ""
