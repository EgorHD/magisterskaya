# src/core/watermark/hybrid.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

from core.models.document import Document
from core.watermark.codec import compress_text, decompress_text, sha256_trunc
from core.watermark.freq_dct import (
    DCTConfig,
    capacity_payload_bytes_dct,
    embed_bytes_dct,
    extract_bytes_dct,
)
from core.watermark.spatial_word_lsb import (
    SpatialWordLSBConfig,
    embed_hashes_in_words,
    extract_hashes_from_words,
)


@dataclass(slots=True)
class HybridConfig:
    # Пространственный слой
    spatial: SpatialWordLSBConfig = field(default_factory=SpatialWordLSBConfig)

    # Частотный слой
    freq: DCTConfig = field(default_factory=DCTConfig)

    # Число байт хеша на слово
    word_hash_bytes: int = 4

    # Максимальный размер резервного текста после сжатия
    max_redundant_bytes: int = 120_000

    # Режим повышенной ёмкости
    capacity_mode: bool = False

    # Параметры лёгкого режима
    light_repetition: int = 1
    light_margin_px: int = 0
    light_delta: float = 18.0


# Эффективная конфигурация DCT с учётом лёгкого режима
def _effective_freq_cfg(cfg: HybridConfig) -> DCTConfig:
    if cfg.capacity_mode:
        return replace(
            cfg.freq,
            repetition=int(cfg.light_repetition),
            margin_px=int(cfg.light_margin_px),
            delta=float(cfg.light_delta),
        )
    return cfg.freq


@dataclass(slots=True)
class HybridExtractResult:
    # Успешность извлечения
    ok: bool

    # Primary-слой
    primary: bytes

    # Redundant-слой
    redundant: bytes

    # Текст ошибки
    error: Optional[str] = None


@dataclass(slots=True)
class HybridVerifyPage:
    # Статус проверки страницы
    ok: bool

    # Индексы изменённых слов
    changed_indices: List[int]

    # Текст ошибки
    error: Optional[str] = None


# Сортировка OCR-спанов по порядку чтения
def _sorted_spans(page) -> list:
    spans = list(page.ocr_result.spans)
    spans.sort(key=lambda sp: (sp.bbox()[1], sp.bbox()[0]))
    return spans


# Сигнатура chunked payload
_MAGIC = b"CHNK"


# Упаковка одного чанка
def _pack_chunk(
    total_parts: int,
    part_index: int,
    full_len: int,
    full_sha8: bytes,
    chunk: bytes,
) -> bytes:
    return (
        _MAGIC
        + int(total_parts).to_bytes(2, "big")
        + int(part_index).to_bytes(2, "big")
        + int(full_len).to_bytes(4, "big")
        + full_sha8[:8]
        + chunk
    )


# Попытка распаковать chunked payload
def _try_unpack_chunk(payload: bytes):
    if not payload or len(payload) < 4 + 2 + 2 + 4 + 8:
        return None

    if payload[:4] != _MAGIC:
        return None

    total = int.from_bytes(payload[4:6], "big")
    idx = int.from_bytes(payload[6:8], "big")
    full_len = int.from_bytes(payload[8:12], "big")
    sha8 = payload[12:20]
    chunk = payload[20:]

    return total, idx, full_len, sha8, chunk


# Встраивание гибридного ЦВЗ в документ
def embed_document(doc: Document, cfg: HybridConfig) -> Document:
    # Общий OCR-текст документа
    full_text = "\n".join((p.ocr_text or "") for p in doc.pages).strip()

    # Сжатый резервный текст
    redundant = compress_text(full_text)
    if len(redundant) > cfg.max_redundant_bytes:
        redundant = redundant[: cfg.max_redundant_bytes]

    # Spatial-слой: хеши слов в каждую страницу
    for page in doc.pages:
        if page.image is None or page.ocr_result is None:
            continue

        spans = _sorted_spans(page)
        hashes = [sha256_trunc(sp.text, nbytes=cfg.word_hash_bytes) for sp in spans]
        page.image = embed_hashes_in_words(page.image, spans, hashes, cfg.spatial)

    # Обычный режим: один и тот же redundant на каждую страницу
    if not cfg.capacity_mode:
        for page in doc.pages:
            if page.image is None or page.ocr_result is None:
                continue
            page.image = embed_bytes_dct(page.image, redundant, cfg.freq)

        return doc

    # Лёгкий режим: дробление redundant по страницам
    freq_cfg = _effective_freq_cfg(cfg)
    full_sha8 = sha256_trunc(full_text, nbytes=8)
    full_len = len(redundant)

    # Ёмкости страниц
    page_caps: List[Tuple[int, int]] = []
    for i, page in enumerate(doc.pages):
        if page.image is None:
            continue

        cap = capacity_payload_bytes_dct(page.image, freq_cfg)
        page_caps.append((i, cap))

    if not page_caps:
        raise ValueError("Нет страниц с изображением для DCT-слоя")

    # Страницы с наибольшей ёмкостью первыми
    page_caps.sort(key=lambda t: t[1], reverse=True)

    # Размер заголовка chunk
    CH_OVERHEAD = 20

    # Эффективные ёмкости
    effective_caps: List[Tuple[int, int]] = []
    total_capacity = 0

    for pi, cap in page_caps:
        eff = max(0, cap - CH_OVERHEAD)
        if eff > 0:
            effective_caps.append((pi, eff))
            total_capacity += eff

    if total_capacity < len(redundant):
        raise ValueError(
            f"Недостаточно ёмкости даже в лёгком режиме: "
            f"нужно {len(redundant)}B, есть {total_capacity}B"
        )

    # Разбиение redundant на чанки
    chunks: List[Tuple[int, bytes]] = []
    pos = 0

    for pi, eff in effective_caps:
        if pos >= len(redundant):
            break

        chunk = redundant[pos:pos + eff]
        chunks.append((pi, chunk))
        pos += len(chunk)

    total_parts = len(chunks)

    # Запись чанков по страницам
    for part_idx, (pi, chunk) in enumerate(chunks):
        page = doc.pages[pi]
        if page.image is None:
            continue

        payload = _pack_chunk(total_parts, part_idx, full_len, full_sha8, chunk)
        page.image = embed_bytes_dct(page.image, payload, freq_cfg)

    return doc


# Проверка страницы по spatial-слою
def verify_page(doc: Document, page_index: int, cfg: HybridConfig) -> HybridVerifyPage:
    page = doc.pages[page_index]

    if page.image is None or page.ocr_result is None:
        return HybridVerifyPage(False, [], "No image or OCR")

    spans = _sorted_spans(page)
    extracted = extract_hashes_from_words(
        page.image,
        spans,
        cfg.word_hash_bytes,
        cfg.spatial,
    )

    changed: List[int] = []

    for i, sp in enumerate(spans):
        exp = sha256_trunc(sp.text, nbytes=cfg.word_hash_bytes)
        got = extracted[i] if i < len(extracted) else b""

        if (not got) or (got != exp):
            changed.append(i)

    return HybridVerifyPage(
        ok=(len(changed) == 0),
        changed_indices=changed,
        error=None,
    )


# Извлечение DCT-слоя с одной страницы
def extract_from_page(doc: Document, page_index: int, cfg: HybridConfig) -> HybridExtractResult:
    page = doc.pages[page_index]

    if page.image is None:
        return HybridExtractResult(False, b"", b"", "No image")

    try:
        freq_cfg = _effective_freq_cfg(cfg)
        redundant = extract_bytes_dct(page.image, freq_cfg)
        return HybridExtractResult(True, b"", redundant, None)
    except Exception as e:
        return HybridExtractResult(
            False,
            b"",
            b"",
            f"Extract error: {type(e).__name__}: {e}",
        )


# Обычное извлечение redundant с любой страницы
def extract_redundant_text_from_any_page(doc: Document, cfg: HybridConfig) -> str:
    for i in range(len(doc.pages)):
        ext = extract_from_page(doc, i, cfg)
        if not ext.ok or not ext.redundant:
            continue

        try:
            return decompress_text(ext.redundant)
        except Exception:
            continue

    return ""


# Сборка chunked redundant со всех страниц
def extract_chunked_redundant_text(doc: Document, cfg: HybridConfig) -> Tuple[str, str]:
    parts_by_key: dict[bytes, dict[int, bytes]] = {}
    meta_by_key: dict[bytes, Tuple[int, int]] = {}

    for i in range(len(doc.pages)):
        ext = extract_from_page(doc, i, cfg)
        if not ext.ok or not ext.redundant:
            continue

        parsed = _try_unpack_chunk(ext.redundant)
        if parsed is None:
            continue

        total, idx, full_len, sha8, chunk = parsed

        if total <= 0 or idx < 0 or idx >= total:
            continue

        if sha8 not in parts_by_key:
            parts_by_key[sha8] = {}
            meta_by_key[sha8] = (total, full_len)

        parts_by_key[sha8][idx] = chunk

    if not parts_by_key:
        return "", "FAIL: no chunked payloads found"

    for sha8, parts in parts_by_key.items():
        total, full_len = meta_by_key.get(sha8, (0, 0))
        if total <= 0:
            continue

        if len(parts) != total:
            return "", f"FAIL: chunks incomplete (have {len(parts)}/{total})"

        blob = b"".join(parts[k] for k in range(total))
        blob = blob[:full_len]

        try:
            text = decompress_text(blob)
            return text, f"OK: restored chunked redundant (parts={total}, bytes={len(blob)})"
        except Exception as e:
            return "", f"FAIL: chunked assembled but decompress failed: {type(e).__name__}: {e}"

    return "", "FAIL: chunked payloads exist, but none assembled"


# Универсальное извлечение redundant-текста
def extract_redundant_text(doc: Document, cfg: HybridConfig) -> Tuple[str, str]:
    # Сначала пробуем chunked-режим
    text, dbg = extract_chunked_redundant_text(doc, cfg)
    if text:
        return text, dbg

    # Потом обычное извлечение с любой страницы
    text2 = extract_redundant_text_from_any_page(doc, cfg)
    if text2:
        return text2, "OK: extracted non-chunk redundant from any page"

    return "", dbg if dbg else "FAIL: could not extract/decompress redundant from any page"