from __future__ import annotations

from core.models.document import Document
from core.integrity.report import IntegrityReport
from core.watermark.hybrid import HybridConfig, extract_from_page, build_primary_for_page


def verify_document(doc: Document, cfg: HybridConfig) -> IntegrityReport:
    broken_pages: list[int] = []
    broken_spans: dict[int, list[int]] = {}

    for pi, page in enumerate(doc.pages):
        if page.ocr_result is None:
            continue

        ext = extract_from_page(doc, pi, cfg)
        if not ext.ok:
            broken_pages.append(pi)
            broken_spans[pi] = []
            continue

        spans = list(page.ocr_result.spans)
        spans.sort(key=lambda sp: (sp.bbox()[1], sp.bbox()[0]))
        words = [sp.text for sp in spans]

        primary_now = build_primary_for_page(words, cfg)
        primary_wm = ext.primary

        hb = cfg.word_hash_bytes
        cnt = min(len(primary_now), len(primary_wm)) // hb

        bad: list[int] = []
        for i in range(cnt):
            a = primary_now[i * hb:(i + 1) * hb]
            b = primary_wm[i * hb:(i + 1) * hb]
            if a != b:
                bad.append(i)

        if bad:
            broken_pages.append(pi)
            broken_spans[pi] = bad

    ok = len(broken_pages) == 0
    msg = "Целостность не нарушена." if ok else f"Нарушение целостности обнаружено (страниц: {len(broken_pages)})."
    return IntegrityReport(ok=ok, broken_pages=broken_pages, broken_spans=broken_spans, message=msg)
