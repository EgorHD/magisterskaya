from __future__ import annotations
from core.integrity.report import IntegrityReport
from core.models.document import Document
from core.watermark.hybrid import HybridConfig, build_primary_for_page, extract_from_page


# Проверка целостности документа
def verify_document(doc: Document, cfg: HybridConfig) -> IntegrityReport:
    # Список повреждённых страниц
    broken_pages: list[int] = []

    # Повреждённые спаны по страницам
    broken_spans: dict[int, list[int]] = {}

    for pi, page in enumerate(doc.pages):
        # Если OCR ещё не выполнен, страницу пропускаем
        if page.ocr_result is None:
            continue

        # Извлечение ЦВЗ со страницы
        ext = extract_from_page(doc, pi, cfg)

        # Если ЦВЗ не извлечён, считаем страницу повреждённой
        if not ext.ok:
            broken_pages.append(pi)
            broken_spans[pi] = []
            continue

        # Получение OCR-спанов страницы
        spans = list(page.ocr_result.spans)

        # Сортировка спанов по положению на странице
        spans.sort(key=lambda sp: (sp.bbox()[1], sp.bbox()[0]))

        # Извлечение слов из спанов
        words = [sp.text for sp in spans]

        # Формирование текущего первичного слоя
        primary_now = build_primary_for_page(words, cfg)

        # Первичный слой, извлечённый из ЦВЗ
        primary_wm = ext.primary

        # Размер одного хеша слова
        hb = cfg.word_hash_bytes

        # Количество сравниваемых слов
        cnt = min(len(primary_now), len(primary_wm)) // hb

        # Индексы повреждённых слов
        bad: list[int] = []

        for i in range(cnt):
            a = primary_now[i * hb:(i + 1) * hb]
            b = primary_wm[i * hb:(i + 1) * hb]

            # Если хеши не совпадают, слово считается изменённым
            if a != b:
                bad.append(i)

        # Если найдены отличия, страница считается повреждённой
        if bad:
            broken_pages.append(pi)
            broken_spans[pi] = bad

    # Итоговый статус проверки
    ok = len(broken_pages) == 0

    # Текстовое сообщение
    msg = (
        "Целостность не нарушена."
        if ok
        else f"Нарушение целостности обнаружено (страниц: {len(broken_pages)})."
    )

    return IntegrityReport(
        ok=ok,
        broken_pages=broken_pages,
        broken_spans=broken_spans,
        message=msg,
    )