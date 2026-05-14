from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from core.models.document import Document, Page
from core.models.ocr import OCRPageResult, OCRResult


# Колбэк прогресса: текущая страница и общее число страниц
ProgressCallback = Callable[[int, int], None]


class OCREngine(ABC):
    # Название OCR-движка
    name: str

    # Распознавание одной страницы
    @abstractmethod
    def recognize_page(self, page: Page) -> OCRPageResult:
        raise NotImplementedError

    # Распознавание всего документа
    def recognize_document(
        self,
        doc: Document,
        progress_cb: Optional[ProgressCallback] = None,
    ) -> OCRResult:
        pages_out: list[OCRPageResult] = []
        total = len(doc.pages)

        for i, page in enumerate(doc.pages):
            # Обновление прогресса перед обработкой страницы
            if progress_cb:
                progress_cb(i, total)

            # OCR одной страницы
            pr = self.recognize_page(page)
            pages_out.append(pr)

            # Синхронизация данных страницы
            page.ocr_result = pr
            page.ocr_text = pr.text()

        # Финальное обновление прогресса
        if progress_cb:
            progress_cb(total, total)

        return OCRResult(pages=pages_out)