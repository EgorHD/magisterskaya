from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from core.models.ocr import OCRPageResult, OCRResult
from core.models.document import Document, Page


ProgressCallback = Callable[[int, int], None]  # (current_page_index, total_pages)


class OCREngine(ABC):
    name: str

    @abstractmethod
    def recognize_page(self, page: Page) -> OCRPageResult:
        """
        Распознаёт одну страницу. На входе page.image (PIL.Image).
        """
        raise NotImplementedError

    def recognize_document(self, doc: Document, progress_cb: Optional[ProgressCallback] = None) -> OCRResult:
        pages_out: list[OCRPageResult] = []
        total = len(doc.pages)

        for i, page in enumerate(doc.pages):
            if progress_cb:
                progress_cb(i, total)

            pr = self.recognize_page(page)
            pages_out.append(pr)

            # синхронизируем удобные поля
            page.ocr_result = pr
            page.ocr_text = pr.text()

        if progress_cb:
            progress_cb(total, total)

        return OCRResult(pages=pages_out)