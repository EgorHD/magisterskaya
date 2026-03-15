# src/app/ui/ocr_worker.py
from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from core.models.document import Document
from core.models.ocr import OCRResult, OCRPageResult
from core.ocr.factory import get_ocr_engine


class OCRWorker(QObject):
    progress = pyqtSignal(int, int)         # current, total
    finished = pyqtSignal(object)           # OCRResult
    failed = pyqtSignal(str)

    def __init__(self, doc: Document, *, lang: str = "ru", table_mode: bool = False) -> None:
        super().__init__()
        self.doc = doc
        self.lang = lang
        self.table_mode = table_mode

    @pyqtSlot()
    def run(self) -> None:
        try:
            engine = get_ocr_engine(lang=self.lang, use_angle_cls=True, use_gpu=False, table_mode=self.table_mode)

            def cb(cur: int, total: int) -> None:
                self.progress.emit(cur, total)

            result = engine.recognize_document(self.doc, progress_cb=cb)
            self.finished.emit(result)

        except Exception as e:
            if str(e).strip() == "0":
                empty_pages = [OCRPageResult(page_index=p.index, spans=[]) for p in self.doc.pages]
                self.finished.emit(OCRResult(pages=empty_pages))
                return

            self.failed.emit(f"{type(e).__name__}: {e!r}")