from __future__ import annotations
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from core.models.document import Document
from core.models.ocr import OCRResult, OCRPageResult
from core.ocr.factory import get_ocr_engine

class OCRWorker(QObject):
    # Сигнал прогресса: текущая страница / общее количество
    progress = pyqtSignal(int, int)

    # Сигнал успешного завершения OCR
    finished = pyqtSignal(object)

    # Сигнал ошибки
    failed = pyqtSignal(str)

    def __init__(
        self,
        doc: Document,
        *,
        lang: str = "ru",
        table_mode: bool = False,
    ) -> None:
        super().__init__()

        # Документ для распознавания
        self.doc = doc

        # Язык OCR
        self.lang = lang

        # Режим обработки таблиц
        self.table_mode = table_mode

    @pyqtSlot()
    def run(self) -> None:
        try:
            # Создание OCR-движка
            engine = get_ocr_engine(
                lang=self.lang,
                use_angle_cls=True,
                use_gpu=False,
                table_mode=self.table_mode,
            )

            # Колбэк для обновления прогресса
            def cb(cur: int, total: int) -> None:
                self.progress.emit(cur, total)

            # Распознавание документа
            result = engine.recognize_document(self.doc, progress_cb=cb)

            # Отправка результата
            self.finished.emit(result)

        except Exception as e:
            # Если OCR вернул пустой результат
            if str(e).strip() == "0":
                empty_pages = [
                    OCRPageResult(page_index=page.index, spans=[])
                    for page in self.doc.pages
                ]
                self.finished.emit(OCRResult(pages=empty_pages))
                return

            # Отправка текста ошибки
            self.failed.emit(f"{type(e).__name__}: {e!r}")