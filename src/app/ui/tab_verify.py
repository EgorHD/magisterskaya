from __future__ import annotations

import os
import time
from pathlib import Path

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFileDialog, QMessageBox, QLabel

from app.ui.widgets import ActionsBar, FileInfoPanel, TextPanel
from app.ui.ocr_worker import OCRWorker

from core.io.loader import load_document
from core.io.errors import DocumentLoadError

from core.models.document import Document
from core.models.ocr import OCRResult

from core.watermark.hybrid import HybridConfig
from core.integrity.restorer import restore_text_from_watermark
from core.integrity.diff import diff_words, format_diffs

# ✅ метрики текста (мягко, отдельно)
from core.metrics.text_metrics import compute_text_similarity


OPEN_FILTER = "Документы (*.pdf *.tif *.tiff *.jpg *.jpeg *.png);;Все файлы (*.*)"
SAVE_REPORT_FILTER = "Отчёт (*.txt);;Все файлы (*.*)"


class VerifyTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._current_path: str | None = None
        self._document: Document | None = None

        self._ocr_thread: QThread | None = None
        self._ocr_worker: OCRWorker | None = None

        # состояние
        self._ocr_full_text: str = ""
        self._report_ready: bool = False
        self._report_text: str = ""

        layout = QVBoxLayout(self)

        self.file_panel = FileInfoPanel("Проверяемый файл (защищённый ЭОД)")
        self.actions = ActionsBar()
        self.actions.set_primary_text("Проверить целостность")
        self.actions.set_secondary_text("Восстановить (показать текст)")

        self.status = QLabel("Готово")
        self.wm_status = QLabel("ЦВЗ: —")

        self.ocr_panel = TextPanel("Распознанный текст (OCR)")
        self.report_panel = TextPanel("Отчёт (включая метрики)")
        self.restored_panel = TextPanel("Восстановленный текст (опционально)")

        layout.addWidget(self.file_panel)
        layout.addWidget(self.actions)
        layout.addWidget(self.status)
        layout.addWidget(self.wm_status)
        layout.addWidget(self.ocr_panel, stretch=1)
        layout.addWidget(self.report_panel, stretch=1)
        layout.addWidget(self.restored_panel, stretch=1)

        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_action_secondary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)

        self.actions.btn_load.clicked.connect(self.on_load_clicked)
        self.actions.btn_action_primary.clicked.connect(self.on_verify_clicked)
        self.actions.btn_action_secondary.clicked.connect(self.on_restore_clicked)
        self.actions.btn_save.clicked.connect(self.on_save_report_clicked)

    def on_load_clicked(self) -> None:
        options = QFileDialog.Option.DontUseNativeDialog
        path, _ = QFileDialog.getOpenFileName(self, "Выберите файл ЭОД", "", OPEN_FILTER, options=options)
        if not path:
            return

        self._current_path = path
        self._document = None

        self._ocr_full_text = ""
        self._report_ready = False
        self._report_text = ""

        self.file_panel.set_path(path)
        self.status.setText("Готово")
        self.wm_status.setText("ЦВЗ: —")
        self.ocr_panel.set_text("Запуск OCR...")
        self.report_panel.clear()
        self.restored_panel.clear()

        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_action_secondary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)

        try:
            doc = load_document(Path(path), pdf_dpi=300)
            self._document = doc
        except DocumentLoadError as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))
            self._reset_ui()
            return

        self._set_busy(True)
        self._start_ocr(self._document)

    def _start_ocr(self, doc: Document) -> None:
        self._ocr_thread = QThread(self)
        self._ocr_worker = OCRWorker(doc, lang="ru")
        self._ocr_worker.moveToThread(self._ocr_thread)

        self._ocr_thread.started.connect(self._ocr_worker.run)
        self._ocr_worker.progress.connect(self._on_ocr_progress)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.failed.connect(self._on_ocr_failed)

        self._ocr_worker.finished.connect(self._ocr_thread.quit)
        self._ocr_worker.failed.connect(self._ocr_thread.quit)
        self._ocr_thread.finished.connect(self._ocr_worker.deleteLater)
        self._ocr_thread.finished.connect(self._ocr_thread.deleteLater)

        self._ocr_thread.start()

    def _on_ocr_progress(self, cur: int, total: int) -> None:
        if total <= 0:
            self.status.setText("OCR: обработка...")
            return
        if cur >= total:
            self.status.setText("OCR: завершение...")
        else:
            self.status.setText(f"OCR: страница {cur + 1} из {total}")

    def _on_ocr_finished(self, result_obj: object) -> None:
        result: OCRResult = result_obj  # type: ignore
        self.status.setText("OCR: готово")
        self.ocr_panel.set_text(result.text() or "(Текст не распознан)")

        if self._document:
            self._ocr_full_text = "\n".join((p.ocr_text or "") for p in self._document.pages)

        self.actions.btn_action_primary.setEnabled(True)
        self.actions.btn_action_secondary.setEnabled(True)
        self._set_busy(False)

    def _on_ocr_failed(self, message: str) -> None:
        self.status.setText("OCR: ошибка")
        self.ocr_panel.set_text("OCR не выполнен.\n\n" f"Причина: {message}")
        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_action_secondary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)
        self._set_busy(False)

    def on_verify_clicked(self) -> None:
        if not self._document:
            return
        if any(p.ocr_result is None for p in self._document.pages):
            QMessageBox.warning(self, "Проверка", "Сначала дождитесь завершения OCR.")
            return

        cfg = HybridConfig()

        t0 = time.perf_counter()

        # 1) Эталонный текст из резервного слоя
        res = restore_text_from_watermark(self._document, cfg)
        if isinstance(res, tuple):
            ref_text, dbg = res
        else:
            ref_text, dbg = res, ""

        if not ref_text:
            verify_time = time.perf_counter() - t0
            self._report_ready = True
            self._report_text = (
                "Итог: ЦВЗ НЕ читается.\n\n"
                "Не удалось извлечь резервный текст из ЦВЗ.\n"
                + (dbg or "")
                + f"\n\nВремя проверки: {verify_time:.3f} сек."
            )
            self.report_panel.set_text(self._report_text)
            self.wm_status.setText("ЦВЗ: НЕ читается")
            self.actions.btn_save.setEnabled(True)
            QMessageBox.warning(self, "Проверка целостности", "ЦВЗ не читается — отчёт сформирован.")
            return

        self.wm_status.setText("ЦВЗ: читается")

        # 2) Diff OCR vs эталон
        diffs = diff_words(self._ocr_full_text, ref_text, max_items=80)
        diff_text = format_diffs(diffs)

        # 3) Метрики по тексту (CER/WER)
        sim = compute_text_similarity(self._ocr_full_text, ref_text)

        verify_time = time.perf_counter() - t0

        metrics_block = (
            "Метрики (OCR vs эталон из ЦВЗ):\n"
            f"- CER: {sim.cer:.4f}  (dist={sim.char_distance}, ref_chars={sim.ref_chars})\n"
            f"- WER: {sim.wer:.4f}  (dist={sim.word_distance}, ref_words={sim.ref_words})\n"
            f"- Отличий (по diff): {len(diffs)}\n"
            f"- Время проверки: {verify_time:.3f} сек.\n"
        )

        if not diffs:
            summary = "Итог: ПОДМЕН/ИЗМЕНЕНИЙ НЕ ОБНАРУЖЕНО.\n"
        else:
            summary = "Итог: ОБНАРУЖЕНО нарушение целостности.\n"

        self._report_text = summary + "\n" + metrics_block + "\n" + diff_text
        self._report_ready = True
        self.report_panel.set_text(self._report_text)
        self.actions.btn_save.setEnabled(True)

        if not diffs:
            QMessageBox.information(self, "Проверка целостности", "Подмен/изменений не обнаружено. Отчёт можно сохранить.")
        else:
            QMessageBox.warning(
                self,
                "Проверка целостности",
                f"Нарушение целостности обнаружено (отличий: {len(diffs)}).\n"
                "См. отчёт ниже — его можно сохранить кнопкой «Сохранить»."
            )

    def on_restore_clicked(self) -> None:
        if not self._document:
            return

        cfg = HybridConfig()
        res = restore_text_from_watermark(self._document, cfg)
        if isinstance(res, tuple):
            text, dbg = res
        else:
            text, dbg = res, ""

        if not text:
            QMessageBox.warning(self, "Восстановление", "Не удалось восстановить текст.\n\n" + (dbg or ""))
            return

        self.restored_panel.set_text(text)
        QMessageBox.information(self, "Восстановление", "Текст извлечён из резервного слоя (показан ниже).")

    def on_save_report_clicked(self) -> None:
        if not self._report_ready:
            QMessageBox.information(self, "Сохранение", "Сначала выполните «Проверить целостность», чтобы сформировать отчёт.")
            return

        base_dir = os.path.dirname(self._current_path or "")
        base_name = os.path.splitext(os.path.basename(self._current_path or "document"))[0]
        default_name = f"{base_name}_report.txt"

        options = QFileDialog.Option.DontUseNativeDialog
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить отчёт",
            os.path.join(base_dir, default_name),
            SAVE_REPORT_FILTER,
            options=options,
        )
        if not save_path:
            return

        try:
            Path(save_path).write_text(self._report_text or "", encoding="utf-8", errors="ignore")
            QMessageBox.information(self, "Сохранение", f"Отчёт сохранён:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", f"{type(e).__name__}: {e}")

    def _set_busy(self, busy: bool) -> None:
        self.actions.btn_load.setEnabled(not busy)
        if busy:
            self.actions.btn_action_primary.setEnabled(False)
            self.actions.btn_action_secondary.setEnabled(False)
            self.actions.btn_save.setEnabled(False)

    def _reset_ui(self) -> None:
        self.file_panel.clear()
        self.status.setText("Готово")
        self.wm_status.setText("ЦВЗ: —")
        self.ocr_panel.clear()
        self.report_panel.clear()
        self.restored_panel.clear()
        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_action_secondary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)
