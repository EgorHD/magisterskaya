# src/app/ui/tab_verify.py
from __future__ import annotations

import os
import time
from pathlib import Path

from PyQt6.QtWidgets import QHBoxLayout, QSpinBox, QDoubleSpinBox

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFileDialog, QMessageBox, QLabel, QCheckBox

from app.ui.widgets import ActionsBar, FileInfoPanel, TextPanel
from app.ui.ocr_worker import OCRWorker
from app.ui.settings import AppSettings

from core.io.loader import load_document
from core.io.errors import DocumentLoadError

from core.models.document import Document
from core.models.ocr import OCRResult

from core.watermark.hybrid import HybridConfig
from core.integrity.restorer import restore_text_from_watermark
from core.integrity.diff import diff_words, format_diffs

from core.metrics.text_metrics import compute_text_similarity


OPEN_FILTER = "Документы (*.pdf *.tif *.tiff *.jpg *.jpeg *.png);;Все файлы (*.*)"
SAVE_REPORT_FILTER = "Отчёт (*.txt);;Все файлы (*.*)"


class VerifyTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._settings = AppSettings()

        self._current_path: str | None = None
        self._document: Document | None = None

        self._ocr_thread: QThread | None = None
        self._ocr_worker: OCRWorker | None = None

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

        self.chk_table_mode = QCheckBox("Табличный режим OCR (лучше для счетов/таблиц)")
        self.chk_table_mode.setChecked(False)

        # --- Лёгкий режим ЦВЗ ---
        self.chk_wm_capacity = QCheckBox("Настраиваемый режим ЦВЗ")
        self.chk_wm_capacity.setChecked(True)

        row = QHBoxLayout()
        row.addWidget(self.chk_wm_capacity)

        self.sb_lr = QSpinBox()
        self.sb_lr.setRange(1, 10)
        self.sb_lr.setValue(1)
        self.sb_lr.setPrefix("LR=")
        row.addWidget(self.sb_lr)

        self.sb_lm = QSpinBox()
        self.sb_lm.setRange(0, 400)
        self.sb_lm.setValue(0)
        self.sb_lm.setPrefix("LM=")
        row.addWidget(self.sb_lm)

        self.sb_ld = QDoubleSpinBox()
        self.sb_ld.setRange(1.0, 80.0)
        self.sb_ld.setDecimals(1)
        self.sb_ld.setSingleStep(1.0)
        self.sb_ld.setValue(18.0)
        self.sb_ld.setPrefix("LD=")
        row.addWidget(self.sb_ld)

        layout.addLayout(row)

        self._set_light_controls_enabled(self.chk_wm_capacity.isChecked())
        self.chk_wm_capacity.toggled.connect(self._set_light_controls_enabled)

        self.ocr_panel = TextPanel("Распознанный текст (OCR)")
        self.report_panel = TextPanel("Отчёт (включая метрики)")
        self.restored_panel = TextPanel("Восстановленный текст (опционально)")

        layout.addWidget(self.file_panel)
        layout.addWidget(self.actions)
        layout.addWidget(self.status)
        layout.addWidget(self.wm_status)
        layout.addWidget(self.chk_table_mode)
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
        start_dir = self._settings.last_dir()
        path, _ = QFileDialog.getOpenFileName(self, "Выберите файл ЭОД", start_dir, OPEN_FILTER, options=options)
        if not path:
            return

        self._settings.set_last_dir(os.path.dirname(path))

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

        pdf_dpi = 400 if self.chk_table_mode.isChecked() else 300

        try:
            doc = load_document(Path(path), pdf_dpi=pdf_dpi)
            self._document = doc
        except DocumentLoadError as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))
            self._reset_ui()
            return

        self._set_busy(True)
        self._start_ocr(self._document)

    def _set_light_controls_enabled(self, enabled: bool) -> None:
        self.sb_lr.setEnabled(enabled)
        self.sb_lm.setEnabled(enabled)
        self.sb_ld.setEnabled(enabled)

    def _start_ocr(self, doc: Document) -> None:
        self._ocr_thread = QThread(self)
        self._ocr_worker = OCRWorker(doc, lang="ru", table_mode=self.chk_table_mode.isChecked())
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

    def on_save_report_clicked(self) -> None:
        if not self._report_ready or not self._report_text:
            return

        options = QFileDialog.Option.DontUseNativeDialog
        start_dir = self._settings.last_dir()

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт", os.path.join(start_dir, "report.txt"), SAVE_REPORT_FILTER, options=options
        )
        if not save_path:
            return

        self._settings.set_last_dir(os.path.dirname(save_path))

        try:
            Path(save_path).write_text(self._report_text, encoding="utf-8")
            QMessageBox.information(self, "Отчёт", f"Отчёт сохранён:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"{type(e).__name__}: {e}")

    # ---- остальное оставь как у тебя (verify/restore/_reset_ui/_set_busy) ----
    def on_verify_clicked(self) -> None:
        cfg = HybridConfig()
        cfg.capacity_mode = self.chk_wm_capacity.isChecked()

        if cfg.capacity_mode:
            cfg.light_repetition = int(self.sb_lr.value())
            cfg.light_margin_px = int(self.sb_lm.value())
            cfg.light_delta = float(self.sb_ld.value())
        if not self._document:
            return
        if any(p.ocr_result is None for p in self._document.pages):
            QMessageBox.warning(self, "Проверка", "Сначала дождитесь завершения OCR.")
            return

        cfg = HybridConfig()
        cfg.capacity_mode = self.chk_wm_capacity.isChecked()

        t0 = time.perf_counter()

        # 1) Эталонный текст из резервного слоя (частотного слоя ЦВЗ)
        res = restore_text_from_watermark(self._document, cfg)
        if isinstance(res, tuple):
            ref_text, dbg = res
        else:
            ref_text, dbg = res, ""

        if not ref_text:
            self.wm_status.setText("ЦВЗ: не извлечён (нет эталонного текста)")
            self.report_panel.set_text("Не удалось извлечь эталонный текст из ЦВЗ.\n\n" + (dbg or ""))
            self._report_ready = False
            self.actions.btn_save.setEnabled(False)
            QMessageBox.warning(self, "Проверка целостности",
                                "Не удалось извлечь эталонный текст из резервного слоя ЦВЗ.")
            return

        self.wm_status.setText("ЦВЗ: извлечён (эталонный текст получен)")

        # 2) Текущий текст (OCR)
        ocr_text = self._ocr_full_text or ""

        # 3) Метрики текста (CER/WER)
        sim = compute_text_similarity(ocr_text, ref_text)

        # 4) Отличия (токены/слова)
        diffs = diff_words(ocr_text, ref_text, max_items=50)

        elapsed = time.perf_counter() - t0

        # 5) Формируем отчёт
        lines = []
        lines.append("ОТЧЁТ О ПРОВЕРКЕ ЦЕЛОСТНОСТИ ЭОД")
        lines.append("=" * 40)
        lines.append(f"Файл: {self._current_path or '(неизвестно)'}")
        lines.append(f"Страниц: {len(self._document.pages)}")
        lines.append("")
        lines.append("Статус ЦВЗ: извлечён (резервный слой)")
        lines.append("")
        lines.append("Метрики сходства текста (OCR vs эталон из ЦВЗ):")
        lines.append(f"- CER: {sim.cer:.6f} (dist={sim.char_distance}, ref_chars={sim.ref_chars})")
        lines.append(f"- WER: {sim.wer:.6f} (dist={sim.word_distance}, ref_words={sim.ref_words})")
        lines.append("")
        lines.append(f"Отличия (первые {min(len(diffs), 50)}):")
        lines.append(format_diffs(diffs))
        lines.append("")
        lines.append(f"Время проверки: {elapsed:.3f} сек.")
        if dbg:
            lines.append("")
            lines.append("Диагностика извлечения:")
            lines.append(dbg)

        report = "\n".join(lines).strip()

        self._report_text = report
        self.report_panel.set_text(report)
        self._report_ready = True
        self.actions.btn_save.setEnabled(True)

        if not diffs:
            QMessageBox.information(self, "Проверка целостности",
                                    "Подмен/изменений не обнаружено. Отчёт можно сохранить.")
        else:
            QMessageBox.warning(
                self,
                "Проверка целостности",
                f"Нарушение целостности обнаружено (отличий: {len(diffs)}).\n"
                "См. отчёт ниже — его можно сохранить кнопкой «Сохранить»."
            )

    def on_restore_clicked(self) -> None:
        cfg = HybridConfig()
        cfg.capacity_mode = self.chk_wm_capacity.isChecked()

        if cfg.capacity_mode:
            cfg.light_repetition = int(self.sb_lr.value())
            cfg.light_margin_px = int(self.sb_lm.value())
            cfg.light_delta = float(self.sb_ld.value())
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

    def _set_busy(self, busy: bool) -> None:
        self.actions.btn_load.setEnabled(not busy)
        self.chk_table_mode.setEnabled(not busy)
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