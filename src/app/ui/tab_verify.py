from __future__ import annotations

import os
import time
from pathlib import Path

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.ui.ocr_worker import OCRWorker
from app.ui.page_preview import PagePreview
from app.ui.settings import AppSettings
from app.ui.widgets import ActionsBar, FileInfoPanel, StatusProgressBlock, TextPanel
from core.integrity.diff import diff_words, format_diffs
from core.integrity.restorer import restore_text_from_watermark
from core.io.errors import DocumentLoadError
from core.io.loader import load_document
from core.metrics.text_metrics import compute_text_similarity
from core.models.document import Document
from core.models.ocr import OCRResult
from core.watermark.hybrid import HybridConfig


# Фильтр открытия документов
OPEN_FILTER = "Документы (*.pdf *.tif *.tiff *.jpg *.jpeg *.png);;Все файлы (*.*)"


class VerifyTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Настройки приложения
        self._settings = AppSettings()

        # Текущий путь к файлу
        self._current_path: str | None = None

        # Загруженный документ
        self._document: Document | None = None

        # Поток OCR
        self._ocr_thread: QThread | None = None
        self._ocr_worker: OCRWorker | None = None

        # Полный текст OCR
        self._ocr_full_text: str = ""

        # Главный горизонтальный layout
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # Левая колонка
        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)

        # Правая колонка
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)

        # Панель информации о файле
        self.file_panel = FileInfoPanel("Проверяемый файл")

        # Панель действий
        self.actions = ActionsBar()
        self.actions.set_primary_text("Проверить целостность")
        self.actions.set_secondary_text("—")

        # Блок статуса
        self.status_block = StatusProgressBlock()

        # Режим OCR для таблиц
        self.chk_table_mode = QCheckBox("Табличный режим OCR")
        self.chk_table_mode.setChecked(False)

        # Режим настройки параметров ЦВЗ
        self.chk_wm_capacity = QCheckBox("Настраиваемый режим ЦВЗ")
        self.chk_wm_capacity.setChecked(True)

        # Строка параметров ЦВЗ
        row = QHBoxLayout()
        row.setSpacing(10)
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

        # Выбор страницы
        self.page_selector = QComboBox()
        self.page_selector.setEnabled(False)
        self.page_selector.currentIndexChanged.connect(self._on_page_changed)

        # Просмотр документа
        self.preview = PagePreview()

        # Текстовые панели
        self.ocr_panel = TextPanel("Распознанный текст")
        self.report_panel = TextPanel("Отчёт проверки")

        # Можно сделать текстовые блоки чуть компактнее
        self.ocr_panel.setMaximumHeight(190)
        self.report_panel.setMaximumHeight(220)

        # Сборка левой колонки
        left_layout.addWidget(self.file_panel)
        left_layout.addWidget(self.actions)
        left_layout.addWidget(self.status_block)
        left_layout.addWidget(self.chk_table_mode)
        left_layout.addLayout(row)
        left_layout.addWidget(self.page_selector)
        left_layout.addWidget(self.ocr_panel)
        left_layout.addWidget(self.report_panel)

        # Сборка правой колонки
        right_layout.addWidget(self.preview, stretch=1)

        # Добавление колонок в главный layout
        root.addLayout(left_layout, 2)
        root.addLayout(right_layout, 3)

        # Начальное состояние полей параметров
        self._set_light_controls_enabled(self.chk_wm_capacity.isChecked())
        self.chk_wm_capacity.toggled.connect(self._set_light_controls_enabled)

        # Кнопки до загрузки файла
        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_action_secondary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)

        # Скрываем неиспользуемые кнопки
        self.actions.btn_action_secondary.hide()
        self.actions.btn_save.hide()

        # Подключение кнопок
        self.actions.btn_load.clicked.connect(self.on_load_clicked)
        self.actions.btn_action_primary.clicked.connect(self.on_verify_clicked)

    # Включение и выключение полей параметров
    def _set_light_controls_enabled(self, enabled: bool) -> None:
        self.sb_lr.setEnabled(enabled)
        self.sb_lm.setEnabled(enabled)
        self.sb_ld.setEnabled(enabled)

    # Загрузка документа
    def on_load_clicked(self) -> None:
        options = QFileDialog.Option.DontUseNativeDialog
        start_dir = self._settings.last_dir()

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл ЭОД",
            start_dir,
            OPEN_FILTER,
            options=options,
        )
        if not path:
            return

        self._settings.set_last_dir(os.path.dirname(path))

        # Сброс состояния
        self._current_path = path
        self._document = None
        self._ocr_full_text = ""

        # Обновление интерфейса
        self.file_panel.set_path(path)
        self.status_block.set_status("Готово")
        self.status_block.hide_progress()
        self.ocr_panel.set_text("Запуск OCR...")
        self.report_panel.clear()

        self.page_selector.blockSignals(True)
        self.page_selector.clear()
        self.page_selector.setEnabled(False)
        self.page_selector.blockSignals(False)

        try:
            self.preview.clear()
        except Exception:
            pass

        self.actions.btn_action_primary.setEnabled(False)

        # DPI для PDF зависит от режима OCR
        pdf_dpi = 400 if self.chk_table_mode.isChecked() else 300

        try:
            doc = load_document(Path(path), pdf_dpi=pdf_dpi)
            self._document = doc
        except DocumentLoadError as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))
            self._reset_ui()
            return
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", f"{type(e).__name__}: {e}")
            self._reset_ui()
            return

        # Показываем первую страницу до OCR
        if self._document.pages and self._document.pages[0].image is not None:
            try:
                self.preview.set_page(self._document.pages[0].image, None)
            except Exception:
                pass

        # Запуск OCR
        self._set_busy(True)
        self.status_block.set_progress("OCR: подготовка...", 0, 0)
        self._start_ocr(self._document)

    # Запуск OCR в отдельном потоке
    def _start_ocr(self, doc: Document) -> None:
        self._ocr_thread = QThread(self)
        self._ocr_worker = OCRWorker(
            doc,
            lang="ru",
            table_mode=self.chk_table_mode.isChecked(),
        )
        self._ocr_worker.moveToThread(self._ocr_thread)

        # Сигналы OCR
        self._ocr_thread.started.connect(self._ocr_worker.run)
        self._ocr_worker.progress.connect(self._on_ocr_progress)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.failed.connect(self._on_ocr_failed)

        # Завершение потока
        self._ocr_worker.finished.connect(self._ocr_thread.quit)
        self._ocr_worker.failed.connect(self._ocr_thread.quit)
        self._ocr_thread.finished.connect(self._ocr_worker.deleteLater)
        self._ocr_thread.finished.connect(self._ocr_thread.deleteLater)

        self._ocr_thread.start()

    # Обновление прогресса OCR
    def _on_ocr_progress(self, cur: int, total: int) -> None:
        if total <= 0:
            self.status_block.set_progress("OCR: обработка...", 0, 0)
            return

        if cur >= total:
            self.status_block.set_progress("OCR: завершение...", total, total)
        else:
            self.status_block.set_progress(
                f"OCR: страница {cur + 1} из {total}",
                cur + 1,
                total,
            )

    # Успешное завершение OCR
    def _on_ocr_finished(self, result_obj: object) -> None:
        result: OCRResult = result_obj  # type: ignore

        self.status_block.set_status("OCR: готово")
        self.status_block.hide_progress()
        self.ocr_panel.set_text(result.text() or "(Текст не распознан)")

        if self._document:
            # Сбор полного текста
            self._ocr_full_text = "\n".join((p.ocr_text or "") for p in self._document.pages)

            # Заполнение списка страниц
            self.page_selector.blockSignals(True)
            self.page_selector.clear()

            for i in range(len(self._document.pages)):
                self.page_selector.addItem(f"Страница {i + 1}")

            self.page_selector.setEnabled(len(self._document.pages) > 1)
            self.page_selector.setCurrentIndex(0)
            self.page_selector.blockSignals(False)

            # Отрисовка первой страницы
            self._render_page(0)

        self.actions.btn_action_primary.setEnabled(True)
        self._set_busy(False)

    # Ошибка OCR
    def _on_ocr_failed(self, message: str) -> None:
        self.status_block.set_status("OCR: ошибка")
        self.status_block.hide_progress()
        self.ocr_panel.set_text("OCR не выполнен\n\n" f"Причина: {message}")
        self.actions.btn_action_primary.setEnabled(False)
        self._set_busy(False)

    # Переключение страницы
    def _on_page_changed(self, idx: int) -> None:
        if idx < 0:
            return
        self._render_page(idx)

    # Отрисовка выбранной страницы
    def _render_page(self, idx: int) -> None:
        if not self._document or idx >= len(self._document.pages):
            return

        page = self._document.pages[idx]

        # Показ страницы с OCR-обводкой
        if page.image is not None:
            self.preview.set_page(page.image, page.ocr_result)

        # Показ текста текущей страницы
        if page.ocr_result and page.ocr_result.spans:
            self.ocr_panel.set_text("\n".join(sp.text for sp in page.ocr_result.spans))
        else:
            self.ocr_panel.set_text("(Текст не распознан)")

    # Проверка целостности документа
    def on_verify_clicked(self) -> None:
        cfg = HybridConfig()
        cfg.capacity_mode = self.chk_wm_capacity.isChecked()

        # Настройка параметров ЦВЗ
        if cfg.capacity_mode:
            cfg.light_repetition = int(self.sb_lr.value())
            cfg.light_margin_px = int(self.sb_lm.value())
            cfg.light_delta = float(self.sb_ld.value())

        if not self._document:
            return

        if any(p.ocr_result is None for p in self._document.pages):
            QMessageBox.warning(self, "Проверка", "Сначала дождитесь завершения OCR")
            return

        self.status_block.set_busy_indeterminate("Проверка целостности...")
        QApplication.processEvents()

        t0 = time.perf_counter()

        # Извлечение эталонного текста из ЦВЗ
        res = restore_text_from_watermark(self._document, cfg)
        if isinstance(res, tuple):
            ref_text, dbg = res
        else:
            ref_text, dbg = res, ""

        # Если извлечение не удалось
        if not ref_text:
            self.report_panel.set_text("Не удалось извлечь эталонный текст из ЦВЗ\n\n" + (dbg or ""))
            self.status_block.set_status("Проверка завершена с ошибкой")
            self.status_block.hide_progress()

            QMessageBox.warning(
                self,
                "Проверка целостности",
                "Не удалось извлечь эталонный текст из резервного слоя ЦВЗ",
            )
            return

        # Сравнение OCR и эталонного текста
        ocr_text = self._ocr_full_text or ""
        sim = compute_text_similarity(ocr_text, ref_text)
        diffs = diff_words(ocr_text, ref_text, max_items=50)
        elapsed = time.perf_counter() - t0

        # Формирование отчёта
        lines = []
        lines.append("ОТЧЁТ О ПРОВЕРКЕ ЦЕЛОСТНОСТИ ЭОД")
        lines.append("=" * 40)
        lines.append(f"Файл: {self._current_path or '(неизвестно)'}")
        lines.append(f"Страниц: {len(self._document.pages)}")
        lines.append("")
        lines.append("Метрики сходства текста")
        lines.append(f"- CER: {sim.cer:.6f} (dist={sim.char_distance}, ref_chars={sim.ref_chars})")
        lines.append(f"- WER: {sim.wer:.6f} (dist={sim.word_distance}, ref_words={sim.ref_words})")
        lines.append("")
        lines.append(f"Отличия (первые {min(len(diffs), 50)})")
        lines.append(format_diffs(diffs))
        lines.append("")
        lines.append(f"Время проверки: {elapsed:.3f} сек")

        if dbg:
            lines.append("")
            lines.append("Диагностика извлечения")
            lines.append(dbg)

        report = "\n".join(lines).strip()
        self.report_panel.set_text(report)

        self.status_block.set_status("Проверка завершена")
        self.status_block.hide_progress()

        # Итоговое сообщение
        if not diffs:
            QMessageBox.information(self, "Проверка целостности", "Подмен и изменений не обнаружено")
        else:
            QMessageBox.warning(
                self,
                "Проверка целостности",
                f"Нарушение целостности обнаружено\nОтличий: {len(diffs)}",
            )

    # Переключение интерфейса в занятый режим
    def _set_busy(self, busy: bool) -> None:
        self.actions.btn_load.setEnabled(not busy)
        self.chk_table_mode.setEnabled(not busy)
        self.page_selector.setEnabled(
            (not busy) and bool(self._document) and len(self._document.pages) > 1
        )

        if busy:
            self.actions.btn_action_primary.setEnabled(False)

    # Сброс интерфейса
    def _reset_ui(self) -> None:
        self.file_panel.clear()
        self.status_block.set_status("Готово")
        self.status_block.hide_progress()
        self.ocr_panel.clear()
        self.report_panel.clear()

        self.page_selector.clear()
        self.page_selector.setEnabled(False)

        try:
            self.preview.clear()
        except Exception:
            pass

        self.actions.btn_action_primary.setEnabled(False)