from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFileDialog, QMessageBox, QLabel, QComboBox
)

from app.ui.widgets import ActionsBar, FileInfoPanel, TextPanel
from app.ui.ocr_worker import OCRWorker
from app.ui.page_preview import PagePreview

from core.io.loader import load_document
from core.io.errors import DocumentLoadError
from core.io.saver import save_document

from core.models.document import Document, DocumentFormat
from core.models.ocr import OCRResult

from core.watermark.hybrid import HybridConfig, embed_document, extract_from_page


OPEN_FILTER = "Документы (*.pdf *.tif *.tiff *.jpg *.jpeg *.png);;Все файлы (*.*)"
SAVE_FILTER_LOSSLESS = "TIFF (*.tif *.tiff);;PNG (*.png);;JPEG (*.jpg *.jpeg)"


class ProtectTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._current_path: str | None = None
        self._document: Document | None = None
        self._is_watermarked: bool = False

        self._ocr_thread: QThread | None = None
        self._ocr_worker: OCRWorker | None = None

        layout = QVBoxLayout(self)

        self.file_panel = FileInfoPanel("Входной файл (ЭОД)")
        self.actions = ActionsBar()
        self.actions.set_primary_text("Встроить ЦВЗ")
        self.actions.set_secondary_text("—")

        self.status = QLabel("Готово")

        self.page_selector = QComboBox()
        self.page_selector.setEnabled(False)
        self.page_selector.currentIndexChanged.connect(self._on_page_changed)

        self.preview = PagePreview()
        self.text_panel = TextPanel("Распознанный текст (OCR)")

        layout.addWidget(self.file_panel)
        layout.addWidget(self.actions)
        layout.addWidget(self.status)
        layout.addWidget(self.page_selector)
        layout.addWidget(self.preview, stretch=2)
        layout.addWidget(self.text_panel, stretch=1)

        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_action_secondary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)

        self.actions.btn_load.clicked.connect(self.on_load_clicked)
        self.actions.btn_action_primary.clicked.connect(self.on_embed_clicked)
        self.actions.btn_save.clicked.connect(self.on_save_clicked)

    def on_load_clicked(self) -> None:
        options = QFileDialog.Option.DontUseNativeDialog
        path, _ = QFileDialog.getOpenFileName(self, "Выберите файл ЭОД", "", OPEN_FILTER, options=options)
        if not path:
            return

        self._current_path = path
        self._is_watermarked = False
        self._document = None
        self.page_selector.clear()
        self.page_selector.setEnabled(False)

        try:
            doc = load_document(Path(path), pdf_dpi=300)
            self._document = doc
        except DocumentLoadError as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))
            self._reset_ui()
            return

        self.file_panel.set_path(path)

        if self._document.pages and self._document.pages[0].image is not None:
            self.preview.set_page(self._document.pages[0].image, None)

        self.text_panel.set_text("Запуск OCR...")
        self._set_busy(True)
        self._start_ocr(self._document)

    def _start_ocr(self, doc: Document) -> None:
        if self._ocr_thread is not None and self._ocr_thread.isRunning():
            return

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

        if self._document:
            self.page_selector.blockSignals(True)
            self.page_selector.clear()
            for i in range(len(self._document.pages)):
                self.page_selector.addItem(f"Страница {i + 1}")
            self.page_selector.setEnabled(len(self._document.pages) > 1)
            self.page_selector.setCurrentIndex(0)
            self.page_selector.blockSignals(False)
            self._render_page(0)
        else:
            self.text_panel.set_text(result.text() or "(Текст не распознан)")

        self.actions.btn_action_primary.setEnabled(True)
        self.actions.btn_save.setEnabled(False)
        self._set_busy(False)

    def _on_ocr_failed(self, message: str) -> None:
        self.status.setText("OCR: ошибка")
        self.text_panel.set_text(f"OCR не выполнен.\n\nПричина: {message}")
        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)
        self._set_busy(False)

    def _on_page_changed(self, idx: int) -> None:
        if idx < 0:
            return
        self._render_page(idx)

    def _render_page(self, idx: int) -> None:
        if not self._document or idx >= len(self._document.pages):
            return
        page = self._document.pages[idx]
        if page.image is not None:
            self.preview.set_page(page.image, page.ocr_result)

        if page.ocr_result and page.ocr_result.spans:
            self.text_panel.set_text("\n".join([sp.text for sp in page.ocr_result.spans]))
        else:
            self.text_panel.set_text("(Текст не распознан)")

    def on_embed_clicked(self) -> None:
        if not self._document:
            return
        if any(p.ocr_result is None for p in self._document.pages):
            QMessageBox.warning(self, "ЦВЗ", "Сначала дождитесь завершения OCR.")
            return

        cfg = HybridConfig()

        try:
            embed_document(self._document, cfg)
            self._is_watermarked = True

            # самопроверка (в памяти)
            ext = extract_from_page(self._document, 0, cfg)
            if not ext.ok:
                QMessageBox.critical(self, "ЦВЗ", f"Встроили, но не смогли извлечь сразу.\n{ext.error}")
                self.actions.btn_save.setEnabled(False)
                return

            QMessageBox.information(
                self, "ЦВЗ",
                "ЦВЗ встроен и успешно извлечён сразу после встраивания.\n"
                "Важно: для LSB-слоя сохраняйте результат в PNG/TIFF (lossless)."
            )
            self.actions.btn_save.setEnabled(True)

        except Exception as e:
            self._is_watermarked = False
            self.actions.btn_save.setEnabled(False)
            QMessageBox.critical(self, "Ошибка", f"Не удалось встроить ЦВЗ:\n{type(e).__name__}: {e}")

    def on_save_clicked(self) -> None:
        if not self._current_path or not self._is_watermarked or not self._document:
            return

        base_dir = os.path.dirname(self._current_path)
        base_name = os.path.splitext(os.path.basename(self._current_path))[0]

        # По умолчанию предлагаем lossless:
        # если вход PDF → лучше TIFF, иначе сохраняем в исходном расширении, но фильтр всё равно lossless
        default_ext = ".tiff" if self._document.doc_format == DocumentFormat.PDF else (Path(self._current_path).suffix.lower() or ".png")
        default_name = f"{base_name}_protected{default_ext}"

        options = QFileDialog.Option.DontUseNativeDialog
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить защищённый файл (рекомендуется PNG/TIFF)",
            os.path.join(base_dir, default_name),
            SAVE_FILTER_LOSSLESS,
            options=options,
        )
        if not save_path:
            return

        # если пользователь не указал расширение — добавим по выбранному фильтру
        sp = Path(save_path)
        if sp.suffix == "":
            if selected_filter.startswith("TIFF"):
                save_path = str(sp.with_suffix(".tiff"))
            elif selected_filter.startswith("PNG"):
                save_path = str(sp.with_suffix(".png"))
            elif selected_filter.startswith("JPEG"):
                save_path = str(sp.with_suffix(".jpg"))

        try:
            save_document(self._document, save_path)
            QMessageBox.information(self, "Сохранение", f"Файл сохранён:\n{save_path}\n\nПроверяйте именно этот PNG/TIFF.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", f"{type(e).__name__}: {e}")

    def _set_busy(self, busy: bool) -> None:
        self.actions.btn_load.setEnabled(not busy)
        self.page_selector.setEnabled((not busy) and bool(self._document) and len(self._document.pages) > 1)
        if busy:
            self.actions.btn_action_primary.setEnabled(False)
            self.actions.btn_save.setEnabled(False)

    def _reset_ui(self) -> None:
        self.file_panel.clear()
        self.status.setText("Готово")
        self.text_panel.clear()
        self.page_selector.clear()
        self.page_selector.setEnabled(False)
        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)
