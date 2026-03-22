# src/app/ui/tab_protect.py
from __future__ import annotations
import os
import time
from pathlib import Path
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFileDialog, QMessageBox, QLabel, QComboBox, QCheckBox
)
from app.ui.widgets import ActionsBar, FileInfoPanel, TextPanel
from app.ui.ocr_worker import OCRWorker
from app.ui.page_preview import PagePreview
from app.ui.settings import AppSettings
from PyQt6.QtWidgets import QHBoxLayout, QSpinBox, QDoubleSpinBox
from core.io.loader import load_document
from core.io.errors import DocumentLoadError
from core.io.saver import save_document
from core.models.document import Document, DocumentFormat
from core.models.ocr import OCRResult
from core.watermark.hybrid import HybridConfig, embed_document, extract_from_page
from core.metrics.image_quality import compare_images
OPEN_FILTER = "Документы (*.pdf *.tif *.tiff *.jpg *.jpeg *.png);;Все файлы (*.*)"
SAVE_FILTER_LOSSLESS = "TIFF (*.tif *.tiff);;PNG (*.png);;JPEG (*.jpg *.jpeg)"

class ProtectTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._settings = AppSettings()

        self._current_path: str | None = None
        self._document: Document | None = None
        self._is_watermarked: bool = False

        self._orig_images: list = []

        self._ocr_thread: QThread | None = None
        self._ocr_worker: OCRWorker | None = None

        layout = QVBoxLayout(self)

        self.file_panel = FileInfoPanel("Входной файл (ЭОД)")
        self.actions = ActionsBar()
        self.actions.set_primary_text("Встроить ЦВЗ")
        self.actions.set_secondary_text("—")

        self.status = QLabel("Готово")

        # ✅ режим для таблиц
        self.chk_table_mode = QCheckBox("Табличный режим OCR (лучше для счетов/таблиц)")
        self.chk_table_mode.setChecked(False)

        # --- Лёгкий режим ЦВЗ ---
        self.chk_wm_capacity = QCheckBox("Настраиваемый режим ЦВЗ")
        self.chk_wm_capacity.setChecked(True)


        row = QHBoxLayout()
        row.addWidget(self.chk_wm_capacity)

        self.sb_lr = QSpinBox()
        self.sb_lr.setRange(1, 10)
        self.sb_lr.setValue(1)  # LR
        self.sb_lr.setPrefix("LR=")
        self.sb_lr.setToolTip("Light Repetition (повторы бит): больше = надёжнее, меньше ёмкость")
        row.addWidget(self.sb_lr)

        self.sb_lm = QSpinBox()
        self.sb_lm.setRange(0, 400)  # можно больше, но обычно 0..120 хватает
        self.sb_lm.setValue(0)  # LM
        self.sb_lm.setPrefix("LM=")
        self.sb_lm.setToolTip("Light Margin (поля, px): больше = стабильнее, меньше ёмкость")
        row.addWidget(self.sb_lm)

        self.sb_ld = QDoubleSpinBox()
        self.sb_ld.setRange(1.0, 80.0)
        self.sb_ld.setDecimals(1)
        self.sb_ld.setSingleStep(1.0)
        self.sb_ld.setValue(18.0)  # LD
        self.sb_ld.setPrefix("LD=")
        self.sb_ld.setToolTip("Light Delta (сила): больше = устойчивее, но может сильнее портить изображение")
        row.addWidget(self.sb_ld)

        layout.addLayout(row)

        # по умолчанию выключено (активируем только при включении лёгкого режима)
        self._set_light_controls_enabled(self.chk_wm_capacity.isChecked())
        self.chk_wm_capacity.toggled.connect(self._set_light_controls_enabled)

        self.page_selector = QComboBox()
        self.page_selector.setEnabled(False)
        self.page_selector.currentIndexChanged.connect(self._on_page_changed)

        self.preview = PagePreview()
        self.text_panel = TextPanel("Распознанный текст (OCR)")

        layout.addWidget(self.file_panel)
        layout.addWidget(self.actions)
        layout.addWidget(self.status)
        layout.addWidget(self.chk_table_mode)
        layout.addWidget(self.page_selector)
        layout.addWidget(self.preview, stretch=2)
        layout.addWidget(self.text_panel, stretch=1)

        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_action_secondary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)

        self.actions.btn_load.clicked.connect(self.on_load_clicked)
        self.actions.btn_action_primary.clicked.connect(self.on_embed_clicked)
        self.actions.btn_save.clicked.connect(self.on_save_clicked)

    def _stop_ocr_thread(self) -> None:
        if self._ocr_thread is not None:
            try:
                if self._ocr_thread.isRunning():
                    self._ocr_thread.quit()
                    self._ocr_thread.wait(2000)
            except Exception:
                pass
        self._ocr_thread = None
        self._ocr_worker = None

    def _clear_loaded_document_state(self) -> None:
        self._stop_ocr_thread()

        self._current_path = None
        self._document = None
        self._is_watermarked = False
        self._orig_images = []

        self.page_selector.blockSignals(True)
        self.page_selector.clear()
        self.page_selector.setEnabled(False)
        self.page_selector.blockSignals(False)

        self.text_panel.clear()
        self.status.setText("Готово")

        try:
            self.preview.clear()
        except Exception:
            # если у preview нет clear(), просто ничего не делаем
            pass

        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)

    def on_load_clicked(self) -> None:
        options = QFileDialog.Option.DontUseNativeDialog
        start_dir = self._settings.last_dir()

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл ЭОД",
            start_dir,
            OPEN_FILTER,
            options=options
        )
        if not path:
            return

        # СНАЧАЛА полностью очищаем старое состояние
        self._clear_loaded_document_state()

        self._settings.set_last_dir(os.path.dirname(path))
        self._current_path = path

        pdf_dpi = 400 if self.chk_table_mode.isChecked() else 300

        try:
            doc = load_document(Path(path), pdf_dpi=pdf_dpi)
        except DocumentLoadError as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))
            self._reset_ui()
            return
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", f"{type(e).__name__}: {e}")
            self._reset_ui()
            return

        self._document = doc
        self.file_panel.set_path(path)

        # сохраняем оригинальные изображения
        self._orig_images = []
        try:
            for p in self._document.pages:
                self._orig_images.append(p.image.copy() if p.image is not None else None)
        except Exception:
            self._orig_images = []

        # показываем первую страницу, но без OCR-рамок
        if self._document.pages and self._document.pages[0].image is not None:
            try:
                self.preview.set_page(self._document.pages[0].image, None)
            except Exception:
                pass

        self.text_panel.set_text("Запуск OCR...")
        self._set_busy(True)
        self._start_ocr(self._document)

    def _set_light_controls_enabled(self, enabled: bool) -> None:
        self.sb_lr.setEnabled(enabled)
        self.sb_lm.setEnabled(enabled)
        self.sb_ld.setEnabled(enabled)

    def _start_ocr(self, doc: Document) -> None:
        if self._ocr_thread is not None and self._ocr_thread.isRunning():
            return

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

    # ----- дальше оставь как у тебя (без изменений), я не трогаю embed/save/preview -----
    # ВАЖНО: ниже вставь свой исходный код методов:
    # _on_ocr_progress, _on_ocr_finished, _on_ocr_failed, on_embed_clicked, on_save_clicked, ...
    # а также _set_busy, _reset_ui, _on_page_changed и т.д.

    def _on_page_changed(self, idx: int) -> None:
        # твоя логика
        if not self._document or idx < 0 or idx >= len(self._document.pages):
            return
        p = self._document.pages[idx]
        if p.image is None:
            return
        spans = p.ocr_result.spans if p.ocr_result else None
        self.preview.set_page(p.image, spans)

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

    def _on_page_changed(self, idx: int) -> None:
        if idx < 0:
            return
        self._render_page(idx)

    def _render_page(self, idx: int) -> None:
        if not self._document or idx >= len(self._document.pages):
            return
        page = self._document.pages[idx]

        if page.image is not None:
            # ✅ ВАЖНО: передаём OCRPageResult, как было у тебя
            self.preview.set_page(page.image, page.ocr_result)

        if page.ocr_result and page.ocr_result.spans:
            self.text_panel.set_text("\n".join([sp.text for sp in page.ocr_result.spans]))
        else:
            self.text_panel.set_text("(Текст не распознан)")

    def _on_ocr_failed(self, message: str) -> None:
        self.status.setText("OCR: ошибка")
        self.text_panel.set_text("OCR не выполнен.\n\n" f"Причина: {message}")
        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)
        self._set_busy(False)

    def on_embed_clicked(self) -> None:
        if not self._document:
            return
        if any(p.ocr_result is None for p in self._document.pages):
            QMessageBox.warning(self, "ЦВЗ", "Сначала дождитесь завершения OCR.")
            return

        cfg = HybridConfig()

        t0 = time.perf_counter()
        try:
            cfg = HybridConfig()
            cfg.capacity_mode = self.chk_wm_capacity.isChecked()
            cfg = HybridConfig()
            cfg.capacity_mode = self.chk_wm_capacity.isChecked()

            if cfg.capacity_mode:
                cfg.light_repetition = int(self.sb_lr.value())
                cfg.light_margin_px = int(self.sb_lm.value())
                cfg.light_delta = float(self.sb_ld.value())
            embed_document(self._document, cfg)
            embed_time = time.perf_counter() - t0
            self._is_watermarked = True

            # самопроверка (в памяти)
            ok_any = False
            errs = []
            for i in range(len(self._document.pages)):
                ext = extract_from_page(self._document, i, cfg)
                if ext.ok:
                    ok_any = True
                    break
                errs.append(f"стр.{i + 1}: {ext.error}")
            if not ok_any:
                QMessageBox.critical(self, "ЦВЗ",
                                     "Встроили, но не смогли извлечь ни с одной страницы.\n" + "\n".join(errs))
                self.actions.btn_save.setEnabled(False)
                return

            # ✅ метрики незаметности (если есть оригиналы)
            qual_list = []
            if self._orig_images and len(self._orig_images) == len(self._document.pages):
                for i, p in enumerate(self._document.pages):
                    if p.image is None:
                        continue
                    orig = self._orig_images[i] if i < len(self._orig_images) else None
                    if orig is None:
                        continue
                    try:
                        qual_list.append(compare_images(orig, p.image))
                    except Exception:
                        pass

            msg = (
                "ЦВЗ встроен и успешно извлечён сразу после встраивания.\n"
                "Важно: для LSB-слоя сохраняйте результат в PNG/TIFF (lossless).\n\n"
            )

            if qual_list:
                mse = sum(q.mse for q in qual_list) / len(qual_list)
                psnr = sum(q.psnr for q in qual_list) / len(qual_list)
                ssim = sum(q.ssim for q in qual_list) / len(qual_list)
                msg += (
                    "Незаметность (среднее по страницам):\n"
                    f"MSE: {mse:.6f}\n"
                    f"PSNR: {psnr:.3f}\n"
                    f"SSIM: {ssim:.6f}\n\n"
                )

            msg += f"Время встраивания: {embed_time:.3f} сек."

            QMessageBox.information(self, "ЦВЗ", msg)
            self.actions.btn_save.setEnabled(True)

        except Exception as e:
            self._is_watermarked = False
            self.actions.btn_save.setEnabled(False)
            QMessageBox.critical(self, "Ошибка", f"Не удалось встроить ЦВЗ:\n{type(e).__name__}: {e}")

    def on_save_clicked(self) -> None:
        if not self._current_path or not self._is_watermarked or not self._document:
            return

        base_name = os.path.splitext(os.path.basename(self._current_path))[0]
        default_ext = ".tiff" if self._document.doc_format == DocumentFormat.PDF else (Path(self._current_path).suffix.lower() or ".png")
        default_name = f"{base_name}_protected{default_ext}"

        options = QFileDialog.Option.DontUseNativeDialog

        start_dir = self._settings.last_dir()
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить защищённый файл (рекомендуется PNG/TIFF)",
            os.path.join(start_dir, default_name),
            SAVE_FILTER_LOSSLESS,
            options=options,
        )
        if not save_path:
            return

        self._settings.set_last_dir(os.path.dirname(save_path))

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
        self.chk_table_mode.setEnabled(not busy)
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