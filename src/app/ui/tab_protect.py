from __future__ import annotations

import math
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
from core.io.errors import DocumentLoadError
from core.io.loader import load_document
from core.io.saver import save_document
from core.metrics.image_quality import compare_images
from core.models.document import Document, DocumentFormat
from core.models.ocr import OCRResult
from core.watermark.hybrid import HybridConfig, embed_document, extract_from_page


# Фильтр открытия файлов
OPEN_FILTER = "Документы (*.pdf *.tif *.tiff *.jpg *.jpeg *.png);;Все файлы (*.*)"

# Фильтр сохранения результата
SAVE_FILTER_LOSSLESS = "TIFF (*.tif *.tiff);;PNG (*.png);;JPEG (*.jpg *.jpeg)"


class ProtectTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Настройки приложения
        self._settings = AppSettings()

        # Текущий путь к файлу
        self._current_path: str | None = None

        # Загруженный документ
        self._document: Document | None = None

        # Флаг успешного встраивания ЦВЗ
        self._is_watermarked: bool = False

        # Копии исходных изображений страниц
        self._orig_images: list = []

        # Поток OCR
        self._ocr_thread: QThread | None = None
        self._ocr_worker: OCRWorker | None = None

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
        self.file_panel = FileInfoPanel("Входной файл")

        # Панель действий
        self.actions = ActionsBar()
        self.actions.set_primary_text("Встроить ЦВЗ")
        self.actions.set_secondary_text("—")

        # Блок статуса и прогресса
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

        # Параметр LR
        self.sb_lr = QSpinBox()
        self.sb_lr.setRange(1, 10)
        self.sb_lr.setValue(1)
        self.sb_lr.setPrefix("LR=")
        self.sb_lr.setToolTip("Light Repetition")
        row.addWidget(self.sb_lr)

        # Параметр LM
        self.sb_lm = QSpinBox()
        self.sb_lm.setRange(0, 400)
        self.sb_lm.setValue(0)
        self.sb_lm.setPrefix("LM=")
        self.sb_lm.setToolTip("Light Margin")
        row.addWidget(self.sb_lm)

        # Параметр LD
        self.sb_ld = QDoubleSpinBox()
        self.sb_ld.setRange(1.0, 80.0)
        self.sb_ld.setDecimals(1)
        self.sb_ld.setSingleStep(1.0)
        self.sb_ld.setValue(18.0)
        self.sb_ld.setPrefix("LD=")
        self.sb_ld.setToolTip("Light Delta")
        row.addWidget(self.sb_ld)

        # Выбор страницы
        self.page_selector = QComboBox()
        self.page_selector.setEnabled(False)
        self.page_selector.currentIndexChanged.connect(self._on_page_changed)

        # Просмотр страницы
        self.preview = PagePreview()

        # Панель распознанного текста
        self.text_panel = TextPanel("Распознанный текст")
        self.text_panel.setMaximumHeight(220)

        # Сборка левой колонки
        left_layout.addWidget(self.file_panel)
        left_layout.addWidget(self.actions)
        left_layout.addWidget(self.status_block)
        left_layout.addWidget(self.chk_table_mode)
        left_layout.addLayout(row)
        left_layout.addWidget(self.page_selector)
        left_layout.addWidget(self.text_panel)

        # Сборка правой колонки
        right_layout.addWidget(self.preview, stretch=1)

        # Добавление колонок
        root.addLayout(left_layout, 2)
        root.addLayout(right_layout, 3)

        # Начальное состояние параметров ЦВЗ
        self._set_light_controls_enabled(self.chk_wm_capacity.isChecked())
        self.chk_wm_capacity.toggled.connect(self._set_light_controls_enabled)

        # Кнопки недоступны до загрузки файла
        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_action_secondary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)

        # Вторичная кнопка не используется
        self.actions.btn_action_secondary.hide()

        # Подключение кнопок
        self.actions.btn_load.clicked.connect(self.on_load_clicked)
        self.actions.btn_action_primary.clicked.connect(self.on_embed_clicked)
        self.actions.btn_save.clicked.connect(self.on_save_clicked)

    # Остановка OCR-потока
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

    # Очистка состояния после предыдущего документа
    def _clear_loaded_document_state(self) -> None:
        self._stop_ocr_thread()

        self._current_path = None
        self._document = None
        self._is_watermarked = False
        self._orig_images = []

        # Очистка списка страниц
        self.page_selector.blockSignals(True)
        self.page_selector.clear()
        self.page_selector.setEnabled(False)
        self.page_selector.blockSignals(False)

        # Очистка текста и статуса
        self.text_panel.clear()
        self.status_block.set_status("Готово")
        self.status_block.hide_progress()

        # Очистка предпросмотра
        try:
            self.preview.clear()
        except Exception:
            pass

        # Блокировка действий
        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)

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

        # Сброс предыдущего состояния
        self._clear_loaded_document_state()
        self._settings.set_last_dir(os.path.dirname(path))
        self._current_path = path

        # DPI для PDF зависит от режима OCR
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

        # Сохранение документа
        self._document = doc
        self.file_panel.set_path(path)

        # Копирование исходных изображений страниц
        self._orig_images = []
        try:
            for p in self._document.pages:
                self._orig_images.append(p.image.copy() if p.image is not None else None)
        except Exception:
            self._orig_images = []

        # Показ первой страницы
        if self._document.pages and self._document.pages[0].image is not None:
            try:
                self.preview.set_page(self._document.pages[0].image, None)
            except Exception:
                pass

        # Подготовка к OCR
        self.text_panel.set_text("Запуск OCR...")
        self._set_busy(True)
        self.status_block.set_progress("OCR: подготовка...", 0, 0)

        # Запуск OCR
        self._start_ocr(self._document)

    # Включение и выключение параметров ЦВЗ
    def _set_light_controls_enabled(self, enabled: bool) -> None:
        self.sb_lr.setEnabled(enabled)
        self.sb_lm.setEnabled(enabled)
        self.sb_ld.setEnabled(enabled)

    # Запуск OCR в отдельном потоке
    def _start_ocr(self, doc: Document) -> None:
        if self._ocr_thread is not None and self._ocr_thread.isRunning():
            return

        # Создание потока и worker
        self._ocr_thread = QThread(self)
        self._ocr_worker = OCRWorker(
            doc,
            lang="ru",
            table_mode=self.chk_table_mode.isChecked(),
        )
        self._ocr_worker.moveToThread(self._ocr_thread)

        # Подключение сигналов OCR
        self._ocr_thread.started.connect(self._ocr_worker.run)
        self._ocr_worker.progress.connect(self._on_ocr_progress)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.failed.connect(self._on_ocr_failed)

        # Завершение и очистка потока
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

    # Обработка успешного завершения OCR
    def _on_ocr_finished(self, result_obj: object) -> None:
        result: OCRResult = result_obj  # type: ignore

        self.status_block.set_status("OCR: готово")
        self.status_block.hide_progress()

        if self._document:
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
        else:
            self.text_panel.set_text(result.text() or "(Текст не распознан)")

        # Разблокировка действий после OCR
        self.actions.btn_action_primary.setEnabled(True)
        self.actions.btn_save.setEnabled(False)
        self._set_busy(False)

    # Обработка ошибки OCR
    def _on_ocr_failed(self, message: str) -> None:
        self.status_block.set_status("OCR: ошибка")
        self.status_block.hide_progress()
        self.text_panel.set_text("OCR не выполнен\n\n" f"Причина: {message}")
        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)
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

        # Показ изображения и OCR-рамок
        if page.image is not None:
            self.preview.set_page(page.image, page.ocr_result)

        # Показ распознанного текста
        if page.ocr_result and page.ocr_result.spans:
            self.text_panel.set_text("\n".join(sp.text for sp in page.ocr_result.spans))
        else:
            self.text_panel.set_text("(Текст не распознан)")

    # Запуск встраивания ЦВЗ
    def on_embed_clicked(self) -> None:
        if not self._document:
            return

        if any(p.ocr_result is None for p in self._document.pages):
            QMessageBox.warning(self, "ЦВЗ", "Сначала дождитесь завершения OCR")
            return

        cfg = HybridConfig()

        self.status_block.set_busy_indeterminate("Встраивание ЦВЗ...")
        QApplication.processEvents()

        t0 = time.perf_counter()
        try:
            # Создание конфигурации ЦВЗ
            cfg = HybridConfig()
            cfg.capacity_mode = self.chk_wm_capacity.isChecked()

            # Повторное создание конфигурации оставлено без изменений
            cfg = HybridConfig()
            cfg.capacity_mode = self.chk_wm_capacity.isChecked()

            # Настройка параметров лёгкого режима
            if cfg.capacity_mode:
                cfg.light_repetition = int(self.sb_lr.value())
                cfg.light_margin_px = int(self.sb_lm.value())
                cfg.light_delta = float(self.sb_ld.value())

            # Встраивание ЦВЗ
            embed_document(self._document, cfg)
            embed_time = time.perf_counter() - t0
            self._is_watermarked = True

            # Проверка извлекаемости ЦВЗ
            ok_any = False
            errs = []

            for i in range(len(self._document.pages)):
                ext = extract_from_page(self._document, i, cfg)
                if ext.ok:
                    ok_any = True
                    break
                errs.append(f"стр.{i + 1}: {ext.error}")

            if not ok_any:
                self.status_block.set_status("Ошибка встраивания ЦВЗ")
                self.status_block.hide_progress()
                QMessageBox.critical(
                    self,
                    "ЦВЗ",
                    "Встроили, но не смогли извлечь ни с одной страницы\n" + "\n".join(errs),
                )
                self.actions.btn_save.setEnabled(False)
                return

            # Расчёт метрик качества
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

            # Сообщение о результате
            msg = (
                "ЦВЗ успешно встроен\n"
                "Важно: сохраняйте результат в PNG/TIFF\n\n"
            )

            if qual_list:
                mse = sum(q.mse for q in qual_list) / len(qual_list)

                if mse <= 0:
                    psnr = float("inf")
                else:
                    psnr = 10.0 * math.log10((255.0 * 255.0) / mse)

                ssim = sum(q.ssim for q in qual_list) / len(qual_list)

                msg += (
                    "Незаметность (среднее по страницам):\n"
                    f"MSE: {mse:.6f}\n"
                    f"PSNR: {psnr:.3f}\n"
                    f"SSIM: {ssim:.6f}\n\n"
                )

            msg += f"Время встраивания: {embed_time:.3f} сек"

            # Обновляем текущий показ страницы после встраивания
            if self.page_selector.currentIndex() >= 0:
                self._render_page(self.page_selector.currentIndex())
            else:
                self._render_page(0)

            self.status_block.set_status("ЦВЗ встроен")
            self.status_block.hide_progress()
            QMessageBox.information(self, "ЦВЗ", msg)
            self.actions.btn_save.setEnabled(True)

        except Exception as e:
            self._is_watermarked = False
            self.actions.btn_save.setEnabled(False)
            self.status_block.set_status("Ошибка встраивания ЦВЗ")
            self.status_block.hide_progress()
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось встроить ЦВЗ:\n{type(e).__name__}: {e}",
            )

    # Сохранение защищённого файла
    def on_save_clicked(self) -> None:
        if not self._current_path or not self._is_watermarked or not self._document:
            return

        # Базовое имя файла
        base_name = os.path.splitext(os.path.basename(self._current_path))[0]

        # Расширение по умолчанию
        default_ext = (
            ".tiff"
            if self._document.doc_format == DocumentFormat.PDF
            else (Path(self._current_path).suffix.lower() or ".png")
        )

        default_name = f"{base_name}_protected{default_ext}"

        options = QFileDialog.Option.DontUseNativeDialog
        start_dir = self._settings.last_dir()

        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить защищённый файл",
            os.path.join(start_dir, default_name),
            SAVE_FILTER_LOSSLESS,
            options=options,
        )
        if not save_path:
            return

        self._settings.set_last_dir(os.path.dirname(save_path))

        # Подстановка расширения, если оно не указано
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
            QMessageBox.information(
                self,
                "Сохранение",
                f"Файл сохранён:\n{save_path}\n\nПроверяйте именно этот PNG/TIFF",
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", f"{type(e).__name__}: {e}")

    # Переключение интерфейса в занятый режим
    def _set_busy(self, busy: bool) -> None:
        self.actions.btn_load.setEnabled(not busy)
        self.page_selector.setEnabled(
            (not busy) and bool(self._document) and len(self._document.pages) > 1
        )
        self.chk_table_mode.setEnabled(not busy)

        if busy:
            self.actions.btn_action_primary.setEnabled(False)
            self.actions.btn_save.setEnabled(False)

    # Сброс элементов интерфейса
    def _reset_ui(self) -> None:
        self.file_panel.clear()
        self.status_block.set_status("Готово")
        self.status_block.hide_progress()
        self.text_panel.clear()
        self.page_selector.clear()
        self.page_selector.setEnabled(False)

        try:
            self.preview.clear()
        except Exception:
            pass

        self.actions.btn_action_primary.setEnabled(False)
        self.actions.btn_save.setEnabled(False)