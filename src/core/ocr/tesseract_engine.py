# src/core/ocr/tesseract_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from core.models.document import Page
from core.models.ocr import OCRPageResult, OCRSpan
from core.ocr.engine_base import OCREngine
from core.ocr.errors import OCREngineError, OCREngineNotAvailable


@dataclass(slots=True)
class TesseractConfig:
    # Языки распознавания
    langs: str = "rus+eng"

    # Режим сегментации страницы
    psm: int = 6

    # Режим OCR-движка
    oem: int = 3

    # Режим предобработки таблиц
    table_mode: bool = False

    # Явный путь к tesseract.exe
    tesseract_cmd: Optional[str] = None


class TesseractOCREngine(OCREngine):
    # Название движка
    name = "TesseractOCR"

    def __init__(self, config: TesseractConfig | None = None) -> None:
        self.config = config or TesseractConfig()

        # Импорт pytesseract
        try:
            import pytesseract  # type: ignore
        except Exception as e:
            raise OCREngineNotAvailable(
                "pytesseract не установлен. Добавьте pytesseract в requirements.txt"
            ) from e

        self._pt = pytesseract

        # Установка пути к tesseract.exe
        if self.config.tesseract_cmd:
            self._pt.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        # Проверка доступности Tesseract
        try:
            _ = self._pt.get_tesseract_version()
        except Exception as e:
            raise OCREngineNotAvailable(
                "Tesseract не найден. Установите Tesseract OCR и добавьте его в PATH "
                "или укажите путь tesseract_cmd в настройках."
            ) from e

    # Распознавание одной страницы
    def recognize_page(self, page: Page) -> OCRPageResult:
        if page.image is None:
            raise OCREngineError("У страницы отсутствует image.")

        pil_img = page.image

        # Перевод изображения в numpy
        if isinstance(pil_img, Image.Image):
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            np_img = np.asarray(pil_img)
        else:
            np_img = np.asarray(pil_img)

        # Предобработка изображения
        proc = _preprocess_for_tesseract(
            np_img,
            table_mode=self.config.table_mode,
        )

        custom = f"--oem {self.config.oem} --psm {self.config.psm}"

        # Запуск Tesseract OCR
        try:
            data = self._pt.image_to_data(
                proc,
                lang=self.config.langs,
                config=custom,
                output_type=self._pt.Output.DICT,
            )
        except Exception as e:
            raise OCREngineError(f"Tesseract OCR не выполнился: {e}") from e

        spans: list[OCRSpan] = []
        n = len(data.get("text", []))

        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue

            # Уверенность распознавания
            conf_str = str(data.get("conf", ["-1"])[i])
            try:
                conf = float(conf_str)
            except Exception:
                conf = -1.0

            # Координаты найденного блока
            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            quad = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
            spans.append(OCRSpan(quad=quad, text=text, confidence=conf))

        return OCRPageResult(page_index=page.index, spans=spans)


# Предобработка изображения перед OCR
def _preprocess_for_tesseract(
    np_img: np.ndarray,
    *,
    table_mode: bool = False,
) -> np.ndarray:
    import cv2

    # Перевод RGB в grayscale
    if np_img.ndim == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img.copy()

    # Подавление шума
    if table_mode:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    else:
        gray = cv2.medianBlur(gray, 3)

    # Предобработка для таблиц
    if table_mode:
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            12,
        )

        # Удаление линий таблицы
        thr = _remove_table_lines(thr)

        # Склеивание разрывов символов
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        return thr

    # Обычная бинаризация
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return thr


# Удаление горизонтальных и вертикальных линий таблицы
def _remove_table_lines(bin_img: np.ndarray) -> np.ndarray:
    import cv2

    # Инверсия бинарного изображения
    inv = 255 - bin_img

    h, w = inv.shape[:2]

    # Размеры ядер относительно размера страницы
    hor_len = max(20, w // 30)
    ver_len = max(20, h // 30)

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_len, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_len))

    # Выделение горизонтальных линий
    horizontal = cv2.erode(inv, hor_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, hor_kernel, iterations=2)

    # Выделение вертикальных линий
    vertical = cv2.erode(inv, ver_kernel, iterations=1)
    vertical = cv2.dilate(vertical, ver_kernel, iterations=2)

    # Объединение найденных линий
    lines = cv2.bitwise_or(horizontal, vertical)

    # Удаление линий из изображения
    inv_wo = cv2.bitwise_and(inv, cv2.bitwise_not(lines))

    # Возврат к обычной бинаризации
    out = 255 - inv_wo
    return out