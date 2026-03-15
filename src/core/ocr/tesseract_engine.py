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
    """
    Настройки Tesseract.
    psm: режим сегментации страницы
    oem: движок распознавания (3 = default)
    langs: строка языков, например "rus+eng"
    table_mode: спец.предобработка для таблиц (удаление линий и т.п.)
    """
    langs: str = "rus+eng"
    psm: int = 6
    oem: int = 3
    table_mode: bool = False
    tesseract_cmd: Optional[str] = None


class TesseractOCREngine(OCREngine):
    name = "TesseractOCR"

    def __init__(self, config: TesseractConfig | None = None) -> None:
        self.config = config or TesseractConfig()

        try:
            import pytesseract  # type: ignore
        except Exception as e:
            raise OCREngineNotAvailable("pytesseract не установлен. Добавьте pytesseract в requirements.txt") from e

        self._pt = pytesseract

        if self.config.tesseract_cmd:
            self._pt.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        try:
            _ = self._pt.get_tesseract_version()
        except Exception as e:
            raise OCREngineNotAvailable(
                "Tesseract не найден. Установите Tesseract OCR и добавьте его в PATH "
                "или укажите путь tesseract_cmd в настройках."
            ) from e

    def recognize_page(self, page: Page) -> OCRPageResult:
        if page.image is None:
            raise OCREngineError("У страницы отсутствует image.")

        pil_img = page.image
        if isinstance(pil_img, Image.Image):
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            np_img = np.asarray(pil_img)
        else:
            np_img = np.asarray(pil_img)

        proc = _preprocess_for_tesseract(np_img, table_mode=self.config.table_mode)

        custom = f"--oem {self.config.oem} --psm {self.config.psm}"
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

            conf_str = str(data.get("conf", ["-1"])[i])
            try:
                conf = float(conf_str)
            except Exception:
                conf = -1.0

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            quad = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
            spans.append(OCRSpan(quad=quad, text=text, confidence=conf))

        return OCRPageResult(page_index=page.index, spans=spans)


def _preprocess_for_tesseract(np_img: np.ndarray, *, table_mode: bool = False) -> np.ndarray:
    """
    Предобработка под Tesseract.
    Обычный режим: серый -> небольшой blur -> adaptive threshold.
    Табличный режим: усиление + удаление линий таблицы + close для символов.
    """
    import cv2  # локально, чтобы не ломать окружение без opencv

    # RGB -> Gray
    if np_img.ndim == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img.copy()

    # Лёгкое подавление шума (таблицы часто страдают от "зерна")
    if table_mode:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    else:
        gray = cv2.medianBlur(gray, 3)

    # Adaptive threshold (параметры для таблиц делаем мягче)
    # Для таблиц иногда лучше INV, но мы держим обычный вариант и удаляем линии.
    if table_mode:
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,   # block size (чуть больше)
            12,   # C
        )
        thr = _remove_table_lines(thr)

        # Чуть "склеить" разорванные цифры/буквы после вычитания линий
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        return thr

    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return thr


def _remove_table_lines(bin_img: np.ndarray) -> np.ndarray:
    """
    Удаление горизонтальных и вертикальных линий таблицы из бинарного изображения.
    На входе bin_img: белый фон (255), чёрный текст/линии (0) — как после THRESH_BINARY.
    """
    import cv2

    # Работаем с инверсией, чтобы линии были "белыми" объектами на чёрном фоне
    inv = 255 - bin_img

    h, w = inv.shape[:2]
    # Длины ядер подбираются относительно размера страницы
    hor_len = max(20, w // 30)
    ver_len = max(20, h // 30)

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_len, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_len))

    # Выделяем линии (эрозия -> дилатация)
    horizontal = cv2.erode(inv, hor_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, hor_kernel, iterations=2)

    vertical = cv2.erode(inv, ver_kernel, iterations=1)
    vertical = cv2.dilate(vertical, ver_kernel, iterations=2)

    lines = cv2.bitwise_or(horizontal, vertical)

    # Вычитаем линии из исходной инверсии
    inv_wo = cv2.bitwise_and(inv, cv2.bitwise_not(lines))

    # Возвращаем обратно в "обычную" бинаризацию (белый фон, чёрный текст)
    out = 255 - inv_wo
    return out