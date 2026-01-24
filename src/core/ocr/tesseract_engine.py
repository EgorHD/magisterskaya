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
    psm: режим сегментации страницы (6 часто хорош для документов)
    oem: движок распознавания (3 = default)
    langs: строка языков, например "rus+eng"
    """
    langs: str = "rus+eng"
    psm: int = 6
    oem: int = 3
    # Можно указать явный путь к tesseract.exe, если он не в PATH
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

        # Если tesseract.exe не в PATH — можно задать путь
        if self.config.tesseract_cmd:
            self._pt.pytesseract.tesseract_cmd = self.config.tesseract_cmd

        # Быстрая проверка доступности
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

        # Предобработка: серый + адаптивная бинаризация
        proc = _preprocess_for_tesseract(np_img)

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

            # quad как прямоугольник
            quad = (
                (x, y),
                (x + w, y),
                (x + w, y + h),
                (x, y + h),
            )
            spans.append(OCRSpan(quad=quad, text=text, confidence=conf))

        return OCRPageResult(page_index=page.index, spans=spans)


def _preprocess_for_tesseract(np_img: np.ndarray) -> np.ndarray:
    import cv2
    cv2.setNumThreads(1)
    """
    Улучшает распознавание: grayscale + adaptive threshold.
    """
    import cv2  # type: ignore

    if np_img.ndim == 3:
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = np_img

    # лёгкое подавление шума
    gray = cv2.medianBlur(gray, 3)

    # адаптивная бинаризация
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    return thr
