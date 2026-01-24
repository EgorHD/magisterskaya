from __future__ import annotations

from core.ocr.engine_base import OCREngine
from core.ocr.tesseract_engine import TesseractOCREngine, TesseractConfig

_engine: OCREngine | None = None


def get_ocr_engine(*, lang: str = "ru", use_angle_cls: bool = True, use_gpu: bool = False) -> OCREngine:
    """
    Сейчас используем стабильный Tesseract.
    lang игнорируем (в Tesseract задаётся строкой langs).
    """
    global _engine
    if _engine is None:
        # Если у вас tesseract.exe НЕ в PATH — укажите путь тут:
        # cfg = TesseractConfig(langs="rus+eng", psm=6, oem=3, tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        cfg = TesseractConfig(
            langs="rus+eng",
            psm=6,
            oem=3,
            tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )
        _engine = TesseractOCREngine(cfg)
    return _engine
