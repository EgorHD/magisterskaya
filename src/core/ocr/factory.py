from __future__ import annotations

from core.ocr.engine_base import OCREngine
from core.ocr.tesseract_engine import TesseractOCREngine, TesseractConfig

_engine: OCREngine | None = None


def get_ocr_engine(
    *,
    lang: str = "ru",
    use_angle_cls: bool = True,
    use_gpu: bool = False,
    table_mode: bool = False,
) -> OCREngine:
    """
    Сейчас используем стабильный Tesseract.
    table_mode включает режим для таблиц (PSM=11 + спец.предобработка).
    """
    global _engine

    # Создаём новый engine при первом вызове.
    # Если mode меняется на лету — проще создать новый, чтобы конфиг не "залипал".
    if _engine is None or getattr(_engine, "_table_mode", None) != table_mode:
        cfg = TesseractConfig(
            langs="rus+eng",
            psm=11 if table_mode else 6,
            oem=3,
            table_mode=table_mode,
            tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        )
        eng = TesseractOCREngine(cfg)
        # маленькая пометка для кэша
        setattr(eng, "_table_mode", table_mode)
        _engine = eng

    return _engine