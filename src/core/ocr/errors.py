class OCREngineError(Exception):
    """Общая ошибка OCR-движка."""


class OCREngineNotAvailable(OCREngineError):
    """OCR-движок недоступен (нет зависимостей/моделей и т.п.)."""