# Общая ошибка OCR-движка
class OCREngineError(Exception):
    pass


# OCR-движок недоступен
class OCREngineNotAvailable(OCREngineError):
    pass