from __future__ import annotations
import os

# Стартовая папка для выбора файлов
DEFAULT_START_DIR = r"C:\Users\egork\OneDrive\Рабочий стол\Магистерская"

class AppSettings:
    def __init__(self) -> None:
        pass

    # Получение стартовой директории
    def last_dir(self) -> str:
        if os.path.isdir(DEFAULT_START_DIR):
            return DEFAULT_START_DIR
        return os.path.expanduser("~")

    # Сохранение директории сейчас не используется
    def set_last_dir(self, path: str) -> None:
        return