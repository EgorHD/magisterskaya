# src/app/ui/settings.py
from __future__ import annotations
import os
from PyQt6.QtCore import QSettings

DEFAULT_START_DIR = r"C:\Users\egork\OneDrive\Рабочий стол\Магистерская"

class AppSettings:
    def __init__(self) -> None:
        self._s = QSettings("VKR", "DocIntegrityWatermark")

    def last_dir(self) -> str:
        # ✅ всегда начинаем с твоей папки
        return DEFAULT_START_DIR if os.path.isdir(DEFAULT_START_DIR) else os.path.expanduser("~")

    def set_last_dir(self, path: str) -> None:
        # можно оставить пустым (тогда всегда будет только DEFAULT_START_DIR)
        # или сохранять, если хочешь запоминать последнее место:
        # if path: self._s.setValue("last_dir", path)
        return