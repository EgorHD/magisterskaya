from __future__ import annotations
import os
import sys

# Источник загрузки моделей PaddleX
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"

# Ограничение числа потоков CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Отключение проверки источников моделей
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Флаги Paddle
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

from PyQt6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


# Точка входа в приложение
def main() -> int:
    # Создание Qt-приложения
    app = QApplication(sys.argv)

    # Создание главного окна
    window = MainWindow()
    window.show()

    # Запуск цикла обработки событий
    return app.exec()


# Запуск приложения
if __name__ == "__main__":
    raise SystemExit(main())