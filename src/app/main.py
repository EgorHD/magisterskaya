import os
import sys
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# 1) гарантированно отключаем проверку model hosters
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# 2) флаги Paddle (оставим, но тоже жёстко)
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

from PyQt6.QtWidgets import QApplication
from app.ui.main_window import MainWindow

def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())