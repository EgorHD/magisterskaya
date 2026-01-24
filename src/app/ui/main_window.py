from PyQt6.QtWidgets import QMainWindow, QTabWidget

from app.ui.tab_protect import ProtectTab
from app.ui.tab_verify import VerifyTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Система защиты целостности ЭОД (prototype)")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.protect_tab = ProtectTab(parent=self)
        self.verify_tab = VerifyTab(parent=self)

        self.tabs.addTab(self.protect_tab, "Защита ЭОД")
        self.tabs.addTab(self.verify_tab, "Проверка целостности ЭОД")

        # По ТЗ: изначально открыта вкладка защиты
        self.tabs.setCurrentIndex(0)