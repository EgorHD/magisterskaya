from PyQt6.QtWidgets import QMainWindow, QTabWidget
from app.ui.tab_protect import ProtectTab
from app.ui.tab_verify import VerifyTab

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        # Параметры главного окна
        self.setWindowTitle("Система защиты целостности ЭОД")
        self.resize(1080, 760)
        self.setMinimumSize(980, 680)

        # Основной контейнер вкладок
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Вкладка защиты документа
        self.protect_tab = ProtectTab(parent=self)

        # Вкладка проверки документа
        self.verify_tab = VerifyTab(parent=self)

        # Добавление вкладок в интерфейс
        self.tabs.addTab(self.protect_tab, "Защита ЭОД")
        self.tabs.addTab(self.verify_tab, "Проверка ЭОД")

        # Первая активная вкладка
        self.tabs.setCurrentIndex(0)

        # Общий стиль приложения
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f6f7fb;
            }

            QTabWidget::pane {
                border: 1px solid #d7dbe7;
                background: #ffffff;
                top: -1px;
            }

            QTabBar::tab {
                background: #edf1f7;
                border: 1px solid #d7dbe7;
                padding: 8px 14px;
                margin-right: 4px;
                min-width: 120px;
            }

            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom-color: #ffffff;
            }

            QGroupBox {
                border: 1px solid #d7dbe7;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 8px;
                background: #ffffff;
                font-weight: 600;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }

            QLabel {
                color: #1f2937;
            }

            QPushButton {
                background: #ffffff;
                border: 1px solid #cfd6e4;
                border-radius: 10px;
                padding: 6px 12px;
            }

            QPushButton:hover {
                background: #f3f6fb;
            }

            QPushButton:pressed {
                background: #e7edf8;
            }

            QPushButton:disabled {
                color: #9aa4b2;
                background: #f4f5f7;
            }

            QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                border: 1px solid #cfd6e4;
                border-radius: 8px;
                background: #ffffff;
                padding: 4px 6px;
            }

            QProgressBar {
                border: 1px solid #cfd6e4;
                border-radius: 8px;
                text-align: center;
                background: #ffffff;
                min-height: 16px;
            }

            QProgressBar::chunk {
                background: #6d8cff;
                border-radius: 7px;
            }
            """
        )