from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit, QHBoxLayout, QPushButton, QGroupBox
)


class TextPanel(QGroupBox):
    """
    Панель для вывода текста (OCR/восстановленного).
    """
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self)

        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("Здесь будет отображаться распознанный / восстановленный текст документа.")
        layout.addWidget(self.text)

    def set_text(self, value: str) -> None:
        self.text.setPlainText(value)

    def clear(self) -> None:
        self.text.clear()


class FileInfoPanel(QGroupBox):
    """
    Панель для отображения выбранного файла.
    """
    def __init__(self, title: str = "Файл", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self)

        self.path_label = QLabel("Файл не выбран", self)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setWordWrap(True)

        layout.addWidget(self.path_label)

    def set_path(self, path: str) -> None:
        self.path_label.setText(path)

    def clear(self) -> None:
        self.path_label.setText("Файл не выбран")


class ActionsBar(QWidget):
    """
    Унифицированная панель кнопок.
    """
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_load = QPushButton("Загрузить файл")
        self.btn_action_primary = QPushButton("Основное действие")
        self.btn_action_secondary = QPushButton("Вторичное действие")
        self.btn_save = QPushButton("Сохранить")

        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_action_primary)
        layout.addWidget(self.btn_action_secondary)
        layout.addWidget(self.btn_save)
        layout.addStretch(1)

    def set_primary_text(self, text: str) -> None:
        self.btn_action_primary.setText(text)

    def set_secondary_text(self, text: str) -> None:
        self.btn_action_secondary.setText(text)