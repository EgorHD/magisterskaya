from __future__ import annotations
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

class TextPanel(QGroupBox):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)

        # Основной layout панели
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Поле для вывода текста
        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setPlaceholderText(
            "Здесь будет отображаться распознанный, восстановленный текст или отчёт"
        )
        self.text.setMinimumHeight(130)

        layout.addWidget(self.text)

    # Установка текста
    def set_text(self, value: str) -> None:
        self.text.setPlainText(value)

    # Очистка текста
    def clear(self) -> None:
        self.text.clear()

class FileInfoPanel(QGroupBox):
    def __init__(self, title: str = "Файл", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)

        # Основной layout панели
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Метка с путём к файлу
        self.path_label = QLabel("Файл не выбран", self)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setWordWrap(True)

        layout.addWidget(self.path_label)

    # Установка пути к файлу
    def set_path(self, path: str) -> None:
        self.path_label.setText(path)

    # Очистка панели
    def clear(self) -> None:
        self.path_label.setText("Файл не выбран")


class ActionsBar(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Горизонтальная панель кнопок
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Кнопки действий
        self.btn_load = QPushButton("Загрузить")
        self.btn_action_primary = QPushButton("Основное действие")
        self.btn_action_secondary = QPushButton("Вторичное действие")
        self.btn_save = QPushButton("Сохранить")

        # Общие настройки кнопок
        buttons = [
            self.btn_load,
            self.btn_action_primary,
            self.btn_action_secondary,
            self.btn_save,
        ]

        for btn in buttons:
            btn.setMinimumHeight(36)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layout.addWidget(btn, 1)

    # Текст основной кнопки
    def set_primary_text(self, text: str) -> None:
        self.btn_action_primary.setText(text)

    # Текст дополнительной кнопки
    def set_secondary_text(self, text: str) -> None:
        self.btn_action_secondary.setText(text)


class StatusProgressBlock(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Основной layout блока статуса
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Текстовый статус
        self.status_label = QLabel("Готово")

        # Индикатор прогресса
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)

    # Установка текста статуса
    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    # Скрытие прогресса и сброс состояния
    def hide_progress(self) -> None:
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

    # Бесконечный индикатор занятости
    def set_busy_indeterminate(self, text: str) -> None:
        self.status_label.setText(text)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

    # Установка прогресса выполнения
    def set_progress(self, text: str, current: int, total: int) -> None:
        self.status_label.setText(text)
        self.progress.setVisible(True)

        if total <= 0:
            self.progress.setRange(0, 0)
            return

        self.progress.setRange(0, total)
        self.progress.setValue(max(0, min(current, total)))