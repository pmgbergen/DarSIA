from PySide6.QtWidgets import (
    QMainWindow,
    QApplication,
    QLabel,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QSlider,
    QProgressBar,
    QComboBox,
    QListWidget,
    QRadioButton,
)
from PySide6.QtCore import Qt
import toml


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Darsia")

        container = QWidget()
        self.setCentralWidget(container)

        layout = QVBoxLayout(container)

        label = QLabel("One")
        label.setAlignment(Qt.AlignCenter)

        layout.addWidget(label)

        button = QPushButton("Click me")
        line_edit = QLineEdit()
        text_edit = QTextEdit()
        combo = QComboBox()
        combo.addItems(["One", "Two", "Three"])
        liste = QListWidget()
        liste.addItems(["One", "Two", "Three"])

        layout.addWidget(button)
        layout.addWidget(text_edit)
        layout.addWidget(line_edit)
        layout.addWidget(combo)
        layout.addWidget(liste)


if __name__ == "__main__":

    app = QApplication()
    window = MainWindow()
    window.show()
    app.exec()
