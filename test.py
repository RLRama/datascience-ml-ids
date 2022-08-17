from tkinter import Button
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QLineEdit, QVBoxLayout, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
import sys

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("icon.png"))
        self.setWindowTitle("Data Science - TheRAMBros")

        layout = QGridLayout()
        self.setLayout(layout)

        label1 = QLabel("Nombre de usuario: ", parent=self)
        layout.addWidget(label1, 0 ,0)

        label2 = QLabel("Contraseña: ", parent=self)
        layout.addWidget(label2, 1 ,0)

        input1 = QLineEdit()
        layout.addWidget(input1, 0, 1)

        input2 = QLineEdit()
        layout.addWidget(input2, 1, 1)

        button = QPushButton("Iniciar")
        button.setFixedWidth(60)
        layout.addWidget(button, 2, 1, Qt.AlignmentFlag.AlignCenter)

app = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec())