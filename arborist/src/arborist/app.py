"""
The cross-platform app for efficiently performing Bayesian causal inference and supervised learning tasks using tree-based models, including BCF, BART, and XBART.
"""

import importlib.metadata
import sys

from PySide6 import QtWidgets
from arborist.src.layout import Ui_MainWindow

class Arborist(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("arborist")
        self.show()

def main():
    app_module = sys.modules["__main__"].__package__
    metadata = importlib.metadata.metadata(app_module)

    QtWidgets.QApplication.setApplicationName(metadata["Formal-Name"])

    app = QtWidgets.QApplication(sys.argv)
    main_window = Arborist()
    sys.exit(app.exec())
