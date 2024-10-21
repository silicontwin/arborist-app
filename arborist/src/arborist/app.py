"""
The cross-platform app for efficiently performing Bayesian causal inference and supervised learning tasks using tree-based models, including BCF, BART, and XBART.
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTreeView, QTextEdit, QWidget, QSplitter
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QFileSystemModel


class Arborist(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Arborist - File Browser")

        # Create a splitter to divide the file browser and file viewer
        splitter = QSplitter(Qt.Horizontal)

        # File browser (left panel)
        self.file_model = QFileSystemModel()
        
        # Get the user's desktop directory
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

        # Set the root path to the desktop
        self.file_model.setRootPath(desktop_path)
        
        self.tree = QTreeView()
        self.tree.setModel(self.file_model)

        # Set the root index to the desktop directory
        self.tree.setRootIndex(self.file_model.index(desktop_path))

        # Double click to open the file
        self.tree.doubleClicked.connect(self.on_file_double_click)

        splitter.addWidget(self.tree)

        # File viewer (right panel)
        self.file_viewer = QTextEdit(self)
        self.file_viewer.setReadOnly(True)

        splitter.addWidget(self.file_viewer)

        # Set up layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Set the initial splitter sizes
        splitter.setSizes([300, 700])

    def on_file_double_click(self, index):
        # Get the file path from the model index
        file_path = self.file_model.filePath(index)

        # Load the file and display its contents
        self.load_file(file_path)

    def load_file(self, file_path):
        try:
            # Open the file and read its contents
            with open(file_path, 'r') as file:
                content = file.read()

            # Display the content in the file viewer
            self.file_viewer.setText(content)
        except Exception as e:
            self.file_viewer.setText(f"Error loading file: {e}")


def main():
    app = QApplication(sys.argv)
    main_window = Arborist()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
