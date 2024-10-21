"""
The cross-platform app for efficiently performing Bayesian causal inference and supervised learning tasks using tree-based models, including BCF, BART, and XBART.
"""

import sys
import os
import csv
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QTreeView, QTableView, QWidget, QSplitter, QHeaderView
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import QFileSystemModel
from operator import itemgetter  # For sorting the data


class CSVTableModel(QAbstractTableModel):
    def __init__(self, data, headers):
        super().__init__()
        self._data = data
        self._headers = headers
        self._sort_order = Qt.AscendingOrder  # Default sort order

    def rowCount(self, parent=QModelIndex()):
        # Return the number of rows
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        # Return the number of columns
        return len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            # Return the data for display purposes
            return self._data[index.row()][index.column()]
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            # Handle headers
            if orientation == Qt.Horizontal:
                return self._headers[section]  # Horizontal headers
            else:
                return str(section + 1)  # Row numbers
        return None

    def sort(self, column, order):
        """Sort the data by the given column index and order."""
        self.layoutAboutToBeChanged.emit()

        # Sort data using the column index
        try:
            self._data.sort(key=itemgetter(column), reverse=(order == Qt.DescendingOrder))
        except Exception as e:
            print(f"Error sorting data: {e}")

        self.layoutChanged.emit()

        # Save the sort order (ascending or descending)
        self._sort_order = order


class Arborist(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Arborist")

        # Set the default geometry to 1600x900
        self.resize(1600, 900)

        # Allow resizing smaller and larger than the default size
        self.setMinimumSize(800, 600)

        # Center the window on the screen
        self.center_window()

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

        # Adjust file browser column widths
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Minimum width for columns
        self.tree.header().setMinimumSectionSize(100)

        # Double click to open the file
        self.tree.doubleClicked.connect(self.on_file_double_click)

        splitter.addWidget(self.tree)

        # File viewer (right panel)
        self.file_viewer = QTableView(self)
        splitter.addWidget(self.file_viewer)

        # Enable sorting by column headers
        self.file_viewer.setSortingEnabled(True)

        # Set up layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Set the initial splitter sizes
        splitter.setSizes([300, 1300])

    def center_window(self):
        # Get the screen geometry to center the window
        screen = self.screen()  # Get the current screen
        screen_geometry = screen.availableGeometry()  # Get available screen size (excluding taskbars, etc.)
        window_geometry = self.frameGeometry()

        # Calculate the center point of the screen
        center_point = screen_geometry.center()

        # Move the center of the window to the screen's center point
        window_geometry.moveCenter(center_point)

        # Move the window's top-left corner to the calculated center
        self.move(window_geometry.topLeft())

    def on_file_double_click(self, index):
        # Get the file path from the model index
        file_path = self.file_model.filePath(index)

        # Load the file and display its contents
        if file_path.endswith('.csv'):
            self.load_csv_file(file_path)
        else:
            self.file_viewer.setModel(None)  # Clear the table if non-CSV file

    def load_csv_file(self, file_path):
        try:
            with open(file_path, newline='') as file:
                reader = csv.reader(file)
                data = list(reader)

            if len(data) > 0:
                # Separate the first row as headers and the rest as data
                headers = data[0]
                table_data = data[1:]

                # Use the custom model to set the data and headers
                model = CSVTableModel(table_data, headers)
                self.file_viewer.setModel(model)

                # Automatically adjust the column width to fit the content and header
                self.file_viewer.resizeColumnsToContents()

        except Exception as e:
            print(f"Error loading file: {e}")
            self.file_viewer.setModel(None)


def main():
    app = QApplication(sys.argv)
    main_window = Arborist()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
