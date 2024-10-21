"""
The cross-platform app for efficiently performing Bayesian causal inference and supervised learning tasks using tree-based models, including BCF, BART, and XBART.
"""

import sys
import os
import csv
from operator import itemgetter  # For sorting the data
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QTreeView, QTableView, QWidget, QSplitter, 
                               QHeaderView, QTabWidget, QPushButton, QHBoxLayout, QFileSystemModel, QLabel)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel


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
        self._data.sort(key=itemgetter(column), reverse=(order == Qt.DescendingOrder))

        self.layoutChanged.emit()

        # Save the sort order (ascending or descending)
        self._sort_order = order


class DatasetFileFilterProxyModel(QSortFilterProxyModel):
    """Filter proxy model to filter files by dataset extensions."""
    def __init__(self, dataset_extensions, parent=None):
        super().__init__(parent)
        self.dataset_extensions = dataset_extensions

    def filterAcceptsRow(self, sourceRow, sourceParent):
        index = self.sourceModel().index(sourceRow, 0, sourceParent)
        file_name = self.sourceModel().fileName(index)

        if self.sourceModel().isDir(index):
            # Always accept directories
            return True

        # Accept files that have one of the dataset extensions
        _, ext = os.path.splitext(file_name)
        return ext.lower() in self.dataset_extensions


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

        # Create a tab widget for switching between the file browser and dataset viewer
        self.tabs = QTabWidget()

        # Create the file browser tab
        file_browser_tab = QWidget()
        file_browser_layout = QVBoxLayout()
        self.splitter = QSplitter(Qt.Horizontal)

        # File browser (left panel) with dataset filtering
        self.file_model = QFileSystemModel()

        # Get the user's desktop directory
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')

        # Set the root path to the desktop
        self.file_model.setRootPath(desktop_path)

        # Create a proxy model to filter by dataset extensions
        self.proxy_model = DatasetFileFilterProxyModel({'.csv', '.sav', '.dta'})
        self.proxy_model.setSourceModel(self.file_model)

        self.tree = QTreeView()
        self.tree.setModel(self.proxy_model)

        # Set the root index to the desktop directory
        self.tree.setRootIndex(self.proxy_model.mapFromSource(self.file_model.index(desktop_path)))

        # Adjust file browser column widths
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        # Minimum width for columns
        self.tree.header().setMinimumSectionSize(100)

        # Double click to open the file
        self.tree.doubleClicked.connect(self.on_file_double_click)

        self.splitter.addWidget(self.tree)

        # File viewer (right panel)
        self.file_viewer = QTableView(self)
        self.splitter.addWidget(self.file_viewer)

        # Enable sorting by column headers
        self.file_viewer.setSortingEnabled(True)

        # Split for the file browser and viewer
        self.splitter.setSizes([300, 1300])

        # Add a button to open the dataset in the second tab (analytics view)
        self.open_button = QPushButton("Analyze Dataset")
        self.open_button.setVisible(False)  # Initially hide the button
        self.open_button.clicked.connect(self.open_in_analytics_view)

        file_browser_layout.addWidget(self.splitter)
        file_browser_layout.addWidget(self.open_button)

        file_browser_tab.setLayout(file_browser_layout)

        # Add the tabs to the main layout
        self.tabs.addTab(file_browser_tab, "Browse")

        # Create the analytics view tab
        self.create_analytics_view_tab()

        # Set up the layout for the main window
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tabs)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def create_analytics_view_tab(self):
        # Create the analytics view tab
        self.analytics_tab = QWidget()
        self.analytics_tab_layout = QVBoxLayout()

        # Set margin and spacing to 0
        self.analytics_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.analytics_tab_layout.setSpacing(0)

        # No dataset message
        self.no_dataset_label = QLabel('Please select a dataset from the "Browse" tab')
        self.no_dataset_label.setAlignment(Qt.AlignCenter)

        self.analytics_viewer = QTableView()

        # Enable sorting for analytics view
        self.analytics_viewer.setSortingEnabled(True)

        # Add both the label and the viewer to the layout (only one will be visible at a time)
        self.analytics_tab_layout.addWidget(self.no_dataset_label)
        self.analytics_tab_layout.addWidget(self.analytics_viewer)

        # Add a placeholder toolbar at the top (minimized height)
        self.toolbar = QWidget()
        self.toolbar_layout = QHBoxLayout()
        self.toolbar.setLayout(self.toolbar_layout)
        self.toolbar.setFixedHeight(0)  # Minimize the height of the toolbar

        self.analytics_tab_layout.addWidget(self.toolbar)

        self.analytics_tab.setLayout(self.analytics_tab_layout)
        self.tabs.addTab(self.analytics_tab, "Analyze")

        # Initially, only show the message label, not the dataset viewer
        self.no_dataset_label.setVisible(True)
        self.analytics_viewer.setVisible(False)

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
        source_index = self.proxy_model.mapToSource(index)
        file_path = self.file_model.filePath(source_index)

        # Load the file and display its contents in the file viewer
        if file_path.endswith(('.csv', '.sav', '.dta')):
            self.load_csv_file(file_path, self.file_viewer)
            self.open_button.setVisible(True)  # Show the 'Open' button after a dataset is loaded
        else:
            self.file_viewer.setModel(None)  # Clear the table if non-CSV file
            self.open_button.setVisible(False)  # Hide the 'Open' button if no dataset is loaded

    def load_csv_file(self, file_path, table_view):
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
                table_view.setModel(model)

                # Automatically adjust the column width to fit the content and header
                table_view.resizeColumnsToContents()

        except Exception as e:
            print(f"Error loading file: {e}")
            table_view.setModel(None)

    def open_in_analytics_view(self):
        # Load the current file from the file_viewer into the analytics viewer
        model = self.file_viewer.model()

        if model:
            # Show the dataset and hide the message
            self.analytics_viewer.setModel(model)
            self.analytics_viewer.resizeColumnsToContents()
            self.no_dataset_label.setVisible(False)
            self.analytics_viewer.setVisible(True)
        else:
            # Show the message and hide the dataset viewer
            self.no_dataset_label.setVisible(True)
            self.analytics_viewer.setVisible(False)

        # Switch to the analytics tab
        self.tabs.setCurrentIndex(1)


def main():
    app = QApplication(sys.argv)
    main_window = Arborist()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
