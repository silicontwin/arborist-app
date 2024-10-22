"""
The cross-platform app for efficiently performing Bayesian causal inference and supervised learning tasks using tree-based models, including BCF, BART, and XBART.
"""

import sys
import os
import csv
from operator import itemgetter  # For sorting the data
from PySide6.QtWidgets import (QApplication, QMainWindow, QTreeView, QTableView, QFileSystemModel, QLabel, QPushButton, QTabWidget, QWidget, QSplitter, QHeaderView)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from arborist.layouts.browse import Ui_BrowseTab
from arborist.layouts.analyze import Ui_AnalyzeTab

# CSVTableModel for handling the CSV data and sorting
class CSVTableModel(QAbstractTableModel):
    def __init__(self, data, headers):
        super().__init__()
        self._data = data
        self._headers = headers

    def rowCount(self, parent=QModelIndex()):
        # Return the number of rows in the data
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        # Return the number of columns in the headers
        return len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            # Return the data at the current index
            return self._data[index.row()][index.column()]
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                # Return the column header
                return self._headers[section]
            else:
                # Return the row number
                return str(section + 1)
        return None

    def sort(self, column, order):
        """Sort the data by the given column index and order."""
        self.layoutAboutToBeChanged.emit()

        # Sort data using the column index
        self._data.sort(key=itemgetter(column), reverse=(order == Qt.DescendingOrder))
        self.layoutChanged.emit()

# FilterProxyModel to filter out non-dataset files like .csv, .sav, .dta
class DatasetFileFilterProxyModel(QSortFilterProxyModel):
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

        # Initialize UI components
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Arborist")
        
        # Set the default window size
        self.resize(1600, 900)

        # Allow resizing the window between 800x600 and full screen
        self.setMinimumSize(800, 600)

        # Center the window on the screen
        self.center_window()

        # Create a tab widget for switching between the file browser and dataset viewer
        self.tabs = QTabWidget()

        # Load the UI for both tabs (Browse and Analyze)
        self.load_browse_tab_ui()
        self.load_analyze_tab_ui()

        # Add the tabs to the main layout
        self.tabs.addTab(self.browse_tab, "Browse")
        self.tabs.addTab(self.analyze_tab, "Analyze")

        # Set the central widget for the main window
        self.setCentralWidget(self.tabs)

    def load_browse_tab_ui(self):
        """Load and set up the browse tab UI (for the file browser)."""
        # Initialize the generated UI class for the Browse tab
        self.browse_tab = QWidget()
        self.browse_ui = Ui_BrowseTab()
        self.browse_ui.setupUi(self.browse_tab)  # Setup the UI on the QWidget

        # Set up the file system model and tree view with dataset filtering
        self.file_model = QFileSystemModel()
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        self.file_model.setRootPath(desktop_path)

        # Create a proxy model to filter by dataset extensions (.csv, .sav, .dta)
        dataset_extensions = {'.csv', '.sav', '.dta'}
        self.proxy_model = DatasetFileFilterProxyModel(dataset_extensions)
        self.proxy_model.setSourceModel(self.file_model)

        # Set up the tree view to use the filtered model
        self.tree = self.browse_ui.treeView
        self.tree.setModel(self.proxy_model)

        # Set the root index to the desktop directory
        self.tree.setRootIndex(self.proxy_model.mapFromSource(self.file_model.index(desktop_path)))
        # self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)  # Auto-adjust column width
        self.tree.header().setMinimumSectionSize(100)

        # Double click to open the file
        self.tree.doubleClicked.connect(self.on_file_double_click)

        # Set up the file viewer
        self.file_viewer = self.browse_ui.file_viewer
        self.file_viewer.setSortingEnabled(True)

        # Set splitter sizes to 300 for the file browser, 1300 for the file viewer
        self.browse_ui.splitter.setSizes([300, 1300])

        # "Analyze Dataset" button setup
        self.open_button = self.browse_ui.analyze_button
        self.open_button.setVisible(False)  # Initially hidden until a dataset is selected
        self.open_button.clicked.connect(self.open_in_analytics_view)

    def load_analyze_tab_ui(self):
        """Load and set up the analyze tab UI (for the dataset analysis)."""
        self.analyze_tab = QWidget()
        self.analyze_ui = Ui_AnalyzeTab()
        self.analyze_ui.setupUi(self.analyze_tab)

        # No dataset label and analytics viewer
        self.no_dataset_label = self.analyze_ui.no_dataset_label
        self.analytics_viewer = self.analyze_ui.analytics_viewer

        # Initially, show only the "No dataset" label, not the dataset viewer
        self.no_dataset_label.setVisible(True)
        self.analytics_viewer.setVisible(False)

    def center_window(self):
        """Center the window on the screen."""
        screen = self.screen()  # Get the current screen
        screen_geometry = screen.availableGeometry()
        window_geometry = self.frameGeometry()

        # Calculate the center point of the screen
        center_point = screen_geometry.center()

        # Move the center of the window to the screen's center point
        window_geometry.moveCenter(center_point)

        # Move the window's top-left corner to the calculated center
        self.move(window_geometry.topLeft())

    def on_file_double_click(self, index):
        """Handle double-click events on files in the tree view."""
        source_index = self.proxy_model.mapToSource(index)
        file_path = self.file_model.filePath(source_index)

        # Only allow files with dataset extensions to be opened
        if file_path.endswith(('.csv', '.sav', '.dta')):
            self.load_csv_file(file_path, self.file_viewer)
            self.open_button.setVisible(True)  # Show the 'Open' button after a dataset is loaded
        else:
            self.file_viewer.setModel(None)  # Clear the table if non-CSV file
            self.open_button.setVisible(False)  # Hide the 'Open' button if no dataset is loaded

    def load_csv_file(self, file_path, table_view):
        """Load the selected CSV file and display its contents."""
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
        """Open the dataset in the analytics view (second tab)."""
        model = self.file_viewer.model()
        if model:
            # Show the dataset in the analytics tab
            self.analytics_viewer.setModel(model)
            self.analytics_viewer.resizeColumnsToContents()
            self.no_dataset_label.setVisible(False)
            self.analytics_viewer.setVisible(True)
        else:
            # Show the "No dataset" message if no dataset is loaded
            self.no_dataset_label.setVisible(True)
            self.analytics_viewer.setVisible(False)

        # Switch to the analytics tab
        self.tabs.setCurrentIndex(1)

# Main function to start the application
def main():
    app = QApplication(sys.argv)
    main_window = Arborist()
    main_window.show()
    sys.exit(app.exec())

# Entry point of the script
if __name__ == "__main__":
    main()
