"""
The cross-platform app for efficiently performing Bayesian causal inference and supervised learning tasks using tree-based models, including BCF, BART, and XBART.
"""

import sys
import os
import pandas as pd
from PySide6.QtWidgets import (QApplication, QMainWindow, QTreeView, QTableView, QFileSystemModel, QLabel, QPushButton, QTabWidget, QWidget, QSplitter, QHeaderView, QComboBox)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PySide6.QtGui import QColor
from arborist.layouts.browse import Ui_BrowseTab
from arborist.layouts.analyze import Ui_AnalyzeTab

CHUNK_SIZE = 10000  # Number of rows to load per chunk

# Lazy loading for handling large datasets in chunks with sorting enabled
class PandasTableModel(QAbstractTableModel):
    def __init__(self, chunk_iter, headers):
        super().__init__()
        self.chunk_iter = chunk_iter
        self.headers = headers
        self._data = pd.DataFrame()  # Store all loaded data
        self._sort_order = Qt.AscendingOrder  # Default sort order
        self._sort_column = None  # No sort initially
        self.selected_column = None  # Outcome variable column to highlight
        self.load_next_chunk()  # Load the first chunk

    def rowCount(self, parent=QModelIndex()):
        # Return the number of rows in the current data
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        # Return the number of columns in the dataset
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            # Return data from the loaded dataset
            return str(self._data.iloc[index.row(), index.column()])

        # Highlight the selected outcome variable column
        if role == Qt.BackgroundRole and self.selected_column is not None:
            if index.column() == self.selected_column:
                return QColor("#FFFFCB")

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                # Return the column header
                return self.headers[section]
            else:
                # Return the row number
                return str(section + 1)
        return None

    def load_next_chunk(self):
        """Load the next chunk of data and append it to the model."""
        try:
            chunk = next(self.chunk_iter)
            old_row_count = self.rowCount()
            self.beginInsertRows(QModelIndex(), old_row_count, old_row_count + len(chunk) - 1)
            self._data = pd.concat([self._data, chunk], ignore_index=True)
            self.endInsertRows()

            # Apply sorting to new chunk if already sorted
            if self._sort_column is not None:
                self.sort(self._sort_column, self._sort_order)

        except StopIteration:
            print("No more chunks available.")

    def can_fetch_more(self):
        """Check if there are more chunks to fetch."""
        return True  # Always try to fetch more until iteration ends

    def fetchMore(self, parent=QModelIndex()):
        """Fetch the next chunk of data."""
        self.load_next_chunk()

    def sort(self, column, order):
        """Sort the data by the given column index and order."""
        self.layoutAboutToBeChanged.emit()

        # Sort loaded data
        self._data.sort_values(by=self._data.columns[column], ascending=(order == Qt.AscendingOrder), inplace=True)
        self._data.reset_index(drop=True, inplace=True)

        self._sort_order = order
        self._sort_column = column

        self.layoutChanged.emit()

    def set_highlighted_column(self, column_index):
            """Set the column to highlight in yellow."""
            self.selected_column = column_index
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
        self.browse_ui.setupUi(self.browse_tab)

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

        # Outcome variable dropdown
        self.outcome_combo = self.analyze_ui.comboBox

        # Initially, show only the "No dataset" label, not the dataset viewer
        self.no_dataset_label.setVisible(True)
        self.analytics_viewer.setVisible(False)

        # Initially, show only the "No dataset" label, not the dataset viewer
        self.outcome_combo.currentIndexChanged.connect(self.highlight_selected_column)

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
        """Load the selected CSV file and display its contents in chunks."""
        try:
            chunk_iter = pd.read_csv(file_path, chunksize=CHUNK_SIZE)  # Load in chunks
            headers = next(chunk_iter).columns.tolist()  # Extract headers from the first chunk
            model = PandasTableModel(chunk_iter, headers)
            table_view.setModel(model)

            # Enable sorting
            table_view.setSortingEnabled(True)

            # Automatically adjust the column width to fit the content and header
            table_view.resizeColumnsToContents()

            self.outcome_combo.clear()  # Clear the outcome variable dropdown
            self.outcome_combo.addItems(headers)  # Add the column names to the dropdown

            # Connect the scroll event for lazy loading
            table_view.verticalScrollBar().valueChanged.connect(lambda value: self.on_scroll(value, table_view))

        except Exception as e:
            print(f"Error loading file: {e}")
            table_view.setModel(None)

    def on_scroll(self, value, table_view):
        """Handle scrolling to load more data."""
        # When scrolled to the bottom, fetch more data
        if value == table_view.verticalScrollBar().maximum():
            model = table_view.model()
            if model and model.can_fetch_more():  # If more chunks are available
                model.fetchMore()

    def highlight_selected_column(self):
        """Highlight the selected column in the analytics viewer."""
        selected_index = self.outcome_combo.currentIndex()
        model = self.analytics_viewer.model()
        if model:
            model.set_highlighted_column(selected_index)

    def open_in_analytics_view(self):
        """Open the dataset in the analytics view (second tab)."""
        model = self.file_viewer.model()
        if model:
            # Show the dataset in the analytics tab
            self.analytics_viewer.setModel(model)
            self.analytics_viewer.resizeColumnsToContents()

            # Connect scroll event for lazy loading in the analytics tab
            self.analytics_viewer.verticalScrollBar().valueChanged.connect(lambda value: self.on_scroll(value, self.analytics_viewer))

            # Clear the outcome variable dropdown and add the column names
            self.outcome_combo.clear()
            self.outcome_combo.addItems(model.headers)

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
