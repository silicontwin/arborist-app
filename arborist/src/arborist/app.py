"""
The cross-platform app for efficiently performing Bayesian causal inference and supervised learning tasks using tree-based models, including BCF, BART, and XBART.
"""

import sys
import os
import pandas as pd
import numpy as np
import itertools
import pyarrow.dataset as ds
import pyarrow as pa
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTreeView,
    QTableView,
    QFileSystemModel,
    QLabel,
    QPushButton,
    QTabWidget,
    QWidget,
    QSplitter,
    QHeaderView,
    QComboBox,
)
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PySide6.QtGui import QColor
from arborist.layouts.browse import Ui_BrowseTab
from arborist.layouts.analyze import Ui_AnalyzeTab
from stochtree import BCFModel, BARTModel

CHUNK_SIZE = 10000  # Number of rows to load per chunk

# Lazy loading for handling large datasets in chunks with sorting enabled
class PandasTableModel(QAbstractTableModel):
    def __init__(self, data, headers):
        super().__init__()
        self.headers = headers
        self._sort_order = Qt.AscendingOrder  # Default sort order
        self._sort_column = None  # No sort initially
        self.selected_column_name = None  # Outcome variable column to highlight

        if isinstance(data, pd.DataFrame):
            self._data = data
            self.has_more_chunks = False
        else:
            self.chunk_iter = data
            self._data = pd.DataFrame()  # Store all loaded data
            self.has_more_chunks = True
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
            value = self._data.iloc[index.row(), index.column()]
            if pd.isnull(value):
                return ''
            return str(value)

        # Highlight the selected outcome variable column and prediction columns
        if role == Qt.BackgroundRole:
            column_name = self.headers[index.column()]
            if self.selected_column_name == column_name:
                return QColor("#FFFFCB")  # Light yellow
            elif column_name in ['Posterior Average (y hat)', '2.5th percentile', '97.5th percentile']:
                return QColor("#CCCCFF")  # Light blue

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
        if not self.has_more_chunks:
            return
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
            self.has_more_chunks = False  # No more chunks to load

    def can_fetch_more(self):
        """Check if there are more chunks to fetch."""
        return self.has_more_chunks

    def fetchMore(self, parent=QModelIndex()):
        """Fetch the next chunk of data."""
        if self.has_more_chunks:
            self.load_next_chunk()

    def sort(self, column, order):
        """Sort the data by the given column index and order."""
        self.layoutAboutToBeChanged.emit()

        # Sort loaded data
        self._data.sort_values(
            by=self._data.columns[column],
            ascending=(order == Qt.AscendingOrder),
            inplace=True,
        )
        self._data.reset_index(drop=True, inplace=True)

        self._sort_order = order
        self._sort_column = column

        self.layoutChanged.emit()

    def set_highlighted_column(self, column_name):
        """Set the column to highlight in yellow."""
        self.selected_column_name = column_name
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
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        self.file_model.setRootPath(desktop_path)

        # Create a proxy model to filter by dataset extensions (.csv, .sav, .dta)
        dataset_extensions = {".csv", ".sav", ".dta"}
        self.proxy_model = DatasetFileFilterProxyModel(dataset_extensions)
        self.proxy_model.setSourceModel(self.file_model)

        # Set up the tree view to use the filtered model
        self.tree = self.browse_ui.treeView
        self.tree.setModel(self.proxy_model)

        # Set the root index to the desktop directory
        self.current_directory = desktop_path  # Store the current directory path
        source_index = self.file_model.index(desktop_path)
        self.current_root_index = self.proxy_model.mapFromSource(source_index)
        self.tree.setRootIndex(self.current_root_index)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)  # Auto-adjust column width
        self.tree.header().setMinimumSectionSize(100)

        # Double click to open the file or enter directory
        self.tree.doubleClicked.connect(self.on_file_double_click)

        # Set up the file viewer
        self.file_viewer = self.browse_ui.file_viewer
        self.file_viewer.setSortingEnabled(True)

        # Set splitter sizes to 600 for the file browser, 1000 for the file viewer
        self.browse_ui.splitter.setSizes([600, 1000])

        # "Analyze Dataset" button setup
        self.open_button = self.browse_ui.analyze_button
        self.open_button.setVisible(False)  # Initially hidden until a dataset is selected
        self.open_button.clicked.connect(self.open_in_analytics_view)

        # Back and Forward buttons
        self.back_button = self.browse_ui.back_button
        self.forward_button = self.browse_ui.forward_button
        self.back_button.clicked.connect(self.navigate_back)
        self.forward_button.clicked.connect(self.navigate_forward)

        # Keep track of navigation history using directory paths
        self.history = [self.current_directory]
        self.history_index = 0

        # Initialize navigation buttons
        self.back_button.setEnabled(False)
        self.forward_button.setEnabled(False)

    def load_analyze_tab_ui(self):
        """Load and set up the analyze tab UI (for the dataset analysis)."""
        self.analyze_tab = QWidget()
        self.analyze_ui = Ui_AnalyzeTab()
        self.analyze_ui.setupUi(self.analyze_tab)

        # No dataset label and analytics viewer
        self.no_dataset_label = self.analyze_ui.no_dataset_label
        self.analytics_viewer = self.analyze_ui.analytics_viewer

        # Outcome variable dropdown
        self.outcome_combo = self.analyze_ui.outcomeComboBox

        # Treatment variable dropdown
        self.treatment_combo = self.analyze_ui.treatmentComboBox

        # Train Model button
        self.train_button = self.analyze_ui.trainButton
        self.train_button.clicked.connect(self.train_model)

        # Initially, show only the "No dataset" label, not the dataset viewer
        self.no_dataset_label.setVisible(True)
        self.analytics_viewer.setVisible(False)

        # Outcome and treatment variable selection
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
        """Handle double-click events on files and directories in the tree view."""
        source_index = self.proxy_model.mapToSource(index)
        file_path = self.file_model.filePath(source_index)

        if self.file_model.isDir(source_index):
            # It's a directory, navigate into it by setting the root index
            self.navigate_to_directory(file_path)
        elif file_path.endswith((".csv", ".sav", ".dta")):
            # Open the file
            self.load_csv_file(file_path, self.file_viewer)
            self.open_button.setVisible(True)  # Show the 'Open' button after a dataset is loaded
        else:
            self.file_viewer.setModel(None)  # Clear the table if non-dataset file
            self.open_button.setVisible(False)  # Hide the 'Open' button if no dataset is loaded

    def navigate_to_directory(self, directory_path):
        """Navigate to the directory represented by the given path."""
        # Update navigation history
        if self.history_index < len(self.history) - 1:
            # If we have moved back in history and then navigate to a new directory
            self.history = self.history[:self.history_index + 1]
        self.history.append(directory_path)
        self.history_index += 1

        # Update current directory
        self.current_directory = directory_path

        # Get the index for the directory
        source_index = self.file_model.index(directory_path)
        proxy_index = self.proxy_model.mapFromSource(source_index)

        # Set the root index to the new directory
        self.tree.setRootIndex(proxy_index)
        self.current_root_index = proxy_index

        # Update navigation buttons
        self.back_button.setEnabled(self.history_index > 0)
        self.forward_button.setEnabled(self.history_index < len(self.history) - 1)

    def navigate_back(self):
        """Navigate back in the directory history."""
        if self.history_index > 0:
            self.history_index -= 1
            directory_path = self.history[self.history_index]
            self.current_directory = directory_path

            # Get the index for the directory
            source_index = self.file_model.index(directory_path)
            proxy_index = self.proxy_model.mapFromSource(source_index)

            # Set the root index
            self.tree.setRootIndex(proxy_index)
            self.current_root_index = proxy_index

            # Update navigation buttons
            self.back_button.setEnabled(self.history_index > 0)
            self.forward_button.setEnabled(self.history_index < len(self.history) - 1)

    def navigate_forward(self):
        """Navigate forward in the directory history."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            directory_path = self.history[self.history_index]
            self.current_directory = directory_path

            # Get the index for the directory
            source_index = self.file_model.index(directory_path)
            proxy_index = self.proxy_model.mapFromSource(source_index)

            # Set the root index
            self.tree.setRootIndex(proxy_index)
            self.current_root_index = proxy_index

            # Update navigation buttons
            self.back_button.setEnabled(self.history_index > 0)
            self.forward_button.setEnabled(self.history_index < len(self.history) - 1)

    def load_csv_file(self, file_path, table_view):
        """Load the selected CSV file and display its contents in chunks."""
        try:
            self.current_file_path = file_path  # Store the current file path
            chunk_iter = pd.read_csv(file_path, chunksize=CHUNK_SIZE)  # Load in chunks
            first_chunk = next(chunk_iter)
            headers = first_chunk.columns.tolist()  # Extract headers from the first chunk
            # Re-create chunk_iter including the first chunk
            chunk_iter = itertools.chain([first_chunk], chunk_iter)
            model = PandasTableModel(chunk_iter, headers)
            table_view.setModel(model)

            # Enable sorting
            table_view.setSortingEnabled(True)

            # Automatically adjust the column width to fit the content and header
            table_view.resizeColumnsToContents()

            self.outcome_combo.clear()  # Clear the outcome variable dropdown
            self.outcome_combo.addItems(headers)  # Add the column names to the dropdown

            # Connect the scroll event for lazy loading
            table_view.verticalScrollBar().valueChanged.connect(
                lambda value: self.on_scroll(value, table_view)
            )

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
        selected_var = self.outcome_combo.currentText()
        model = self.analytics_viewer.model()
        if model and selected_var in model.headers:
            model.set_highlighted_column(selected_var)

    def open_in_analytics_view(self):
        """Open the dataset in the analytics view (second tab)."""
        if hasattr(self, 'current_file_path'):
            # Reload the dataset with initial chunk for the analytics view
            try:
                chunk_iter = pd.read_csv(self.current_file_path, chunksize=CHUNK_SIZE)
                first_chunk = next(chunk_iter)
                headers = first_chunk.columns.tolist()
                # Re-create chunk_iter including the first chunk
                chunk_iter = itertools.chain([first_chunk], chunk_iter)
                model = PandasTableModel(chunk_iter, headers)
                self.analytics_viewer.setModel(model)
                self.analytics_viewer.resizeColumnsToContents()

                # Enable sorting
                self.analytics_viewer.setSortingEnabled(True)

                # Connect scroll event for lazy loading in the analytics tab
                self.analytics_viewer.verticalScrollBar().valueChanged.connect(
                    lambda value: self.on_scroll(value, self.analytics_viewer)
                )

                # Clear the outcome variable dropdown and add the column names
                self.outcome_combo.clear()
                self.outcome_combo.addItems(headers)
                self.treatment_combo.clear()
                self.treatment_combo.addItems(headers)

                self.no_dataset_label.setVisible(False)
                self.analytics_viewer.setVisible(True)
            except Exception as e:
                print(f"Error loading file in analytics view: {e}")
                self.analytics_viewer.setModel(None)
                self.no_dataset_label.setVisible(True)
                self.analytics_viewer.setVisible(False)
        else:
            # Show the "No dataset" message if no dataset is loaded
            self.no_dataset_label.setVisible(True)
            self.analytics_viewer.setVisible(False)

        # Switch to the analytics tab
        self.tabs.setCurrentIndex(1)

    def train_model(self):
        """Train the model using the stochtree library and update the dataset."""
        if not hasattr(self, 'current_file_path'):
            print("No dataset loaded.")
            return

        outcome_var = self.outcome_combo.currentText()

        if not outcome_var:
            print("Please select an outcome variable.")
            return

        try:
            # Read the first row to get headers
            first_chunk = pd.read_csv(self.current_file_path, nrows=0)
            all_columns = first_chunk.columns.tolist()
            
            # Define the columns to read (all columns to maintain original order)
            columns_to_read = all_columns

            # Use pyarrow to read all columns
            dataset = ds.dataset(self.current_file_path, format='csv')

            # Read the columns into a pyarrow table
            table = dataset.to_table(columns=columns_to_read)

            # Convert to pandas DataFrame
            df = table.to_pandas()

            # Proceed with data cleaning and model training
            df_cleaned = df.dropna()
            initial_row_count = len(df)
            observations_removed = initial_row_count - len(df_cleaned)

            # Ensure that outcome variable is in the data
            if outcome_var not in df_cleaned.columns:
                print(f"Outcome variable '{outcome_var}' not found in the data.")
                return

            # Use all columns except the outcome variable as features
            feature_columns = [col for col in df_cleaned.columns if col != outcome_var]

            # Select features and outcome
            X = df_cleaned[feature_columns].to_numpy()
            y = df_cleaned[outcome_var].to_numpy()

            # Ensure that y is numeric
            if not np.issubdtype(y.dtype, np.number):
                print(f"Outcome variable '{outcome_var}' is not numeric.")
                return

            # Standardize the outcome variable
            y_mean = np.mean(y)
            y_std = np.std(y)
            y_standardized = (y - y_mean) / y_std

            # Train the BART model
            bart_model = BARTModel()

            # Sample from the posterior
            bart_model.sample(
                X_train=X,
                y_train=y_standardized,
                X_test=X,
                num_gfr=0,
                num_burnin=100,
                num_mcmc=100
            )

            # Get predictions
            y_pred_samples = bart_model.predict(covariates=X)
            # y_pred_samples = y_pred_samples * y_std + y_mean  # Convert back to original scale

            # Compute posterior summaries over the samples (axis=1)
            posterior_mean = np.mean(y_pred_samples, axis=1)
            percentile_2_5 = np.percentile(y_pred_samples, 2.5, axis=1)
            percentile_97_5 = np.percentile(y_pred_samples, 97.5, axis=1)

            # Add predictions to the DataFrame
            df_cleaned['Posterior Average (y hat)'] = posterior_mean
            df_cleaned['2.5th percentile'] = percentile_2_5
            df_cleaned['97.5th percentile'] = percentile_97_5

            # Reorder the columns to show predictions first
            prediction_cols = ['Posterior Average (y hat)', '2.5th percentile', '97.5th percentile']
            existing_cols = [col for col in df_cleaned.columns if col not in prediction_cols]
            df_cleaned = df_cleaned[prediction_cols + existing_cols]

            # Update the model and the view
            headers = df_cleaned.columns.tolist()
            self.dataframe = df_cleaned  # Update the dataframe with predictions
            model = PandasTableModel(self.dataframe, headers)
            self.analytics_viewer.setModel(model)
            self.analytics_viewer.resizeColumnsToContents()

            # Re-highlight the selected outcome variable column
            self.highlight_selected_column()

            # Print the number of observations removed
            print(f"Number of observations removed due to missing data: {observations_removed}")

        except Exception as e:
            print(f"Error during model training: {e}")


# Main function to start the application
def main():
    app = QApplication(sys.argv)
    main_window = Arborist()
    main_window.show()
    sys.exit(app.exec())


# Entry point of the script
if __name__ == "__main__":
    main()
