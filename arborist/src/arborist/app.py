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
from sklearn.preprocessing import OneHotEncoder
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
    QProgressDialog,
    QMessageBox,
)
from PySide6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor
from arborist.layouts.browse import Ui_BrowseTab
from arborist.layouts.train import Ui_TrainTab
from arborist.layouts.predict import Ui_PredictTab
from stochtree import BCFModel, BARTModel
import time

CHUNK_SIZE = 1000  # Number of rows to load per chunk


# Lazy loading for handling large datasets in chunks with sorting enabled
class PandasTableModel(QAbstractTableModel):
    def __init__(self, data, headers, predictions=None):
        super().__init__()
        self.headers = headers
        self._sort_order = Qt.AscendingOrder  # Default sort order
        self._sort_column = None  # No sort initially
        self.selected_column_name = None  # Outcome variable column to highlight
        self.predictions = predictions

        # Zebra stripe colors
        self.alternate_row_color = QColor("#F5F5F5")  # Light gray for alternate rows
        self.base_row_color = QColor("#FFFFFF")  # White for base rows

        if isinstance(data, pd.DataFrame):
            self._data = data
            self.has_more_chunks = False
        else:
            self.chunk_iter = data
            self._data = pd.DataFrame()  # Store all loaded data
            self.has_more_chunks = True
            self.chunks_loaded = 0  # Track number of chunks loaded
            self.load_next_chunk()  # Load the first chunk

    def rowCount(self, parent=QModelIndex()):
        # Return the number of rows in the current data
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        # Return the number of columns in the dataset
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            # Return data from the loaded dataset
            value = self._data.iloc[index.row(), index.column()]
            if pd.isnull(value):
                return ""
            return str(value)

        elif role == Qt.BackgroundRole:
            # Apply zebra striping
            base_color = (
                self.alternate_row_color if index.row() % 2 else self.base_row_color
            )

            # Then check for special column highlighting
            column_name = self.headers[index.column()]
            if self.selected_column_name == column_name:
                return QColor("#FFFFCB")  # Light yellow for selected column
            elif column_name in [
                "Posterior Average ŷ",
                "2.5th percentile ŷ",
                "97.5th percentile ŷ",
            ]:
                return QColor("#CCCCFF")  # Light blue for prediction columns
            else:
                return base_color

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
            self.beginInsertRows(
                QModelIndex(), old_row_count, old_row_count + len(chunk) - 1
            )

            if self.predictions is not None:
                start_idx = self.chunks_loaded * CHUNK_SIZE
                end_idx = start_idx + len(chunk)

                # Create a new DataFrame with predictions and original data
                prediction_data = {
                    "Posterior Average ŷ": self.predictions["Posterior Mean"][
                        start_idx:end_idx
                    ],
                    "2.5th percentile ŷ": self.predictions["2.5th Percentile"][
                        start_idx:end_idx
                    ],
                    "97.5th percentile ŷ": self.predictions["97.5th Percentile"][
                        start_idx:end_idx
                    ],
                }

                # Combine predictions with chunk data
                combined_chunk = pd.concat(
                    [pd.DataFrame(prediction_data), chunk.reset_index(drop=True)],
                    axis=1,
                )

                self._data = pd.concat([self._data, combined_chunk], ignore_index=True)
            else:
                self._data = pd.concat([self._data, chunk], ignore_index=True)

            self.chunks_loaded += 1
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


class ModelTrainingWorker(QThread):
    """Worker thread for model training to prevent UI freezing."""

    progress = Signal(int)
    finished = Signal(dict, float)
    error = Signal(str)

    def __init__(self, trainer, model_params):
        super().__init__()
        self.trainer = trainer
        self.model_params = model_params
        self._is_running = True

    def run(self):
        """Main method that runs in the separate thread."""
        try:
            start_time = time.time()

            self.progress.emit(10)
            print("Loading data...")
            self.trainer.load_data()

            self.progress.emit(30)
            print("Preparing features...")
            self.trainer.prepare_features()

            self.progress.emit(40)
            print("Training model...")
            self.trainer.train_model(**self.model_params)

            self.progress.emit(80)
            print("Generating predictions...")
            predictions = self.trainer.predict()

            self.progress.emit(100)
            training_time = time.time() - start_time
            self.finished.emit(predictions, training_time)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """Stop the training process."""
        self._is_running = False


# Shared class for model training and preprocessing
class ModelTrainer:
    """ModelTrainer class with error handling and progress tracking."""

    def __init__(self, file_path: str, outcome_var: str, treatment_var: str = None):
        self.file_path = file_path
        self.outcome_var = outcome_var
        self.treatment_var = treatment_var
        self.data = None
        self.data_cleaned = None
        self.X = None
        self.y = None
        self.model = None

    def load_data(self):
        """Load the dataset and preprocess categorical variables."""
        # Check file size before loading
        file_size = os.path.getsize(self.file_path) / (1024 * 1024)
        if file_size > 1000:
            warning = QMessageBox()
            warning.setIcon(QMessageBox.Warning)
            warning.setText(f"Large file detected ({file_size:.1f} MB)")
            warning.setInformativeText("This may consume significant memory. Continue?")
            warning.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            if warning.exec() == QMessageBox.No:
                raise Exception("Operation cancelled by user")

        # Load the dataset in chunks to handle large files
        chunks = []
        for chunk in pd.read_csv(self.file_path, chunksize=CHUNK_SIZE):
            chunks.append(chunk)
        self.data = pd.concat(chunks)

        # Store original column order and row count
        self.original_columns = self.data.columns.tolist()
        original_row_count = len(self.data)

        # One-hot encode non-numeric columns
        categorical_columns = self.data.select_dtypes(
            include=["object", "category"]
        ).columns
        if len(categorical_columns) > 0:
            ohe = OneHotEncoder(sparse_output=False, drop="first")
            ohe_df = pd.DataFrame(
                ohe.fit_transform(self.data[categorical_columns]),
                columns=ohe.get_feature_names_out(categorical_columns),
            )
            self.data = pd.concat(
                [self.data.drop(categorical_columns, axis=1), ohe_df], axis=1
            )

        # Drop missing values
        self.data_cleaned = self.data.dropna()
        self.observations_removed = original_row_count - len(self.data_cleaned)

    def prepare_features(self):
        """Prepare features (X) and outcome (y) for model training."""
        if self.outcome_var not in self.data_cleaned.columns:
            raise ValueError(
                f"Outcome variable '{self.outcome_var}' not found in the data."
            )

        # Features (all columns except outcome variable)
        self.X = self.data_cleaned.drop(columns=[self.outcome_var]).to_numpy()
        self.y = self.data_cleaned[self.outcome_var].to_numpy()

        # Standardize the outcome variable
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)
        self.y_standardized = (self.y - self.y_mean) / self.y_std

    def train_model(
        self,
        model_name: str,
        num_trees: int,
        burn_in: int,
        num_draws: int,
        thinning: int,
    ):
        """Train the model based on the selected type."""
        try:
            if model_name == "BART":
                self.model = BARTModel()
                self.model.sample(
                    X_train=self.X,
                    y_train=self.y_standardized,
                    X_test=self.X,
                    num_trees=num_trees,
                    num_burnin=burn_in,
                    num_mcmc=num_draws,
                )
            elif model_name == "BCF":
                self.model = BCFModel()
                # Add BCF-specific training logic here
        except Exception as e:
            raise RuntimeError(f"Error during model training: {str(e)}")

    def predict(self):
        """Generate predictions and convert them back to the original scale."""
        self.y_pred_samples = self.model.predict(covariates=self.X)
        self.y_pred_samples = (
            self.y_pred_samples * self.y_std + self.y_mean
        )  # Convert back to original scale

        # Verify prediction length matches data length
        if len(self.y_pred_samples) != len(self.data_cleaned):
            raise ValueError(
                f"Prediction length ({len(self.y_pred_samples)}) doesn't match data length ({len(self.data_cleaned)})"
            )

        return {
            "Posterior Mean": np.mean(self.y_pred_samples, axis=1),
            "2.5th Percentile": np.percentile(self.y_pred_samples, 2.5, axis=1),
            "97.5th Percentile": np.percentile(self.y_pred_samples, 97.5, axis=1),
        }


class Arborist(QMainWindow):
    def __init__(self):
        super().__init__()

        self.training_worker = None
        self.progress_dialog = None
        self.full_predictions = None
        self.current_prediction_idx = 0

        # Load the stylesheet
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            style_path = os.path.join(current_dir, "style.qss")
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Error loading stylesheet: {e}")

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Arborist")

        self.statusBar = self.statusBar()
        self.statusBar.showMessage("Ready")

        # Set the default window size
        self.resize(1600, 900)

        # Allow resizing the window between 800x600 and full screen
        self.setMinimumSize(800, 600)

        # Center the window on the screen
        self.center_window()

        # Create a tab widget for switching between the file browser and dataset viewer
        self.tabs = QTabWidget()

        # Load the UI for all tabs (Browse, Train, and Predict)
        self.load_browse_tab_ui()
        self.load_train_tab_ui()
        self.load_predict_tab_ui()

        # Add the tabs to the main layout
        self.tabs.addTab(self.browse_tab, "Browse")
        self.tabs.addTab(self.train_tab, "Train")
        self.tabs.addTab(self.predict_tab, "Predict")

        # Set the central widget for the main window
        self.setCentralWidget(self.tabs)

        # Connect signal to track tab changes
        self.tabs.currentChanged.connect(self.check_model_frame_visibility)

        # Connect signal for model selection change
        self.train_ui.modelComboBox.currentIndexChanged.connect(
            self.check_model_frame_visibility
        )

        # Update the code generation text whenever a UI element changes
        self.train_ui.modelComboBox.currentIndexChanged.connect(
            self.update_code_gen_text
        )
        self.train_ui.treesSpinBox.valueChanged.connect(self.update_code_gen_text)
        self.train_ui.burnInSpinBox.valueChanged.connect(self.update_code_gen_text)
        self.train_ui.drawsSpinBox.valueChanged.connect(self.update_code_gen_text)
        self.train_ui.thinningSpinBox.valueChanged.connect(self.update_code_gen_text)
        self.train_ui.outcomeComboBox.currentIndexChanged.connect(
            self.update_code_gen_text
        )
        self.train_ui.treatmentComboBox.currentIndexChanged.connect(
            self.update_code_gen_text
        )

    def check_model_frame_visibility(self):
        """Show or hide the treatmentFrame based on the selected tab and model."""
        # Check if the current tab is "Train"
        is_train_tab = self.tabs.currentIndex() == 1

        # Check if the selected model is BCF or XBCF
        selected_model = self.train_ui.modelComboBox.currentText()
        is_bcf_xbcf_model = selected_model in ["BCF", "XBCF"]

        # Show or hide the treatment frame
        self.train_ui.treatmentFrame.setVisible(is_train_tab and is_bcf_xbcf_model)

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
        self.tree.header().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )  # Auto-adjust column width
        self.tree.header().setMinimumSectionSize(100)

        # Double click to open the file or enter directory
        self.tree.doubleClicked.connect(self.on_file_double_click)

        # Set up the file viewer
        self.file_viewer = self.browse_ui.file_viewer
        self.file_viewer.setSortingEnabled(True)

        # Set splitter sizes to 600 for the file browser, 1000 for the file viewer
        self.browse_ui.splitter.setSizes([600, 1000])

        # "Train Dataset" button setup
        self.open_button = self.browse_ui.openDatasetButton
        self.open_button.setVisible(
            False
        )  # Initially hidden until a dataset is selected
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

    def load_train_tab_ui(self):
        """Load and set up the train tab UI (for the dataset analysis)."""
        self.train_tab = QWidget()
        self.train_ui = Ui_TrainTab()
        self.train_ui.setupUi(self.train_tab)

        # Initially hide the treatment frame and parameters menu
        self.train_ui.treatmentFrame.setVisible(False)

        # Initially hide the parameters menu
        self.train_ui.parametersMenu.setVisible(False)

        # Initially hide the code generation text edit
        self.train_ui.codeGenTextEdit.setVisible(False)

        # Connect button to toggle parameters menu
        self.train_ui.parametersPushButton.clicked.connect(self.toggle_parameters_menu)

        # Connect button to toggle code generation text edit
        self.train_ui.codeGenPushButton.clicked.connect(self.toggle_code_gen_text)

        # No dataset label and analytics viewer
        self.no_dataset_label = self.train_ui.no_dataset_label
        self.analytics_viewer = self.train_ui.analytics_viewer

        # Outcome variable dropdown
        self.outcome_combo = self.train_ui.outcomeComboBox

        # Treatment variable dropdown
        self.treatment_combo = self.train_ui.treatmentComboBox

        # Train Model button
        self.train_button = self.train_ui.trainButton
        self.train_button.clicked.connect(self.train_model)

        # Initially, show only the "No dataset" label, not the dataset viewer
        self.no_dataset_label.setVisible(True)
        self.analytics_viewer.setVisible(False)

        # Outcome and treatment variable selection
        self.outcome_combo.currentIndexChanged.connect(self.highlight_selected_column)

    def generate_code(self):
        """Generate Python code to reproduce the analysis based on the current UI settings."""
        # Ensure a dataset is loaded
        if not hasattr(self, "current_file_path"):
            return "No dataset loaded."

        outcome_var = self.train_ui.outcomeComboBox.currentText()
        treatment_var = (
            self.train_ui.treatmentComboBox.currentText()
            if self.train_ui.treatmentFrame.isVisible()
            else None
        )
        model_name = self.train_ui.modelComboBox.currentText()

        # Retrieve model parameters from the UI
        num_trees = self.train_ui.treesSpinBox.value()
        burn_in = self.train_ui.burnInSpinBox.value()
        num_draws = self.train_ui.drawsSpinBox.value()

        # Generate the Python code as a string, ensuring proper formatting
        code = f"""
    # Python script to reproduce the analysis
    # Generated by Arborist Version 0.0.1 (arborist.app) on {pd.Timestamp.now()}

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from stochtree import {model_name}Model

    # Load the dataset
    data = pd.read_csv(r'{self.current_file_path}')

    # Outcome variable: {outcome_var}
    outcome_var = data['{outcome_var}']

    # Preprocess categorical variables
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        ohe_df = pd.DataFrame(ohe.fit_transform(data[categorical_columns]), columns=ohe.get_feature_names_out(categorical_columns))
        data = pd.concat([data.drop(categorical_columns, axis=1), ohe_df], axis=1)

    # Drop missing values
    data_cleaned = data.dropna()

    # Feature variables (all except outcome variable)
    X = data_cleaned.drop(columns=['{outcome_var}']).to_numpy()
    y = data_cleaned['{outcome_var}'].to_numpy()

    # Standardize the outcome variable
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_standardized = (y - y_mean) / y_std

    # Model training
    model = {model_name}Model()
    model.sample(
        X_train=X,
        y_train=y_standardized,
        X_test=X,
        num_trees={num_trees},
        num_burnin={burn_in},
        num_mcmc={num_draws}
    )

    # Generate predictions
    y_pred_samples = model.predict(covariates=X)
    y_pred_samples = y_pred_samples * y_std + y_mean  # Convert back to original scale

    # Compute posterior summaries
    posterior_mean = np.mean(y_pred_samples, axis=1)
    percentile_2_5 = np.percentile(y_pred_samples, 2.5, axis=1)
    percentile_97_5 = np.percentile(y_pred_samples, 97.5, axis=1)

    # Display results
    results = pd.DataFrame({{
        'Posterior Mean': posterior_mean,
        '2.5th Percentile': percentile_2_5,
        '97.5th Percentile': percentile_97_5
    }})
    print(results.head())
    """
        return code

    def toggle_code_gen_text(self):
        """Toggle the visibility of the code generation text box."""
        is_visible = self.train_ui.codeGenTextEdit.isVisible()
        self.train_ui.codeGenTextEdit.setVisible(not is_visible)
        if not is_visible:
            self.update_code_gen_text()

    def toggle_parameters_menu(self):
        """Toggle the visibility of the parameters menu."""
        is_visible = self.train_ui.parametersMenu.isVisible()
        self.train_ui.parametersMenu.setVisible(not is_visible)

    def update_code_gen_text(self):
        """Update the code generation text box whenever a UI element changes."""
        code = self.generate_code()
        self.train_ui.codeGenTextEdit.setPlainText(code)

    def load_predict_tab_ui(self):
        """Load and set up the predict tab UI."""
        self.predict_tab = QWidget()
        self.predict_ui = Ui_PredictTab()
        self.predict_ui.setupUi(self.predict_tab)

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
            self.open_button.setVisible(
                True
            )  # Show the 'Open' button after a dataset is loaded
        else:
            self.file_viewer.setModel(None)  # Clear the table if non-dataset file
            self.open_button.setVisible(
                False
            )  # Hide the 'Open' button if no dataset is loaded

    def navigate_to_directory(self, directory_path):
        """Navigate to the directory represented by the given path."""
        # Update navigation history
        if self.history_index < len(self.history) - 1:
            # If we have moved back in history and then navigate to a new directory
            self.history = self.history[: self.history_index + 1]
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
            headers = (
                first_chunk.columns.tolist()
            )  # Extract headers from the first chunk

            # Re-create chunk_iter including the first chunk
            chunk_iter = itertools.chain([first_chunk], chunk_iter)
            model = PandasTableModel(
                chunk_iter, headers, predictions=None
            )  # Explicitly pass None
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
        if hasattr(self, "current_file_path"):
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
        """Train the model using threading and progress tracking."""
        self.train_button.setEnabled(False)
        self.statusBar.showMessage("Initializing training...")

        try:
            if not hasattr(self, "current_file_path"):
                self.statusBar.showMessage("No dataset selected.")
                return

            outcome_var = self.train_ui.outcomeComboBox.currentText()
            if not outcome_var:
                self.statusBar.showMessage("Outcome variable not selected.")
                return

            # Create trainer instance
            trainer = ModelTrainer(self.current_file_path, outcome_var)
            self.statusBar.showMessage("Loading dataset...")

            # Get model parameters from UI
            model_params = {
                "model_name": self.train_ui.modelComboBox.currentText(),
                "num_trees": self.train_ui.treesSpinBox.value(),
                "burn_in": self.train_ui.burnInSpinBox.value(),
                "num_draws": self.train_ui.drawsSpinBox.value(),
                "thinning": self.train_ui.thinningSpinBox.value(),
            }

            # Progress dialog
            self.progress_dialog = QProgressDialog(
                "Training model...", "Cancel", 0, 100, self
            )
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setAutoClose(False)
            self.progress_dialog.setAutoReset(False)

            # Worker thread setup
            self.training_worker = ModelTrainingWorker(trainer, model_params)
            self.training_worker.progress.connect(self.update_progress)
            self.training_worker.finished.connect(self.handle_training_finished)
            self.training_worker.error.connect(self.handle_training_error)
            self.progress_dialog.canceled.connect(self.cancel_training)

            self.training_worker.start()
            self.progress_dialog.show()
            self.statusBar.showMessage("Training model...")

        finally:
            self.train_button.setEnabled(True)

    @Slot(int)
    def update_progress(self, value):
        """Update progress dialog and status bar based on progress value."""
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
        if value == 10:
            self.statusBar.showMessage("Loading data...")
        elif value == 30:
            self.statusBar.showMessage("Preparing features...")
        elif value == 40:
            self.statusBar.showMessage("Training model...")
        elif value == 80:
            self.statusBar.showMessage("Generating predictions...")
        elif value == 100:
            self.statusBar.showMessage("Training complete.")

    @Slot(dict, float)
    def handle_training_finished(self, predictions, training_time):
        """Handle successful model training completion."""
        if self.progress_dialog:
            self.progress_dialog.close()
        self.statusBar.showMessage(
            f"Model training finished in {training_time:.2f} seconds."
        )

        # Update training time display
        self.train_ui.trainingTimeValue.setText(f"{training_time:.2f} seconds")

        try:
            # Reload the data with predictions
            chunk_iter = pd.read_csv(self.current_file_path, chunksize=CHUNK_SIZE)
            first_chunk = next(chunk_iter)
            headers = [
                "Posterior Average ŷ",
                "2.5th percentile ŷ",
                "97.5th percentile ŷ",
            ]
            headers.extend(first_chunk.columns.tolist())
            # Re-create chunk_iter including the first chunk
            chunk_iter = itertools.chain([first_chunk], chunk_iter)
            model = PandasTableModel(chunk_iter, headers, predictions)
            self.analytics_viewer.setModel(model)
            self.analytics_viewer.resizeColumnsToContents()

            # Re-highlight the selected outcome variable column
            self.highlight_selected_column()

            # Print the number of observations removed
            # print(f"Number of observations removed due to missing data: {observations_removed}")

        except Exception as e:
            print(f"Error updating predictions: {e}")

    @Slot(str)
    def handle_training_error(self, error_message):
        """Handle training errors with a proper dialog and status bar update."""
        if self.progress_dialog:
            self.progress_dialog.close()
        self.statusBar.showMessage("Training error encountered.")

        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Training Error")
        error_dialog.setText("An error occurred during model training")
        error_dialog.setDetailedText(error_message)
        error_dialog.exec()

    def cancel_training(self):
        """Cancel the training process."""
        if self.training_worker:
            self.training_worker.stop()
            self.training_worker.wait()
            self.training_worker = None

    def closeEvent(self, event):
        """Handle application shutdown."""
        if self.training_worker:
            self.training_worker.stop()
            self.training_worker.wait()
        event.accept()

    def save_results(self, file_path):
        """Save predictions and model parameters."""
        if hasattr(self, "predictions"):
            # Save implementation
            pass


# Main function to start the application
def main():
    app = QApplication(sys.argv)
    main_window = Arborist()
    main_window.show()
    sys.exit(app.exec())


# Entry point of the script
if __name__ == "__main__":
    main()
