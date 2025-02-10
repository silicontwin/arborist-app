"""
The cross-platform app for efficiently performing Bayesian causal inference and supervised learning tasks using tree-based models, including BCF, BART, and XBART.
"""

import sys
import os
import pandas as pd
import numpy as np
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
    QFileDialog,
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
from PySide6.QtGui import QColor, QAction
from arborist.layouts.browse import Ui_BrowseTab
from arborist.layouts.train import Ui_TrainTab
from arborist.layouts.predict import Ui_PredictTab
from stochtree import BCFModel, BARTModel
import time
import requests
import webbrowser
import pyarrow.csv as pa_csv
import pyarrow.compute as pc

# Current version and repo details
CURRENT_VERSION = "v0.0.1"
GITHUB_REPO = "silicontwin/arborist-app"


# PandasTableModel (the entire dataset is loaded)
class PandasTableModel(QAbstractTableModel):
    def __init__(self, data, headers, predictions=None):
        super().__init__()
        self.headers = headers
        self.selected_column_name = None
        self.predictions = predictions

        # Zebra stripe colors
        self.alternate_row_color = QColor("#F5F5F5")
        self.base_row_color = QColor("#FFFFFF")

        print("Initializing PandasTableModel with data type:", type(data))
        print("Headers:", headers)
        print("Predictions provided:", predictions is not None)

        if isinstance(data, pd.DataFrame):
            self._data = data
            print("DataFrame loaded directly with shape:", self._data.shape)
        else:
            if isinstance(data, str):
                # Load the entire CSV via pyarrow
                table = pa_csv.read_csv(data)
                self._data = table.to_pandas()
            else:
                self._data = pd.DataFrame(data)

            print("Data loaded with shape:", self._data.shape)

            if self.predictions is not None:
                # Create prediction DataFrame
                prediction_data = {}

                # Check if we have CATE predictions (BCF model)
                if "Posterior Mean CATE" in self.predictions:
                    prediction_data.update(
                        {
                            "CATE": self.predictions["Posterior Mean CATE"],
                            "2.5th percentile CATE": self.predictions[
                                "2.5th Percentile CATE"
                            ],
                            "97.5th percentile CATE": self.predictions[
                                "97.5th Percentile CATE"
                            ],
                        }
                    )

                # Add outcome predictions (both BART and BCF)
                prediction_data.update(
                    {
                        "Posterior Average ŷ": self.predictions["Posterior Mean"],
                        "2.5th percentile ŷ": self.predictions["2.5th Percentile"],
                        "97.5th percentile ŷ": self.predictions["97.5th Percentile"],
                    }
                )

                prediction_df = pd.DataFrame(prediction_data)
                self._data = pd.concat([prediction_df, self._data], axis=1)

        # Ensure the index is a simple range index matching the file order
        if not self._data.index.equals(pd.RangeIndex(len(self._data))):
            self._data.reset_index(drop=True, inplace=True)

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if pd.isnull(value):
                return ""
            return str(value)

        elif role == Qt.BackgroundRole:
            # Apply zebra striping based on the actual row number
            base_color = (
                self.alternate_row_color if index.row() % 2 else self.base_row_color
            )

            column_name = self.headers[index.column()]
            if self.selected_column_name == column_name:
                return QColor("#FFFFCB")  # Light yellow for selected column
            elif column_name in [
                "Posterior Average ŷ",
                "2.5th percentile ŷ",
                "97.5th percentile ŷ",
            ]:
                return QColor("#CCCCFF")  # Light blue for outcome predictions
            elif column_name in [
                "CATE",
                "2.5th percentile CATE",
                "97.5th percentile CATE",
            ]:
                return QColor("#FFE5CC")  # Light orange for CATE predictions
            else:
                return base_color

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
            else:
                # Return row numbers starting from 1
                return str(section + 1)
        return None

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
    finished = Signal(dict, float, object)
    error = Signal(str)

    def __init__(self, trainer, model_params):
        super().__init__()
        self.trainer = trainer
        self.model_params = model_params
        self._is_running = True

    def run(self):
        """Main method that runs in the separate thread with cancellation checks."""
        try:
            start_time = time.time()

            self.progress.emit(10)
            print("Loading data...")
            self.trainer.load_data()
            if not self._is_running:
                print("Training cancelled after loading data.")
                return

            self.progress.emit(30)
            print("Preparing features...")
            self.trainer.prepare_features()
            if not self._is_running:
                print("Training cancelled after preparing features.")
                return

            self.progress.emit(40)
            print("Training model...")
            self.trainer.train_model(**self.model_params)
            if not self._is_running:
                print("Training cancelled after training model.")
                return

            self.progress.emit(80)
            print("Generating predictions...")
            predictions = self.trainer.predict()
            if not self._is_running:
                print("Training cancelled after generating predictions.")
                return

            self.progress.emit(100)
            training_time = time.time() - start_time
            self.finished.emit(predictions, training_time, self.trainer.model)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """Stop the training process."""
        self._is_running = False


# Shared class for model training and preprocessing
class ModelTrainer:
    def __init__(self, file_path: str, outcome_var: str, treatment_var: str = None):
        self.file_path = file_path
        self.outcome_var = outcome_var
        self.treatment_var = treatment_var
        self.data = None
        self.data_cleaned = None
        self.X = None
        self.y = None
        self.Z = None
        self.model = None

    def load_data(self):
        print(f"Loading data from: {self.file_path}")

        table = pa_csv.read_csv(self.file_path)
        self.data = table.to_pandas()

        print(f"Initial data shape: {self.data.shape}")
        print("Columns:", self.data.columns.tolist())

        # Store original column order
        self.original_columns = self.data.columns.tolist()

        # # One-hot encode non-numeric columns
        # categorical_columns = self.data.select_dtypes(
        #     include=["object", "category"]
        # ).columns
        # if len(categorical_columns) > 0:
        #     ohe = OneHotEncoder(sparse_output=False, drop="first")
        #     ohe_df = pd.DataFrame(
        #         ohe.fit_transform(self.data[categorical_columns]),
        #         columns=ohe.get_feature_names_out(categorical_columns),
        #     )
        #     self.data = pd.concat(
        #         [self.data.drop(categorical_columns, axis=1), ohe_df], axis=1
        #     )

        # Make copy to avoid modifying original
        self.data_cleaned = self.data.copy()

        # Print column info
        print("\nColumn types:")
        print(self.data.dtypes)
        print("\nMissing values per column:")
        print(self.data.isnull().sum())

        # Convert everything to numeric if possible
        for col in self.data_cleaned.columns:
            try:
                self.data_cleaned[col] = pd.to_numeric(
                    self.data_cleaned[col], errors="coerce"
                )
            except Exception as e:
                print(f"Could not convert column {col} to numeric: {str(e)}")

        print(f"\nData shape after numeric conversion: {self.data_cleaned.shape}")

        # Check if treatment variable exists
        if self.treatment_var is not None:
            if self.treatment_var not in self.data_cleaned.columns:
                raise ValueError(
                    f"Treatment variable '{self.treatment_var}' not found in the data."
                )

        # Drop any completely missing columns
        empty_cols = [
            col
            for col in self.data_cleaned.columns
            if self.data_cleaned[col].isnull().all()
        ]
        if empty_cols:
            print(f"Dropping empty columns: {empty_cols}")
            self.data_cleaned = self.data_cleaned.drop(columns=empty_cols)

        # Drop rows with any missing values
        rows_before = len(self.data_cleaned)
        self.data_cleaned = self.data_cleaned.dropna()
        rows_removed = rows_before - len(self.data_cleaned)

        print(f"\nFinal cleaned data shape: {self.data_cleaned.shape}")
        if rows_removed > 0:
            print(f"Removed {rows_removed} rows with missing values")

        self.observations_removed = rows_removed

    def prepare_features(self):
        """Prepare features (X), outcome (y), and treatment (Z) for model training."""
        if self.outcome_var not in self.data_cleaned.columns:
            raise ValueError(
                f"Outcome variable '{self.outcome_var}' not found in the data."
            )

        if (
            self.treatment_var is not None
            and self.treatment_var not in self.data_cleaned.columns
        ):
            raise ValueError(
                f"Treatment variable '{self.treatment_var}' not found in the data."
            )

        # Features (all columns except outcome and treatment variables if treatment_var is provided)
        if self.treatment_var is not None:
            self.X = self.data_cleaned.drop(
                columns=[self.outcome_var, self.treatment_var]
            ).to_numpy()
            self.Z = self.data_cleaned[self.treatment_var].to_numpy()
        else:
            self.X = self.data_cleaned.drop(columns=[self.outcome_var]).to_numpy()
            self.Z = None

        self.y = self.data_cleaned[self.outcome_var].to_numpy()

        # Flatten `self.y` to ensure it's a 1D array
        self.y = self.y.ravel()

        # Standardize the outcome variable
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)
        self.y_standardized = (self.y - self.y_mean) / self.y_std

    def train_model(self, model_name, num_trees, burn_in, num_draws, thinning):
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
                if self.treatment_var is None:
                    raise ValueError(
                        "Treatment variable must be specified for BCF model."
                    )

                # Ensure Z is properly shaped
                Z_train = self.Z.astype(np.float64)
                Z_test = Z_train

                # Estimate propensity scores
                from sklearn.linear_model import LogisticRegression

                propensity_model = LogisticRegression()
                propensity_model.fit(self.X, Z_train)
                pi_train = propensity_model.predict_proba(self.X)[:, 1]
                pi_test = pi_train

                # Store propensity scores
                self.pi_train = pi_train
                self.pi_test = pi_test

                # Initialize BCF model
                self.model = BCFModel()

                # Set up parameters
                params = {
                    "num_trees_mu": num_trees,
                    "num_trees_tau": max(int(num_trees / 4), 10),
                    "num_burnin": burn_in,
                    "num_mcmc": num_draws,
                    "keep_burnin": False,
                    "keep_gfr": False,
                    "random_seed": 42,  # Add this to the params menu later
                }

                print("\nTraining BCF model with parameters:")
                for key, value in params.items():
                    print(f"{key}: {value}")

                # Train model
                self.model.sample(
                    X_train=self.X,
                    Z_train=Z_train,
                    y_train=self.y_standardized,
                    pi_train=pi_train,
                    X_test=self.X,
                    Z_test=Z_test,
                    pi_test=pi_test,
                    **params,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")

        except Exception as e:
            import traceback

            traceback_str = traceback.format_exc()
            raise RuntimeError(
                f"Error during model training: {str(e)}\n{traceback_str}"
            )

    def predict(self):
        """Generate predictions using the trained model."""
        try:
            if isinstance(self.model, BARTModel):
                # BART prediction handling
                self.y_pred_samples = self.model.predict(covariates=self.X)
                # Convert back to original scale
                y_pred_samples_original = self.y_pred_samples * self.y_std + self.y_mean

                # Calculate summary statistics (no CATE for BART)
                return {
                    "Posterior Mean": np.mean(y_pred_samples_original, axis=1),
                    "2.5th Percentile": np.percentile(
                        y_pred_samples_original, 2.5, axis=1
                    ),
                    "97.5th Percentile": np.percentile(
                        y_pred_samples_original, 97.5, axis=1
                    ),
                }
            elif isinstance(self.model, BCFModel):
                # BCF prediction handling
                if not hasattr(self.model, "y_hat_test"):
                    raise ValueError("No trained BCF model available")

                print("\nAccessing stored predictions...")
                print(f"Number of MCMC samples: {self.model.num_samples}")

                # Get stored predictions
                yhat_samples = self.model.y_hat_test
                tau_samples = self.model.tau_hat_test

                print("\nPrediction shapes:")
                print(f"yhat_samples shape: {yhat_samples.shape}")
                print(f"tau_samples shape: {tau_samples.shape}")

                # Remove singleton dimensions if present
                if yhat_samples.ndim == 3:
                    yhat_samples = yhat_samples.squeeze(-1)
                if tau_samples.ndim == 3:
                    tau_samples = tau_samples.squeeze(-1)

                print("\nAfter squeezing:")
                print(f"yhat_samples shape: {yhat_samples.shape}")
                print(f"tau_samples shape: {tau_samples.shape}")

                # Convert predictions back to original scale
                yhat_samples = yhat_samples * self.y_std + self.y_mean

                # Calculate summary statistics including CATE for BCF
                return {
                    "Posterior Mean": np.mean(yhat_samples, axis=1),
                    "2.5th Percentile": np.percentile(yhat_samples, 2.5, axis=1),
                    "97.5th Percentile": np.percentile(yhat_samples, 97.5, axis=1),
                    "Posterior Mean CATE": np.mean(tau_samples, axis=1),
                    "2.5th Percentile CATE": np.percentile(tau_samples, 2.5, axis=1),
                    "97.5th Percentile CATE": np.percentile(tau_samples, 97.5, axis=1),
                }
            else:
                raise ValueError("No trained model available")

        except Exception as e:
            import traceback

            print(f"\nError in predict method: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())
            print("\nModel attributes:")
            print(f"Has y_hat_test: {'y_hat_test' in dir(self.model)}")
            print(f"Has tau_hat_test: {'tau_hat_test' in dir(self.model)}")
            if hasattr(self.model, "y_hat_test"):
                print(f"y_hat_test type: {type(self.model.y_hat_test)}")
                print(f"y_hat_test shape: {self.model.y_hat_test.shape}")
            if hasattr(self.model, "tau_hat_test"):
                print(f"tau_hat_test type: {type(self.model.tau_hat_test)}")
                print(f"tau_hat_test shape: {self.model.tau_hat_test.shape}")
            print(f"Model type: {type(self.model)}")
            raise RuntimeError(f"Error during prediction: {str(e)}")

    def predict_outcome(self, model):
        """Predict outcomes for new data."""
        if self.data_cleaned is None:
            raise ValueError("No data loaded for prediction.")
        if len(self.data_cleaned) == 0:
            raise ValueError("No valid rows remaining after cleaning data.")
        if model is None:
            raise ValueError("No trained model provided.")

        try:
            print("\nPrediction data info:")
            print("Data shape:", self.data_cleaned.shape)
            print("Columns:", self.data_cleaned.columns.tolist())

            # Get feature columns (all numeric columns)
            feature_cols = self.data_cleaned.select_dtypes(
                include=["int64", "float64"]
            ).columns
            print(f"\nUsing features: {feature_cols.tolist()}")

            X_new = self.data_cleaned[feature_cols].to_numpy()
            print("Feature matrix shape:", X_new.shape)

            # Generate predictions
            print("\nGenerating predictions...")
            y_pred_samples = model.predict(covariates=X_new)
            print("Prediction samples shape:", y_pred_samples.shape)

            # Calculate summary statistics
            predictions = {
                "Posterior Mean": np.mean(y_pred_samples, axis=1),
                "2.5th Percentile": np.percentile(y_pred_samples, 2.5, axis=1),
                "97.5th Percentile": np.percentile(y_pred_samples, 97.5, axis=1),
            }

            print("\nPrediction summary:")
            for key, value in predictions.items():
                print(
                    f"{key} shape: {value.shape if hasattr(value, 'shape') else len(value)}"
                )

            return predictions

        except Exception as e:
            import traceback

            print(f"\nError in predict_outcome: {str(e)}")
            print("Traceback:")
            print(traceback.format_exc())
            raise RuntimeError(f"Error during prediction: {str(e)}")


class Arborist(QMainWindow):
    def __init__(self):
        super().__init__()

        self.trained_model = None  # Attribute to store the trained model instance
        self.training_worker = None
        self.progress_dialog = None
        self.full_predictions = None
        self.current_prediction_idx = 0
        self.dataset_opened = False

        # Load the stylesheet
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            style_path = os.path.join(current_dir, "style.qss")
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Error loading stylesheet: {e}")

        self.init_ui()
        self.create_menu_bar()

    def create_menu_bar(self):
        """Create the menu bar with File menu and Check for Updates option."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        # Create Check for Updates action
        check_updates_action = QAction("Check for Updates", self)
        check_updates_action.setStatusTip("Check for application updates")
        check_updates_action.triggered.connect(self.check_for_updates)
        file_menu.addAction(check_updates_action)

    def check_for_updates(self):
        """Handle the Check for Updates action using the GitHub Releases API."""
        try:
            # Construct the API URL for the latest release
            api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                release_info = response.json()
                latest_version = release_info.get("tag_name", "")
                print(f"Latest version from GitHub: {latest_version}")

                # Parse a version string into a tuple for comparison
                def parse_version(v: str):
                    return tuple(map(int, v.lstrip("v").split(".")))

                if latest_version and parse_version(latest_version) > parse_version(
                    CURRENT_VERSION
                ):
                    # New update available
                    reply = QMessageBox.question(
                        self,
                        "Update Available",
                        f"A new version ({latest_version}) is available. Would you like to download it?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    if reply == QMessageBox.Yes:
                        # Open the GitHub releases page
                        releases_url = (
                            f"https://github.com/{GITHUB_REPO}/releases/latest"
                        )
                        webbrowser.open(releases_url)
                else:
                    QMessageBox.information(
                        self,
                        "No Update Available",
                        "You are running the latest version.",
                    )
            else:
                QMessageBox.warning(
                    self,
                    "Update Check Failed",
                    f"Failed to check for updates. (HTTP Status: {response.status_code})",
                )
        except Exception as e:
            QMessageBox.warning(
                self,
                "Update Check Error",
                f"An error occurred while checking for updates:\n{str(e)}",
            )

    def init_ui(self):
        """Initialize the main UI."""
        self.setWindowTitle("Arborist | " + CURRENT_VERSION)

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

        # Initially disable Train and Predict tabs
        self.tabs.setTabEnabled(1, False)  # Train tab
        self.tabs.setTabEnabled(2, False)  # Predict tab

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
        self.file_viewer.setSortingEnabled(False)
        # Clear any sort indicator so the original order is preserved
        self.file_viewer.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.file_viewer.horizontalHeader().setSortIndicatorShown(False)

        # Add message label to file viewer
        self.no_dataset_message = QLabel(
            "Browse for a dataset using the file browser on the left, then double-click it to have it appear here.",
            self.file_viewer,
        )
        self.no_dataset_message.setAlignment(Qt.AlignCenter)
        self.no_dataset_message.setWordWrap(True)
        self.no_dataset_message.setStyleSheet("color: gray; font-size: 16px;")
        self.no_dataset_message.setGeometry(
            0, 0, self.file_viewer.width(), self.file_viewer.height()
        )
        self.no_dataset_message.show()

        # Adjust layout when resizing
        original_resize_event = self.file_viewer.resizeEvent

        def resize_event_override(event):
            self.no_dataset_message.setGeometry(
                0, 0, self.file_viewer.width(), self.file_viewer.height()
            )
            if callable(original_resize_event):
                original_resize_event(event)

        self.file_viewer.resizeEvent = resize_event_override

        # Set splitter sizes to 600 for the file browser, 1000 for the file viewer
        self.browse_ui.splitter.setSizes([600, 1000])

        # "Train Dataset" button setup
        self.open_button = self.browse_ui.openDatasetButton
        self.open_button.setVisible(
            False
        )  # Initially hidden until a dataset is selected
        self.open_button.clicked.connect(self.open_in_analytics_view)

        # Back, Up, and Forward buttons
        self.back_button = self.browse_ui.back_button
        self.forward_button = self.browse_ui.forward_button
        self.up_button = self.browse_ui.up_button
        self.back_button.clicked.connect(self.navigate_back)
        self.forward_button.clicked.connect(self.navigate_forward)
        self.up_button.clicked.connect(self.navigate_up)

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

        # Connect the reset button for the train tab
        self.train_ui.trainResetButton.clicked.connect(self.reset_train_tab)

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

    def reset_train_tab(self):
        """Reset the train tab to clear loaded dataset and training results.
        This clears the analytics viewer, resets the outcome/treatment combo boxes,
        resets the training time, and navigates the user back to the Browse tab."""
        # Clear the analytics viewer
        self.analytics_viewer.setModel(None)
        # Clear the outcome and treatment variable selections
        self.outcome_combo.clear()
        self.treatment_combo.clear()
        # Show the "No dataset" message and hide the analytics viewer
        self.no_dataset_label.setVisible(True)
        self.analytics_viewer.setVisible(False)
        # Reset the internal state for dataset and trained model
        self.dataset_opened = False
        self.trained_model = None
        self.train_ui.trainingTimeValue.setText("0 seconds")
        self.statusBar.showMessage(
            "Train tab reset. Please select a dataset from the Browse tab."
        )
        # Switch back to the Browse tab
        self.tabs.setCurrentIndex(0)

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
        if model_name == "BART":
            code = f"""
        # Python script to reproduce the analysis
        # Generated by Arborist Version 0.0.1 (https://arborist.app) on {pd.Timestamp.now()}

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
        elif model_name == "BCF":
            if not treatment_var:
                return "Treatment variable not selected."

            code = f"""
    # Python script to reproduce the analysis
    # Generated by Arborist Version 0.0.1 (arborist.app) on {pd.Timestamp.now()}

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from stochtree import {model_name}Model
    from sklearn.linear_model import LogisticRegression

    # Load the dataset
    data = pd.read_csv(r'{self.current_file_path}')

    # Outcome variable: {outcome_var}
    # Treatment variable: {treatment_var}
    y = data['{outcome_var}'].values
    Z = data['{treatment_var}'].values
    X = data.drop(columns=['{outcome_var}', '{treatment_var}']).values

    # Standardize the outcome variable
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_standardized = (y - y_mean) / y_std

    # Estimate propensity scores
    propensity_model = LogisticRegression()
    propensity_model.fit(X, Z)
    pi_train = propensity_model.predict_proba(X)[:, 1]
    pi_test = pi_train  # Using the same data for testing

    # Model training
    model = {model_name}Model()
    model.sample(
        X_train=X,
        Z_train=Z,
        y_train=y_standardized,
        pi_train=pi_train,
        X_test=X,
        Z_test=Z,
        pi_test=pi_test,
        num_burnin={burn_in},
        num_mcmc={num_draws},
        num_trees_mu={num_trees},
        num_trees_tau={int(num_trees / 4)},
    )

    # Generate predictions
    tau_samples, mu_samples, yhat_samples = model.predict(X=X, Z=Z, propensity=pi_test)
    yhat_samples = yhat_samples * y_std + y_mean  # Convert back to original scale

    # Compute posterior summaries
    posterior_mean = np.mean(yhat_samples, axis=1)
    percentile_2_5 = np.percentile(yhat_samples, 2.5, axis=1)
    percentile_97_5 = np.percentile(yhat_samples, 97.5, axis=1)

    posterior_cate_mean = np.mean(tau_samples, axis=1)
    cate_percentile_2_5 = np.percentile(tau_samples, 2.5, axis=1)
    cate_percentile_97_5 = np.percentile(tau_samples, 97.5, axis=1)

    # Display results
    results = pd.DataFrame({{
        'Posterior Mean': posterior_mean,
        '2.5th Percentile': percentile_2_5,
        '97.5th Percentile': percentile_97_5,
        'Posterior Mean CATE': posterior_cate_mean,
        '2.5th Percentile CATE': cate_percentile_2_5,
        '97.5th Percentile CATE': cate_percentile_97_5
    }})
    print(results.head())
    """

        else:
            code = "Model not recognized."

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

        # File selection for the prediction dataset
        self.predict_ui.selectFileButton.clicked.connect(self.select_predict_file)
        self.predict_ui.predictButton.clicked.connect(self.run_prediction)
        self.predict_ui.resetButton.clicked.connect(self.reset_predict_tab)

    def reset_predict_tab(self):
        """Reset the predict tab to allow re-running predictions.
        This clears any loaded prediction data and navigates the user back to the Browse tab.
        """
        # Clear the table view
        self.predict_ui.tableView.setModel(None)
        # Unset the prediction file path so that a new file can be selected
        self.predict_file_path = None
        self.statusBar.showMessage(
            "Predict tab reset. Please select a new file from the Browse tab."
        )
        # Switch to the Browse tab (index 0)
        self.tabs.setCurrentIndex(0)

    def run_prediction(self):
        """Run prediction on a new dataset with the trained model."""
        if self.trained_model is None:
            QMessageBox.warning(
                self,
                "Prediction Error",
                "No trained model available. Please train a model first.",
            )
            return

        try:
            if not hasattr(self, "predict_file_path"):
                QMessageBox.warning(
                    self, "File Error", "Please select a file for prediction."
                )
                return

            # Create progress dialog
            progress = QProgressDialog(
                "Generating predictions...", "Cancel", 0, 100, self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)

            # Initialize trainer and load data using pyarrow (entire dataset)
            trainer = ModelTrainer(self.predict_file_path, outcome_var=None)
            progress.setValue(30)

            trainer.load_data()
            progress.setValue(60)

            # Generate predictions
            predictions = trainer.predict_outcome(self.trained_model)
            progress.setValue(90)

            # Display predictions
            self.display_predictions(predictions)
            progress.setValue(100)

            self.statusBar.showMessage("Predictions generated successfully")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Prediction Error",
                f"An error occurred during prediction:\n{str(e)}",
            )
            self.statusBar.showMessage("Prediction failed")
        finally:
            if "progress" in locals():
                progress.close()

    def display_predictions(self, predictions):
        """Display predictions including appropriate columns based on model type."""
        try:
            print("\nDisplaying predictions:")
            print("Prediction keys:", predictions.keys())
            print("Prediction lengths:", {k: len(v) for k, v in predictions.items()})

            # Load the entire prediction dataset using pyarrow
            df = pa_csv.read_csv(self.predict_file_path).to_pandas()
            original_headers = df.columns.tolist()
            print("Original dataset headers:", original_headers)

            # Check if we have CATE predictions (BCF model)
            is_bcf = "Posterior Mean CATE" in predictions

            # Create combined headers based on model type
            if is_bcf:
                prediction_headers = [
                    "CATE",
                    "2.5th percentile CATE",
                    "97.5th percentile CATE",
                    "Posterior Average ŷ",
                    "2.5th percentile ŷ",
                    "97.5th percentile ŷ",
                ]
            else:
                prediction_headers = [
                    "Posterior Average ŷ",
                    "2.5th percentile ŷ",
                    "97.5th percentile ŷ",
                ]

            combined_headers = prediction_headers + original_headers
            print("Combined headers:", combined_headers)

            # Create prediction DataFrame
            prediction_data = {}
            if is_bcf:
                prediction_data.update(
                    {
                        "CATE": predictions["Posterior Mean CATE"],
                        "2.5th percentile CATE": predictions["2.5th Percentile CATE"],
                        "97.5th percentile CATE": predictions["97.5th Percentile CATE"],
                    }
                )
            prediction_data.update(
                {
                    "Posterior Average ŷ": predictions["Posterior Mean"],
                    "2.5th percentile ŷ": predictions["2.5th Percentile"],
                    "97.5th percentile ŷ": predictions["97.5th Percentile"],
                }
            )
            prediction_df = pd.DataFrame(prediction_data)

            # Concatenate prediction columns with the original dataset
            final_df = pd.concat([prediction_df, df], axis=1)
            model = PandasTableModel(final_df, combined_headers, predictions)
            self.predict_ui.tableView.setModel(model)
            self.predict_ui.tableView.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeToContents
            )
            self.predict_ui.tableView.verticalHeader().setVisible(True)

            # Enable sorting
            self.predict_ui.tableView.setSortingEnabled(False)
            # Clear any sort indicator so that the original order is preserved
            self.predict_ui.tableView.horizontalHeader().setSortIndicator(
                -1, Qt.AscendingOrder
            )
            self.predict_ui.tableView.horizontalHeader().setSortIndicatorShown(False)

            print(
                f"TableView configured with {model.rowCount()} rows and {model.columnCount()} columns"
            )
            self.statusBar.showMessage(
                "Successfully displaying predictions with original data"
            )

        except Exception as e:
            import traceback

            print("Error in display_predictions:", str(e))
            print("Traceback:", traceback.format_exc())
            QMessageBox.critical(
                self, "Display Error", f"Error displaying predictions: {str(e)}"
            )
            self.statusBar.showMessage("Error displaying predictions")

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
            # Open the file if it is a supported dataset
            self.load_csv_file(file_path, self.file_viewer)
            self.open_button.setVisible(True)  # Show the 'Open Dataset' button
        else:
            # Clear the file viewer and hide the open button for unsupported files
            self.file_viewer.setModel(None)
            self.open_button.setVisible(False)
            self.no_dataset_message.show()  # Show the "No dataset" message

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

        # Show the "No dataset" message if no dataset is loaded in the new directory
        self.file_viewer.setModel(None)
        self.no_dataset_message.show()
        self.open_button.setVisible(False)

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

    def navigate_up(self):
        """Navigate up one directory level from the current directory."""
        parent_directory = os.path.dirname(self.current_directory)
        # Check if parent directory exists and is different from the current
        if parent_directory and parent_directory != self.current_directory:
            self.navigate_to_directory(parent_directory)

    def update_tab_states(self):
        """Update the enabled/disabled state of the Train and Predict tabs."""
        # Train tab should only be enabled after Open Dataset button is clicked
        train_tab_enabled = hasattr(self, "dataset_opened") and self.dataset_opened
        self.tabs.setTabEnabled(1, train_tab_enabled)

        # Enable the Predict tab if a model has been trained
        predict_tab_enabled = self.trained_model is not None
        self.tabs.setTabEnabled(2, predict_tab_enabled)

    def load_csv_file(self, file_path, table_view):
        """Load the selected CSV file and display its contents."""
        try:
            self.current_file_path = file_path  # Store the current file path

            # Load entire CSV with pyarrow
            df = pa_csv.read_csv(file_path).to_pandas()

            # Ensure index matches the file order
            df.index = range(len(df))

            headers = df.columns.tolist()
            model = PandasTableModel(df, headers, predictions=None)
            table_view.setModel(model)
            table_view.setSortingEnabled(False)
            table_view.horizontalHeader().setSortIndicatorShown(False)

            # Automatically adjust the column width to fit the content and header
            table_view.resizeColumnsToContents()

            # Update the outcome combo box with column names
            self.outcome_combo.clear()
            self.outcome_combo.addItems(headers)

            # Hide the "No dataset" message and show the open button
            self.no_dataset_message.hide()
            self.open_button.setVisible(True)

            # Update the Train tab state
            self.update_tab_states()
        except Exception as e:
            print(f"Error loading file: {e}")
            table_view.setModel(None)
            # Show the "No dataset" message if loading fails
            self.no_dataset_message.show()
            self.open_button.setVisible(False)

    def highlight_selected_column(self):
        """Highlight the selected column in the analytics viewer."""
        selected_var = self.outcome_combo.currentText()
        model = self.analytics_viewer.model()
        if model and selected_var in model.headers:
            model.set_highlighted_column(selected_var)

    def open_in_analytics_view(self):
        """Open the dataset in the analytics view (Train tab)."""
        if hasattr(self, "current_file_path"):
            try:
                # Load entire CSV with pyarrow
                df = pa_csv.read_csv(self.current_file_path).to_pandas()
                headers = df.columns.tolist()
                model = PandasTableModel(df, headers)
                self.analytics_viewer.setModel(model)
                self.analytics_viewer.resizeColumnsToContents()

                # Enable sorting
                self.analytics_viewer.setSortingEnabled(False)
                # Clear any sort indicator so that the original order is preserved
                self.analytics_viewer.horizontalHeader().setSortIndicator(
                    -1, Qt.AscendingOrder
                )
                self.analytics_viewer.horizontalHeader().setSortIndicatorShown(False)

                # Update outcome and treatment variable dropdowns
                self.outcome_combo.clear()
                self.outcome_combo.addItems(headers)
                self.treatment_combo.clear()
                self.treatment_combo.addItems(headers)

                # Display dataset
                self.no_dataset_label.setVisible(False)
                self.analytics_viewer.setVisible(True)

                # Mark the dataset as opened
                self.dataset_opened = True

                # Enable the Train tab and switch to it
                self.update_tab_states()
                self.tabs.setCurrentIndex(1)
            except Exception as e:
                print(f"Error loading file in analytics view: {e}")
                self.analytics_viewer.setModel(None)
                self.no_dataset_label.setVisible(True)
                self.analytics_viewer.setVisible(False)
        else:
            # Show the "No dataset" message if no dataset is loaded
            self.no_dataset_label.setVisible(True)
            self.analytics_viewer.setVisible(False)

    def train_model(self):
        """Train the model using threading and progress tracking."""
        self.train_button.setEnabled(False)
        self.statusBar.showMessage("Initializing training...")

        try:
            self.start_time = time.time()  # Start the timer
            # Ensure a dataset is loaded and an outcome variable is selected
            if not hasattr(self, "current_file_path"):
                self.statusBar.showMessage("No dataset selected.")
                return

            outcome_var = self.train_ui.outcomeComboBox.currentText()
            if not outcome_var:
                self.statusBar.showMessage("Outcome variable not selected.")
                return

            # Retrieve model parameters from UI elements
            model_name = self.train_ui.modelComboBox.currentText()
            num_trees = self.train_ui.treesSpinBox.value()
            burn_in = self.train_ui.burnInSpinBox.value()
            num_draws = self.train_ui.drawsSpinBox.value()
            thinning = self.train_ui.thinningSpinBox.value()

            # Check for treatment variable if needed
            treatment_var = (
                self.train_ui.treatmentComboBox.currentText()
                if self.train_ui.treatmentFrame.isVisible()
                else None
            )

            # Initialize the ModelTrainer
            self.trainer = ModelTrainer(
                file_path=self.current_file_path,
                outcome_var=outcome_var,
                treatment_var=treatment_var,
            )

            # Model training parameters
            model_params = {
                "model_name": model_name,
                "num_trees": num_trees,
                "burn_in": burn_in,
                "num_draws": num_draws,
                "thinning": thinning,
            }

            # Initialize and start the training worker
            self.training_worker = ModelTrainingWorker(
                trainer=self.trainer, model_params=model_params
            )
            self.training_worker.progress.connect(self.update_progress)
            self.training_worker.finished.connect(self.handle_training_finished)
            self.training_worker.error.connect(self.handle_training_error)
            self.training_worker.start()

            # Set up a progress dialog to track model training
            self.progress_dialog = QProgressDialog(
                "Training model...", "Cancel", 0, 100, self
            )
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.canceled.connect(self.cancel_training)
            self.progress_dialog.setValue(0)

        except Exception as e:
            self.statusBar.showMessage(f"Error initializing training: {str(e)}")
            self.train_button.setEnabled(True)

    @Slot(int)
    def update_progress(self, value):
        """Update progress dialog and status bar based on progress value."""
        if self.progress_dialog:
            self.progress_dialog.setValue(value)

        elapsed_time = time.time() - self.start_time
        if value == 10:
            self.statusBar.showMessage(
                f"Loading data... (Elapsed: {elapsed_time:.2f} seconds)"
            )
        elif value == 30:
            self.statusBar.showMessage(
                f"Preparing features... (Elapsed: {elapsed_time:.2f} seconds)"
            )
        elif value == 40:
            self.statusBar.showMessage(
                f"Training model... (Elapsed: {elapsed_time:.2f} seconds)"
            )
        elif value == 80:
            self.statusBar.showMessage(
                f"Generating predictions... (Elapsed: {elapsed_time:.2f} seconds)"
            )
        elif value == 100:
            self.statusBar.showMessage(
                f"Training complete in {elapsed_time:.2f} seconds."
            )

    @Slot(dict, float, object)
    def handle_training_finished(self, predictions, training_time, model):
        """Handle successful model training completion."""
        if self.progress_dialog:
            self.progress_dialog.close()

        elapsed_time = time.time() - self.start_time
        self.statusBar.showMessage(
            f"Model training finished in {elapsed_time:.2f} seconds."
        )

        # Store the trained model for later use in predictions
        self.trained_model = model
        self.train_ui.trainingTimeValue.setText(f"{training_time:.2f} seconds")

        # Update tab states to enable Predict tab
        self.update_tab_states()

        # Reload the data with predictions
        try:
            # Load entire CSV with pyarrow
            df = pa_csv.read_csv(self.current_file_path).to_pandas()
            headers = df.columns.tolist()
            # Define additional headers for predictions based on model type
            if isinstance(model, BCFModel):
                pred_headers = [
                    "CATE",
                    "2.5th percentile CATE",
                    "97.5th percentile CATE",
                    "Posterior Average ŷ",
                    "2.5th percentile ŷ",
                    "97.5th percentile ŷ",
                ]
            else:
                pred_headers = [
                    "Posterior Average ŷ",
                    "2.5th percentile ŷ",
                    "97.5th percentile ŷ",
                ]
            full_headers = pred_headers + headers

            # Create prediction DataFrame
            prediction_data = {}
            if isinstance(model, BCFModel):
                prediction_data.update(
                    {
                        "CATE": predictions["Posterior Mean CATE"],
                        "2.5th percentile CATE": predictions["2.5th Percentile CATE"],
                        "97.5th percentile CATE": predictions["97.5th Percentile CATE"],
                    }
                )
            prediction_data.update(
                {
                    "Posterior Average ŷ": predictions["Posterior Mean"],
                    "2.5th percentile ŷ": predictions["2.5th Percentile"],
                    "97.5th percentile ŷ": predictions["97.5th Percentile"],
                }
            )
            prediction_df = pd.DataFrame(prediction_data)

            # Concatenate prediction columns with the original dataset
            combined_df = pd.concat([prediction_df, df], axis=1)
            model_table = PandasTableModel(combined_df, full_headers, predictions)
            self.analytics_viewer.setModel(model_table)
            self.analytics_viewer.resizeColumnsToContents()

            # Re-highlight the selected outcome variable column
            self.highlight_selected_column()

            # Print the number of observations removed
            # print(f"Number of observations removed due to missing data: {observations_removed}")

        except Exception as e:
            print(f"Error updating predictions: {e}")
            import traceback

            print("Traceback:", traceback.format_exc())

    @Slot(str)
    def handle_training_error(self, error_message):
        """Handle training errors with detailed error reporting."""
        try:
            # Ensure progress dialog is closed
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None

            # Reset training worker
            if self.training_worker:
                self.training_worker.stop()
                self.training_worker = None

            # Re-enable train button
            self.train_button.setEnabled(True)

            # Update status
            self.statusBar.showMessage("Training error encountered")

            # Create detailed error dialog
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Training Error")
            error_dialog.setText("An error occurred during model training")

            # Add detailed error information
            detailed_text = [
                "Error Details:",
                "-------------",
                str(error_message),
                "",
                "Debug Information:",
                "-----------------",
            ]

            # Add model parameters if available
            if hasattr(self, "trainer"):
                detailed_text.extend(
                    [
                        "Model Configuration:",
                        f"- Outcome variable: {self.trainer.outcome_var}",
                        f"- Treatment variable: {self.trainer.treatment_var}",
                        f"- Data shape: {self.trainer.X.shape if hasattr(self.trainer, 'X') else 'Not available'}",
                        "",
                    ]
                )

            error_dialog.setDetailedText("\n".join(detailed_text))

            # Set minimum width for better readability
            error_dialog.setMinimumWidth(400)

            # Show the dialog
            error_dialog.exec()

        except Exception as e:
            # Fallback error handling
            print(f"Error in error handler: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                "An error occurred while handling the training error.\n"
                f"Original error: {error_message}\n"
                f"Handler error: {str(e)}",
            )

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

    def select_predict_file(self):
        """Open a file dialog to select a file for prediction."""
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset for Prediction",
            desktop_path,  # Start in the Desktop directory
            "CSV Files (*.csv);;All Files (*)",
        )
        if file_path:
            self.predict_file_path = file_path  # Store the path for later use
            self.statusBar.showMessage(f"Selected prediction file: {file_path}")
        else:
            self.statusBar.showMessage("No file selected for prediction.")


# Main function to start the application
def main():
    app = QApplication(sys.argv)
    main_window = Arborist()
    main_window.show()
    sys.exit(app.exec())


# Entry point of the script
if __name__ == "__main__":
    main()
