"""
The cross-platform app for efficiently performing Bayesian causal inference and supervised learning tasks 
using tree-based models, including BCF, BART, and XBART.
"""

import sys
import os
import time
import traceback
import textwrap
import webbrowser
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pyarrow.csv as pa_csv

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileSystemModel,
    QLabel,
    QTabWidget,
    QWidget,
    QHeaderView,
    QProgressDialog,
    QMessageBox,
    QFileDialog,
    QToolButton,
    QHBoxLayout,
    QVBoxLayout,
)
from PySide6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    QThread,
    Signal,
    Slot,
    QSettings,
    QTimer,
)
from PySide6.QtGui import QColor, QAction

from arborist.layouts.load import Ui_LoadTab
from arborist.layouts.train import Ui_TrainTab
from arborist.layouts.predict import Ui_PredictTab
from stochtree import BCFModel, BARTModel

# Current version and repository details
CURRENT_VERSION = "v0.0.1"
GITHUB_REPO = "silicontwin/arborist-app"


def auto_one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically perform one-hot encoding on all categorical columns
    (columns of type 'object' or 'category') in the provided DataFrame.
    """
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_columns) > 0:
        ohe = OneHotEncoder(sparse_output=False, drop="first")
        ohe_df = pd.DataFrame(
            ohe.fit_transform(df[categorical_columns]),
            columns=ohe.get_feature_names_out(categorical_columns),
            index=df.index,
        )
        df = pd.concat([df.drop(categorical_columns, axis=1), ohe_df], axis=1)
        print(f"Performed one-hot encoding on columns: {list(categorical_columns)}")
    return df


class PandasTableModel(QAbstractTableModel):
    """
    A table model that wraps a pandas DataFrame for display in Qt views.

    It supports zebra striping and optional prediction columns.
    """

    def __init__(
        self, data: pd.DataFrame, headers: list[str], predictions: dict = None
    ):
        """
        Initialize the table model with data, headers, and optional prediction results.

        :param data: A pandas DataFrame or a file path string to a CSV.
        :param headers: A list of column names.
        :param predictions: Optional dictionary of prediction results.
        """
        super().__init__()
        self.headers = headers
        self.selected_column_name = None
        self.predictions = predictions

        # Define colors for zebra striping.
        self.alternate_row_color = QColor("#2A2A3A")  # Darker stripe
        self.base_row_color = QColor("#1E1E2F")  # Main dark background

        print("Initializing PandasTableModel with data type:", type(data))
        print("Headers:", headers)
        print("Predictions provided:", predictions is not None)

        if isinstance(data, pd.DataFrame):
            self._data = data
            print("DataFrame loaded directly with shape:", self._data.shape)
        else:
            if isinstance(data, str):
                # Load the entire CSV via pyarrow.
                table = pa_csv.read_csv(data)
                self._data = table.to_pandas()
            else:
                self._data = pd.DataFrame(data)

            print("Data loaded with shape:", self._data.shape)

            if self.predictions is not None:
                # Create a prediction DataFrame if predictions are provided.
                prediction_data = {}
                # Check for CATE predictions (used with the BCF model).
                if "Posterior Mean CATE" in self.predictions:
                    prediction_data.update(
                        {
                            "Posterior Mean CATE": self.predictions[
                                "Posterior Mean CATE"
                            ],
                            "2.5th Percentile CATE": self.predictions[
                                "2.5th Percentile CATE"
                            ],
                            "97.5th Percentile CATE": self.predictions[
                                "97.5th Percentile CATE"
                            ],
                            "Credible Interval Width CATE": self.predictions[
                                "Credible Interval Width CATE"
                            ],
                        }
                    )
                # Add outcome predictions (applicable to both BART and BCF).
                prediction_data.update(
                    {
                        "Posterior Mean": self.predictions["Posterior Mean"],
                        "2.5th Percentile": self.predictions["2.5th Percentile"],
                        "97.5th Percentile": self.predictions["97.5th Percentile"],
                        "Credible Interval Width": self.predictions[
                            "Credible Interval Width"
                        ],
                    }
                )
                prediction_df = pd.DataFrame(prediction_data)
                # Prepend the prediction columns to the data.
                self._data = pd.concat([prediction_df, self._data], axis=1)

        # Reset index to ensure it is a simple range index matching file order.
        if not self._data.index.equals(pd.RangeIndex(len(self._data))):
            self._data.reset_index(drop=True, inplace=True)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of rows in the data."""
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns (based on headers)."""
        return len(self.headers)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        """
        Return the data for the given index and role.

        For DisplayRole, returns string representation; for BackgroundRole, returns
        color information for zebra striping or highlighting.
        """
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if pd.isnull(value):
                return ""
            return str(value)
        elif role == Qt.BackgroundRole:
            column_name = self.headers[index.column()]
            lower = column_name.lower()
            # For prepended prediction columns, check if the header indicates predictions.
            if self.predictions is not None and (
                lower.startswith("posterior")
                or "percentile" in lower
                or "credible" in lower
            ):
                if "cate" in lower:
                    # For CATE predictions, use a warm highlight.
                    return QColor("#8A6A59")
                else:
                    # For other prediction columns, use a dark blue-gray accent.
                    return QColor("#3B4C5D")
            # Otherwise, apply zebra striping.
            base_color = (
                self.alternate_row_color if index.row() % 2 else self.base_row_color
            )
            # If a column is selected, override with an accent color.
            if self.selected_column_name == column_name:
                return QColor("#3A84DF")
            return base_color
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ):
        """
        Return header data for the given section and orientation.
        For horizontal headers, returns the column name; for vertical headers, returns row numbers.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
            else:
                return str(section + 1)  # Row numbers starting at 1.
        return None

    def set_highlighted_column(self, column_name: str) -> None:
        """
        Set the column to highlight and trigger a layout change.

        :param column_name: Name of the column to highlight.
        """
        self.selected_column_name = column_name
        self.layoutChanged.emit()


class DatasetFileFilterProxyModel(QSortFilterProxyModel):
    """
    A proxy model that filters files based on a set of dataset file extensions.

    Only files with extensions in the specified set will be accepted, though all directories are accepted.
    """

    def __init__(self, dataset_extensions: set[str], parent=None):
        super().__init__(parent)
        self.dataset_extensions = dataset_extensions

    def filterAcceptsRow(self, sourceRow: int, sourceParent: QModelIndex) -> bool:
        """
        Determine if the given row should be accepted based on its file extension.

        :param sourceRow: The row number in the source model.
        :param sourceParent: The parent index in the source model.
        :return: True if the row is accepted, False otherwise.
        """
        index = self.sourceModel().index(sourceRow, 0, sourceParent)
        file_name = self.sourceModel().fileName(index)
        if self.sourceModel().isDir(index):
            # Always accept directories.
            return True
        _, ext = os.path.splitext(file_name)
        return ext.lower() in self.dataset_extensions


class ModelTrainingWorker(QThread):
    """
    Worker thread for model training to prevent UI freezing.

    It emits progress, finished, and error signals during the training process.
    """

    progress = Signal(int)
    finished = Signal(dict, object)
    error = Signal(str)

    def __init__(self, trainer, model_params):
        """
        Initialize the worker thread.

        :param trainer: An instance of ModelTrainer.
        :param model_params: Dictionary of parameters for model training.
        """
        super().__init__()
        self.trainer = trainer
        self.model_params = model_params
        self._is_running = True

    def run(self):
        """Run the training process in a separate thread with cancellation checks."""
        try:
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
            self.finished.emit(predictions, self.trainer.model)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        """Stop the training process."""
        self._is_running = False


class ModelTrainer:
    """
    Shared class for model training and preprocessing.

    It handles loading, cleaning, feature extraction, training, and prediction.
    """

    def __init__(self, file_path: str, outcome_var: str, treatment_var: str = None):
        """
        Initialize the ModelTrainer with file path, outcome variable, and optional treatment variable.

        :param file_path: Path to the dataset file.
        :param outcome_var: Name of the outcome variable.
        :param treatment_var: Optional name of the treatment variable.
        """
        self.file_path: str = file_path
        self.outcome_var: str = outcome_var
        self.treatment_var: str | None = treatment_var
        self.data: pd.DataFrame | None = None
        self.data_cleaned: pd.DataFrame | None = None
        self.original_columns: list[str] = []
        self.original_row_count: int = 0
        self.cleaned_row_count: int = 0
        self.observations_removed: int = 0
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.Z: np.ndarray | None = None
        self.model = None

    def load_data(self) -> None:
        """
        Load data from the CSV file using pyarrow, apply one-hot encoding, and clean the data.

        This includes converting columns to numeric when possible, dropping columns that are completely missing,
        and removing rows with any missing values. Original and cleaned row counts are stored.
        """
        print(f"Loading data from: {self.file_path}")
        table = pa_csv.read_csv(self.file_path)
        self.data = table.to_pandas()
        self.original_row_count = len(self.data)
        print(f"Initial data shape: {self.data.shape}")
        print("Columns:", self.data.columns.tolist())
        self.original_columns = self.data.columns.tolist()
        self.data = auto_one_hot_encode(self.data)
        self.data_cleaned = self.data.copy()
        print("\nColumn types after one-hot encoding:")
        print(self.data.dtypes)
        print("\nMissing values per column:")
        print(self.data.isnull().sum())
        for col in self.data_cleaned.columns:
            try:
                self.data_cleaned[col] = pd.to_numeric(
                    self.data_cleaned[col], errors="coerce"
                )
            except Exception as e:
                print(f"Could not convert column {col} to numeric: {str(e)}")
        print(f"\nData shape after numeric conversion: {self.data_cleaned.shape}")
        if self.treatment_var is not None:
            if self.treatment_var not in self.data_cleaned.columns:
                raise ValueError(
                    f"Treatment variable '{self.treatment_var}' not found in the data."
                )
        empty_cols = [
            col
            for col in self.data_cleaned.columns
            if self.data_cleaned[col].isnull().all()
        ]
        if empty_cols:
            print(f"Dropping empty columns: {empty_cols}")
            self.data_cleaned = self.data_cleaned.drop(columns=empty_cols)
        rows_before = len(self.data_cleaned)
        self.data_cleaned = self.data_cleaned.dropna()
        self.observations_removed = rows_before - len(self.data_cleaned)
        self.cleaned_row_count = len(self.data_cleaned)
        print(f"\nFinal cleaned data shape: {self.data_cleaned.shape}")
        if self.observations_removed > 0:
            print(f"Removed {self.observations_removed} rows with missing values")

    def prepare_features(self) -> None:
        """
        Prepare the features (X), outcome (y), and treatment (Z) for model training.

        This function extracts the numeric data from the cleaned dataset and standardizes the outcome variable.
        """
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
        if self.treatment_var is not None:
            self.X = self.data_cleaned.drop(
                columns=[self.outcome_var, self.treatment_var]
            ).to_numpy()
            self.Z = self.data_cleaned[self.treatment_var].to_numpy()
        else:
            self.X = self.data_cleaned.drop(columns=[self.outcome_var]).to_numpy()
            self.Z = None
        self.y = self.data_cleaned[self.outcome_var].to_numpy().ravel()
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)
        if self.y_std == 0:
            raise ValueError(
                "Outcome variable has zero variance. Standardization is not possible."
            )
        self.y_standardized = (self.y - self.y_mean) / self.y_std

    def train_model(
        self,
        model_name: str,
        num_trees: int,
        burn_in: int,
        num_draws: int,
        thinning: int,
    ) -> None:
        """
        Train the model based on the selected type (BART or BCF) using the prepared features.

        For BART, it calls sample with X and the standardized outcome.
        For BCF, it additionally estimates propensity scores and requires a treatment variable.
        """
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
                Z_train = self.Z.astype(np.float64)
                Z_test = Z_train
                from sklearn.linear_model import LogisticRegression

                propensity_model = LogisticRegression()
                propensity_model.fit(self.X, Z_train)
                pi_train = propensity_model.predict_proba(self.X)[:, 1]
                pi_test = pi_train
                self.pi_train = pi_train
                self.pi_test = pi_test
                self.model = BCFModel()
                params = {
                    "num_trees_mu": num_trees,
                    "num_trees_tau": max(int(num_trees / 4), 10),
                    "num_burnin": burn_in,
                    "num_mcmc": num_draws,
                    "keep_burnin": False,
                    "keep_gfr": False,
                    "random_seed": 42,
                }
                print("\nTraining BCF model with parameters:")
                for key, value in params.items():
                    print(f"{key}: {value}")
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
            traceback_str = traceback.format_exc()
            raise RuntimeError(
                f"Error during model training: {str(e)}\n{traceback_str}"
            )

    def predict(self) -> dict:
        """
        Generate predictions using the trained model.

        For BART, predictions are generated using the predict method.
        For BCF, the stored predictions are accessed and processed.
        """
        try:
            if isinstance(self.model, BARTModel):
                self.y_pred_samples = self.model.predict(covariates=self.X)
                y_pred_samples_original = self.y_pred_samples * self.y_std + self.y_mean
                posterior_mean = np.mean(y_pred_samples_original, axis=1)
                perc_2_5 = np.percentile(y_pred_samples_original, 2.5, axis=1)
                perc_97_5 = np.percentile(y_pred_samples_original, 97.5, axis=1)
                ci_width = perc_97_5 - perc_2_5
                return {
                    "Posterior Mean": posterior_mean,
                    "2.5th Percentile": perc_2_5,
                    "97.5th Percentile": perc_97_5,
                    "Credible Interval Width": ci_width,
                }
            elif isinstance(self.model, BCFModel):
                if not hasattr(self.model, "y_hat_test"):
                    raise ValueError("No trained BCF model available")
                print("\nAccessing stored predictions...")
                print(f"Number of MCMC samples: {self.model.num_samples}")
                yhat_samples = self.model.y_hat_test
                tau_samples = self.model.tau_hat_test
                print("\nPrediction shapes:")
                print(f"yhat_samples shape: {yhat_samples.shape}")
                print(f"tau_samples shape: {tau_samples.shape}")
                if yhat_samples.ndim == 3:
                    yhat_samples = yhat_samples.squeeze(-1)
                if tau_samples.ndim == 3:
                    tau_samples = tau_samples.squeeze(-1)
                print("\nAfter squeezing:")
                print(f"yhat_samples shape: {yhat_samples.shape}")
                print(f"tau_samples shape: {tau_samples.shape}")
                yhat_samples = yhat_samples * self.y_std + self.y_mean
                posterior_mean = np.mean(yhat_samples, axis=1)
                perc_2_5 = np.percentile(yhat_samples, 2.5, axis=1)
                perc_97_5 = np.percentile(yhat_samples, 97.5, axis=1)
                ci_width = perc_97_5 - perc_2_5
                posterior_mean_cate = np.mean(tau_samples, axis=1)
                cate_perc_2_5 = np.percentile(tau_samples, 2.5, axis=1)
                cate_perc_97_5 = np.percentile(tau_samples, 97.5, axis=1)
                ci_width_cate = cate_perc_97_5 - cate_perc_2_5
                return {
                    "Posterior Mean": posterior_mean,
                    "2.5th Percentile": perc_2_5,
                    "97.5th Percentile": perc_97_5,
                    "Credible Interval Width": ci_width,
                    "Posterior Mean CATE": posterior_mean_cate,
                    "2.5th Percentile CATE": cate_perc_2_5,
                    "97.5th Percentile CATE": cate_perc_97_5,
                    "Credible Interval Width CATE": ci_width_cate,
                }
            else:
                raise ValueError("No trained model available")
        except Exception as e:
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

    def predict_outcome(self, model) -> dict:
        """
        Predict outcomes for new data using the provided model.

        This method loads the cleaned data, extracts numeric features, generates predictions,
        and calculates summary statistics.
        """
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
            feature_cols = self.data_cleaned.select_dtypes(
                include=["int64", "float64"]
            ).columns
            print(f"\nUsing features: {feature_cols.tolist()}")
            X_new = self.data_cleaned[feature_cols].to_numpy()
            print("Feature matrix shape:", X_new.shape)
            print("\nGenerating predictions...")
            y_pred_samples = model.predict(covariates=X_new)
            print("Prediction samples shape:", y_pred_samples.shape)
            posterior_mean = np.mean(y_pred_samples, axis=1)
            perc_2_5 = np.percentile(y_pred_samples, 2.5, axis=1)
            perc_97_5 = np.percentile(y_pred_samples, 97.5, axis=1)
            ci_width = perc_97_5 - perc_2_5
            predictions = {
                "Posterior Mean": posterior_mean,
                "2.5th Percentile": perc_2_5,
                "97.5th Percentile": perc_97_5,
                "Credible Interval Width": ci_width,
            }
            print("\nPrediction summary:")
            for key, value in predictions.items():
                print(
                    f"{key} shape: {value.shape if hasattr(value, 'shape') else len(value)}"
                )
            return predictions
        except Exception as e:
            print(f"\nError in predict_outcome: {str(e)}")
            print("Traceback:", traceback.format_exc())
            raise RuntimeError(f"Error during prediction: {str(e)}")


class TitleBar(QWidget):
    """
    A custom title bar widget with a title label and control buttons for minimize,
    maximize/restore, and close. It also supports window dragging.
    """

    def __init__(self, parent: QMainWindow = None) -> None:
        super().__init__(parent)
        self.setObjectName("TitleBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.parent = parent
        self._mousePos = None
        self._windowPos = None
        self.setFixedHeight(50)
        self.initUI()

    def initUI(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 0)
        # Title label displays the window title and current version.
        self.titleLabel = QLabel(
            "Arborist <span style='font-size:14px; color:#999999;'>"
            + CURRENT_VERSION
            + " alpha"
            + "</span>",
            self,
        )
        layout.addWidget(self.titleLabel)
        layout.addStretch()

        # Minimize button.
        self.minimizeButton = QToolButton(self)
        self.minimizeButton.setText("–")
        layout.addWidget(self.minimizeButton)
        # Maximize/Restore button.
        self.maximizeButton = QToolButton(self)
        self.maximizeButton.setText("□")
        layout.addWidget(self.maximizeButton)
        # Close button.
        self.closeButton = QToolButton(self)
        self.closeButton.setText("✕")
        layout.addWidget(self.closeButton)

        # Connect signals to their slots.
        self.minimizeButton.clicked.connect(self.parent.showMinimized)
        self.maximizeButton.clicked.connect(self.toggle_max_restore)
        self.closeButton.clicked.connect(self.parent.close)

    def toggle_max_restore(self) -> None:
        if self.parent.isMaximized():
            self.parent.showNormal()
        else:
            self.parent.showMaximized()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._mousePos = event.globalPosition().toPoint()
            self._windowPos = self.parent.pos()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() == Qt.LeftButton and self._mousePos is not None:
            diff = event.globalPosition().toPoint() - self._mousePos
            newPos = self._windowPos + diff
            self.parent.move(newPos)
            event.accept()


class Arborist(QMainWindow):
    """
    The main application window for Arborist.

    Responsible for UI setup, handling user interactions, and coordinating data loading,
    model training, prediction, and result display.
    """

    def __init__(self):
        """Initialize the main application window and load the stylesheet and UI."""
        super().__init__()
        self.trained_model = None  # Store the trained model instance.
        self.training_worker = None
        self.progress_dialog = None
        self.full_predictions = None
        self.current_prediction_idx = 0
        self.dataset_opened = False
        self.trainer = None

        # Remove the native window frame.
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)

        # Load the stylesheet.
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            style_path = os.path.join(current_dir, "style.qss")
            with open(style_path, "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Error loading stylesheet: {e}")

        self.init_ui()
        self.create_menu_bar()

    def create_menu_bar(self) -> None:
        """
        Create the menu bar with a File menu and a 'Check for Updates' action.
        """
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        check_updates_action = QAction("Check for Updates", self)
        check_updates_action.setStatusTip("Check for application updates")
        check_updates_action.triggered.connect(self.check_for_updates)
        file_menu.addAction(check_updates_action)

    def check_for_updates(self) -> None:
        """
        Handle the 'Check for Updates' action using the GitHub Releases API.
        """
        try:
            api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            response = requests.get(api_url, timeout=10)
            if response.status_code == 200:
                release_info = response.json()
                latest_version = release_info.get("tag_name", "")
                print(f"Latest version from GitHub: {latest_version}")

                def parse_version(v: str):
                    return tuple(map(int, v.lstrip("v").split(".")))

                if latest_version and parse_version(latest_version) > parse_version(
                    CURRENT_VERSION
                ):
                    reply = QMessageBox.question(
                        self,
                        "Update Available",
                        f"A new version ({latest_version}) is available. Would you like to download it?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    if reply == QMessageBox.Yes:
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

    def init_ui(self) -> None:
        """
        Initialize the main user interface, including window settings and tab layouts.
        """
        self.setWindowTitle("Arborist | " + CURRENT_VERSION)
        self.statusBar = self.statusBar()
        self.statusBar.showMessage("Ready")
        self.resize(1600, 900)
        self.setMinimumSize(800, 600)
        self.center_window()

        # Create the tabs widget and add content pages.
        self.tabs = QTabWidget()
        self.load_load_tab_ui()
        self.load_train_tab_ui()
        self.load_predict_tab_ui()
        self.tabs.addTab(self.load_tab, "Load")
        self.tabs.addTab(self.train_tab, "Train")
        self.tabs.addTab(self.predict_tab, "Predict")
        # Hide the actual tab bar to remove the visible tabs.
        self.tabs.tabBar().hide()

        self.tabs.setTabEnabled(1, False)  # Disable Train tab initially.
        self.tabs.setTabEnabled(2, False)  # Disable Predict tab initially.
        self.tabs.currentChanged.connect(self.check_model_frame_visibility)
        self.train_ui.modelComboBox.currentIndexChanged.connect(
            self.check_model_frame_visibility
        )
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

        # Container widget to hold our custom title bar and the tabs.
        container = QWidget()
        container.setObjectName("mainContainer")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.titleBar = TitleBar(self)
        layout.addWidget(self.titleBar)
        layout.addWidget(self.tabs)
        self.setCentralWidget(container)

    def check_model_frame_visibility(self) -> None:
        """
        Show or hide the treatment frame in the Train tab based on the selected model.
        """
        is_train_tab = self.tabs.currentIndex() == 1
        selected_model = self.train_ui.modelComboBox.currentText()
        is_bcf_xbcf_model = selected_model in ["BCF", "XBCF"]
        self.train_ui.treatmentFrame.setVisible(is_train_tab and is_bcf_xbcf_model)

    def load_load_tab_ui(self) -> None:
        """
        Set up the Load tab UI for file navigation and dataset selection.
        """
        self.load_tab = QWidget()
        self.load_ui = Ui_LoadTab()
        self.load_ui.setupUi(self.load_tab)

        settings = QSettings("UT Austin", "Arborist")
        default_dir = settings.value(
            "load/lastDirectory", os.path.join(os.path.expanduser("~"), "Desktop")
        )
        self.current_directory = default_dir

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(default_dir)
        dataset_extensions = {".csv", ".sav", ".dta"}
        self.proxy_model = DatasetFileFilterProxyModel(dataset_extensions)
        self.proxy_model.setSourceModel(self.file_model)
        # Disable dynamic sorting so that the proxy does not keep reordering
        self.proxy_model.setDynamicSortFilter(False)
        self.tree = self.load_ui.treeView
        # Freeze updates to prevent visual jumps during the initial load.
        self.tree.setUpdatesEnabled(False)
        # Set the model on the tree view.
        self.tree.setModel(self.proxy_model)
        # Disable the view’s own sorting to avoid reordering while items are added.
        self.tree.setSortingEnabled(False)
        source_index = self.file_model.index(default_dir)
        self.current_root_index = self.proxy_model.mapFromSource(source_index)
        self.tree.setRootIndex(self.current_root_index)
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tree.header().setMinimumSectionSize(100)
        self.tree.doubleClicked.connect(self.on_file_double_click)
        # Re-enable updates once the model is fully set.
        self.tree.setUpdatesEnabled(True)

        self.file_viewer = self.load_ui.file_viewer
        self.file_viewer.setSortingEnabled(False)
        self.file_viewer.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.file_viewer.horizontalHeader().setSortIndicatorShown(False)
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
        original_resize_event = self.file_viewer.resizeEvent

        def resize_event_override(event):
            self.no_dataset_message.setGeometry(
                0, 0, self.file_viewer.width(), self.file_viewer.height()
            )
            if callable(original_resize_event):
                original_resize_event(event)

        self.file_viewer.resizeEvent = resize_event_override
        self.load_ui.splitter.setSizes([600, 1000])
        self.open_button = self.load_ui.openDatasetButton
        self.open_button.setVisible(
            False
        )  # Initially hidden until a dataset is selected.
        self.open_button.clicked.connect(self.open_in_analytics_view)
        self.back_button = self.load_ui.back_button
        self.forward_button = self.load_ui.forward_button
        self.up_button = self.load_ui.up_button
        self.back_button.clicked.connect(self.navigate_back)
        self.forward_button.clicked.connect(self.navigate_forward)
        self.up_button.clicked.connect(self.navigate_up)
        self.history = [self.current_directory]
        self.history_index = 0
        self.back_button.setEnabled(False)
        self.forward_button.setEnabled(False)

        if settings.value("load/rememberDir", "true") == "true":
            self.load_ui.rememberDirCheckBox.setChecked(True)
        else:
            self.load_ui.rememberDirCheckBox.setChecked(False)
        # Update the status bar with the current directory when the app loads.
        self.update_directory_status()

    def update_directory_status(self) -> None:
        self.statusBar.showMessage(f"Current directory: {self.current_directory}")

    def load_train_tab_ui(self) -> None:
        """
        Load and set up the Train tab UI for dataset analysis and model training.
        """
        self.train_tab = QWidget()
        self.train_ui = Ui_TrainTab()
        self.train_ui.setupUi(self.train_tab)
        self.train_ui.treatmentFrame.setVisible(False)
        self.train_ui.parametersMenu.setVisible(False)
        self.train_ui.codeGenTextEdit.setVisible(False)
        self.train_ui.parametersPushButton.clicked.connect(self.toggle_parameters_menu)
        self.train_ui.codeGenPushButton.clicked.connect(self.toggle_code_gen_text)
        self.train_ui.trainResetButton.clicked.connect(self.reset_train_tab)
        self.no_dataset_label = self.train_ui.no_dataset_label
        self.analytics_viewer = self.train_ui.analytics_viewer
        self.outcome_combo = self.train_ui.outcomeComboBox
        self.treatment_combo = self.train_ui.treatmentComboBox
        self.train_button = self.train_ui.trainButton
        self.train_button.clicked.connect(self.train_model)
        self.no_dataset_label.setVisible(True)
        self.analytics_viewer.setVisible(False)
        self.outcome_combo.currentIndexChanged.connect(self.highlight_selected_column)

    def reset_train_tab(self) -> None:
        """
        Reset the application to its initial state, allowing the user to start over.
        This resets the Train tab, Load tab, and Predict tab.
        """
        # Cancel any ongoing training
        if self.training_worker:
            self.cancel_training()

        # Reset internal state variables
        self.current_file_path = None
        self.trainer = None
        self.trained_model = None
        self.dataset_opened = False
        self.predict_file_path = None

        # Reset the Load tab:
        # Clear the file viewer
        self.file_viewer.setModel(None)
        self.no_dataset_message.show()
        # Reset navigation history to the saved directory (if remember is checked), else to Desktop.
        settings = QSettings("UT Austin", "Arborist")
        if settings.value("load/rememberDir", "false") == "true":
            initial_dir = settings.value(
                "load/lastDirectory", os.path.join(os.path.expanduser("~"), "Desktop")
            )
        else:
            initial_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        self.history = [initial_dir]
        self.history_index = 0
        self.back_button.setEnabled(False)
        self.forward_button.setEnabled(False)
        source_index = self.file_model.index(initial_dir)
        self.current_root_index = self.proxy_model.mapFromSource(source_index)
        self.tree.setRootIndex(self.current_root_index)
        # Update the current directory and status bar
        self.current_directory = initial_dir
        self.update_directory_status()

        # Reset the Train tab UI:
        self.analytics_viewer.setModel(None)
        self.outcome_combo.clear()
        self.treatment_combo.clear()
        self.no_dataset_label.setVisible(True)
        self.analytics_viewer.setVisible(False)
        # Re-enable the train button so the user can run another analysis
        self.train_button.setEnabled(True)

        # Reset the Predict tab UI:
        self.predict_ui.tableView.setModel(None)

        # Switch back to the Load tab and disable Train and Predict tabs
        self.tabs.setCurrentIndex(0)
        self.statusBar.showMessage(
            "Application reset. Please select a dataset from the Load tab."
        )
        self.update_tab_states()

    def generate_code(self) -> str:
        """
        Generate a Python script that reproduces the analysis based on the current UI settings.
        The generated code is dedented for proper formatting.
        """
        if not hasattr(self, "current_file_path"):
            return "No dataset loaded."
        outcome_var = self.train_ui.outcomeComboBox.currentText()
        treatment_var = (
            self.train_ui.treatmentComboBox.currentText()
            if self.train_ui.treatmentFrame.isVisible()
            else None
        )
        model_name = self.train_ui.modelComboBox.currentText()
        num_trees = self.train_ui.treesSpinBox.value()
        burn_in = self.train_ui.burnInSpinBox.value()
        num_draws = self.train_ui.drawsSpinBox.value()

        # Format the current time in a human-readable style.
        # Note: Use "%-I" for Unix/Linux and "%#I" for Windows to remove leading zeros from the hour.
        current_time = pd.Timestamp.now().strftime("%b. %d, %Y at %-I:%M%p")

        if model_name == "BART":
            code = textwrap.dedent(
                f"""
                # Generated by Arborist {CURRENT_VERSION} (https://arborist.app) on {current_time}
                
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
                    ohe_df = pd.DataFrame(
                        ohe.fit_transform(data[categorical_columns]),
                        columns=ohe.get_feature_names_out(categorical_columns)
                    )
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
                credible_interval_width = percentile_97_5 - percentile_2_5

                # Display results
                results = pd.DataFrame({{
                    'Posterior Mean': posterior_mean,
                    '2.5th Percentile': percentile_2_5,
                    '97.5th Percentile': percentile_97_5,
                    'Credible Interval Width': credible_interval_width
                }})
                print(results.head())
                """
            )
        elif model_name == "BCF":
            if not treatment_var:
                return "Treatment variable not selected."
            code = textwrap.dedent(
                f"""
                # Generated by Arborist {CURRENT_VERSION} (https://arborist.app) on {current_time}
                
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

                # Compute posterior summaries for outcome predictions
                posterior_mean = np.mean(yhat_samples, axis=1)
                percentile_2_5 = np.percentile(yhat_samples, 2.5, axis=1)
                percentile_97_5 = np.percentile(yhat_samples, 97.5, axis=1)
                credible_interval_width = percentile_97_5 - percentile_2_5

                # Compute posterior summaries for CATE predictions
                posterior_cate_mean = np.mean(tau_samples, axis=1)
                cate_percentile_2_5 = np.percentile(tau_samples, 2.5, axis=1)
                cate_percentile_97_5 = np.percentile(tau_samples, 97.5, axis=1)
                credible_interval_width_cate = cate_percentile_97_5 - cate_percentile_2_5

                # Display results
                results = pd.DataFrame({{
                    'Posterior Mean': posterior_mean,
                    '2.5th Percentile': percentile_2_5,
                    '97.5th Percentile': percentile_97_5,
                    'Credible Interval Width': credible_interval_width,
                    'Posterior Mean CATE': posterior_cate_mean,
                    '2.5th Percentile CATE': cate_percentile_2_5,
                    '97.5th Percentile CATE': cate_percentile_97_5,
                    'Credible Interval Width CATE': credible_interval_width_cate
                }})
                print(results.head())
                """
            )
        else:
            code = "Model not recognized."
        return code

    def toggle_code_gen_text(self) -> None:
        """Toggle the visibility of the code generation text box."""
        is_visible = self.train_ui.codeGenTextEdit.isVisible()
        self.train_ui.codeGenTextEdit.setVisible(not is_visible)
        if not is_visible:
            self.update_code_gen_text()

    def toggle_parameters_menu(self) -> None:
        """Toggle the visibility of the parameters menu."""
        is_visible = self.train_ui.parametersMenu.isVisible()
        self.train_ui.parametersMenu.setVisible(not is_visible)

    def update_code_gen_text(self) -> None:
        """Update the code generation text box whenever a UI element changes."""
        code = self.generate_code()
        self.train_ui.codeGenTextEdit.setPlainText(code)

    def load_predict_tab_ui(self) -> None:
        """Load and set up the Predict tab UI for running predictions."""
        self.predict_tab = QWidget()
        self.predict_ui = Ui_PredictTab()
        self.predict_ui.setupUi(self.predict_tab)
        self.predict_ui.selectFileButton.clicked.connect(self.select_predict_file)
        self.predict_ui.predictButton.clicked.connect(self.run_prediction)
        self.predict_ui.resetButton.clicked.connect(self.reset_predict_tab)

    def reset_predict_tab(self) -> None:
        """
        Reset the Predict tab by clearing any loaded prediction data and switching back to the Load tab.
        """
        self.predict_ui.tableView.setModel(None)
        self.predict_file_path = None
        self.statusBar.showMessage(
            "Predict tab reset. Please select a new file from the Load tab."
        )
        self.tabs.setCurrentIndex(0)
        self.update_tab_states()

    def run_prediction(self) -> None:
        """
        Run predictions on a new dataset using the trained model.

        This method uses a progress dialog during prediction and then displays the results.
        """
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
            progress = QProgressDialog(
                "Generating predictions...", "Cancel", 0, 100, self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            trainer = ModelTrainer(self.predict_file_path, outcome_var=None)
            progress.setValue(30)
            trainer.load_data()
            progress.setValue(60)
            predictions = trainer.predict_outcome(self.trained_model)
            progress.setValue(90)
            self.display_predictions(predictions, trainer.data_cleaned)
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

    def display_predictions(
        self, predictions: dict, cleaned_data: pd.DataFrame
    ) -> None:
        """
        Display the predictions alongside the cleaned dataset.

        This method combines prediction summary columns with the cleaned data and configures the table view.
        """
        try:
            print("\nDisplaying predictions:")
            print("Prediction keys:", predictions.keys())
            print("Prediction lengths:", {k: len(v) for k, v in predictions.items()})
            df = cleaned_data
            headers = df.columns.tolist()
            print("Cleaned dataset headers:", headers)
            is_bcf = "Posterior Mean CATE" in predictions
            if is_bcf:
                prediction_headers = [
                    "Posterior Mean CATE",
                    "2.5th Percentile CATE",
                    "97.5th Percentile CATE",
                    "Credible Interval Width CATE",
                    "Posterior Average ŷ",
                    "2.5th percentile ŷ",
                    "97.5th percentile ŷ",
                    "Credible Interval Width ŷ",
                ]
            else:
                prediction_headers = [
                    "Posterior Average ŷ",
                    "2.5th percentile ŷ",
                    "97.5th percentile ŷ",
                    "Credible Interval Width ŷ",
                ]
            combined_headers = prediction_headers + headers
            print("Combined headers:", combined_headers)
            prediction_data = {}
            if is_bcf:
                prediction_data.update(
                    {
                        "Posterior Mean CATE": predictions["Posterior Mean CATE"],
                        "2.5th Percentile CATE": predictions["2.5th Percentile CATE"],
                        "97.5th Percentile CATE": predictions["97.5th Percentile CATE"],
                        "Credible Interval Width CATE": predictions[
                            "Credible Interval Width CATE"
                        ],
                    }
                )
            prediction_data.update(
                {
                    "Posterior Average ŷ": predictions["Posterior Mean"],
                    "2.5th percentile ŷ": predictions["2.5th Percentile"],
                    "97.5th percentile ŷ": predictions["97.5th Percentile"],
                    "Credible Interval Width ŷ": predictions["Credible Interval Width"],
                }
            )
            prediction_df = pd.DataFrame(prediction_data)
            combined_df = pd.concat([prediction_df, df], axis=1)
            model = PandasTableModel(combined_df, combined_headers, predictions)
            self.predict_ui.tableView.setModel(model)
            self.predict_ui.tableView.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeToContents
            )
            self.predict_ui.tableView.verticalHeader().setVisible(True)
            self.predict_ui.tableView.setSortingEnabled(False)
            self.predict_ui.tableView.horizontalHeader().setSortIndicator(
                -1, Qt.AscendingOrder
            )
            self.predict_ui.tableView.horizontalHeader().setSortIndicatorShown(False)
            print(
                f"TableView configured with {model.rowCount()} rows and {model.columnCount()} columns"
            )
            self.statusBar.showMessage(
                "Successfully displaying predictions with cleaned data"
            )
        except Exception as e:
            print("Error in display_predictions:", str(e))
            print("Traceback:", traceback.format_exc())
            QMessageBox.critical(
                self, "Display Error", f"Error displaying predictions: {str(e)}"
            )
            self.statusBar.showMessage("Error displaying predictions")

    def center_window(self) -> None:
        """
        Center the main window on the current screen.
        """
        screen = self.screen()  # Get the current screen.
        screen_geometry = screen.availableGeometry()
        window_geometry = self.frameGeometry()
        center_point = screen_geometry.center()  # Calculate center point.
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())

    def on_file_double_click(self, index: QModelIndex) -> None:
        """
        Handle double-click events on files and directories in the file tree view.

        Directories will be navigated into; supported dataset files will be loaded.
        """
        source_index = self.proxy_model.mapToSource(index)
        file_path = self.file_model.filePath(source_index)
        if self.file_model.isDir(source_index):
            self.navigate_to_directory(file_path)
        elif file_path.endswith((".csv", ".sav", ".dta")):
            self.load_csv_file(file_path, self.file_viewer)
            self.open_button.setVisible(True)
        else:
            self.file_viewer.setModel(None)
            self.open_button.setVisible(False)
            self.no_dataset_message.show()

    def navigate_to_directory(self, directory_path: str) -> None:
        """
        Navigate to the specified directory and update the file tree view.

        This method updates the navigation history and the UI navigation buttons.
        """
        if self.history_index < len(self.history) - 1:
            self.history = self.history[: self.history_index + 1]
        self.history.append(directory_path)
        self.history_index += 1
        self.current_directory = directory_path
        source_index = self.file_model.index(directory_path)
        proxy_index = self.proxy_model.mapFromSource(source_index)
        self.tree.setRootIndex(proxy_index)
        self.current_root_index = proxy_index
        self.back_button.setEnabled(self.history_index > 0)
        self.forward_button.setEnabled(self.history_index < len(self.history) - 1)
        self.file_viewer.setModel(None)
        self.no_dataset_message.show()
        self.open_button.setVisible(False)
        # Update the stored last directory if the checkbox is checked.
        self.update_last_directory()
        # Update the status bar to show the current directory.
        self.update_directory_status()

    def navigate_back(self) -> None:
        """
        Navigate back in the directory navigation history.
        """
        if self.history_index > 0:
            self.history_index -= 1
            directory_path = self.history[self.history_index]
            self.current_directory = directory_path
            source_index = self.file_model.index(directory_path)
            proxy_index = self.proxy_model.mapFromSource(source_index)
            self.tree.setRootIndex(proxy_index)
            self.current_root_index = proxy_index
            self.back_button.setEnabled(self.history_index > 0)
            self.forward_button.setEnabled(self.history_index < len(self.history) - 1)
            self.update_last_directory()
            self.update_directory_status()

    def navigate_forward(self) -> None:
        """
        Navigate forward in the directory navigation history.
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            directory_path = self.history[self.history_index]
            self.current_directory = directory_path
            source_index = self.file_model.index(directory_path)
            proxy_index = self.proxy_model.mapFromSource(source_index)
            self.tree.setRootIndex(proxy_index)
            self.current_root_index = proxy_index
            self.back_button.setEnabled(self.history_index > 0)
            self.forward_button.setEnabled(self.history_index < len(self.history) - 1)
            self.update_last_directory()
            self.update_directory_status()

    def navigate_up(self) -> None:
        """
        Navigate up one level from the current directory.
        """
        parent_directory = os.path.dirname(self.current_directory)
        if parent_directory and parent_directory != self.current_directory:
            self.navigate_to_directory(parent_directory)

    def update_tab_states(self) -> None:
        """
        Update the enabled/disabled state of the Train and Predict tabs.
        The Train tab should only be enabled after explicitly opening a dataset
        via the Open Dataset button, while the Predict tab requires a trained model.
        """
        # Train tab enabled only after using "Open Dataset" button
        self.tabs.setTabEnabled(1, self.dataset_opened)

        # Predict tab enabled only after successful model training
        predict_tab_enabled = self.trained_model is not None
        self.tabs.setTabEnabled(2, predict_tab_enabled)

    def load_csv_file(self, file_path: str, table_view) -> None:
        """
        Load the selected CSV file using ModelTrainer and display its cleaned contents.
        """
        try:
            self.current_file_path = file_path
            if (not hasattr(self, "trainer")) or (self.trainer is None):
                self.trainer = ModelTrainer(file_path, outcome_var="")
            else:
                self.trainer.file_path = file_path

            self.trainer.load_data()
            df = self.trainer.data_cleaned
            headers = df.columns.tolist()
            model = PandasTableModel(df, headers, predictions=None)
            table_view.setModel(model)
            table_view.setSortingEnabled(False)
            table_view.horizontalHeader().setSortIndicatorShown(False)
            table_view.resizeColumnsToContents()

            self.outcome_combo.clear()
            self.outcome_combo.addItems(headers)
            self.treatment_combo.clear()
            self.treatment_combo.addItems(headers)

            self.no_dataset_message.hide()
            self.open_button.setVisible(True)

            # Don't set dataset_opened here - requires Open Dataset button
            self.update_tab_states()

            self.statusBar.showMessage(
                f"Loaded dataset: original rows {self.trainer.original_row_count}, "
                f"cleaned rows {self.trainer.cleaned_row_count} "
                f"(removed {self.trainer.observations_removed} rows)"
            )
        except Exception as e:
            print(f"Error loading file: {e}")
            table_view.setModel(None)
            self.no_dataset_message.show()
            self.open_button.setVisible(False)

    def highlight_selected_column(self) -> None:
        """
        Highlight the currently selected outcome column in the analytics viewer.
        """
        selected_var = self.outcome_combo.currentText()
        model = self.analytics_viewer.model()
        if model and selected_var in model.headers:
            model.set_highlighted_column(selected_var)

    def open_in_analytics_view(self) -> None:
        """
        Open the dataset in the analytics view (Train tab) using auto-one hot encoding.
        This is triggered by the Open Dataset button and enables the Train tab.
        """
        if hasattr(self, "current_file_path"):
            try:
                if (not hasattr(self, "trainer")) or (self.trainer is None):
                    self.trainer = ModelTrainer(self.current_file_path, outcome_var="")
                else:
                    self.trainer.file_path = self.current_file_path

                self.trainer.load_data()
                df = self.trainer.data_cleaned
                headers = df.columns.tolist()
                model = PandasTableModel(df, headers)
                self.analytics_viewer.setModel(model)
                self.analytics_viewer.resizeColumnsToContents()

                self.outcome_combo.clear()
                self.outcome_combo.addItems(headers)
                self.treatment_combo.clear()
                self.treatment_combo.addItems(headers)

                self.no_dataset_label.setVisible(False)
                self.analytics_viewer.setVisible(True)

                # Set dataset_opened to True only when explicitly opened via this method
                self.dataset_opened = True
                self.update_tab_states()

                # Switch to Train tab
                self.tabs.setCurrentIndex(1)

                self.statusBar.showMessage(
                    f"Opened dataset: original rows {self.trainer.original_row_count}, "
                    f"cleaned rows {self.trainer.cleaned_row_count} "
                    f"(removed {self.trainer.observations_removed} rows)"
                )
            except Exception as e:
                print(f"Error loading file in analytics view: {e}")
                self.analytics_viewer.setModel(None)
                self.no_dataset_label.setVisible(True)
                self.analytics_viewer.setVisible(False)
                self.dataset_opened = False
                self.update_tab_states()

    def train_model(self) -> None:
        """
        Train the selected model using threading and display progress.
        """
        self.train_button.setEnabled(False)
        self.statusBar.showMessage("Initializing training...")

        try:
            # Start tracking time in the UI layer
            self.training_start_time = time.time()

            # Create a QTimer to update the elapsed time display
            self.training_timer = QTimer()
            self.training_timer.timeout.connect(self.update_training_time)
            self.training_timer.start(1000)  # Update every second

            if not hasattr(self, "current_file_path"):
                self.statusBar.showMessage("No dataset selected.")
                return

            outcome_var = self.train_ui.outcomeComboBox.currentText()
            if not outcome_var:
                self.statusBar.showMessage("Outcome variable not selected.")
                return

            # Setup progress dialog with dynamic time display
            self.progress_dialog = QProgressDialog(
                "Initializing training...", "Cancel", 0, 100, self
            )
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.canceled.connect(self.cancel_training)
            self.progress_dialog.setValue(0)

            # Create training parameters
            model_name = self.train_ui.modelComboBox.currentText()
            num_trees = self.train_ui.treesSpinBox.value()
            burn_in = self.train_ui.burnInSpinBox.value()
            num_draws = self.train_ui.drawsSpinBox.value()
            thinning = self.train_ui.thinningSpinBox.value()
            treatment_var = (
                self.train_ui.treatmentComboBox.currentText()
                if self.train_ui.treatmentFrame.isVisible()
                else None
            )

            self.trainer = ModelTrainer(
                file_path=self.current_file_path,
                outcome_var=outcome_var,
                treatment_var=treatment_var,
            )

            model_params = {
                "model_name": model_name,
                "num_trees": num_trees,
                "burn_in": burn_in,
                "num_draws": num_draws,
                "thinning": thinning,
            }

            self.training_worker = ModelTrainingWorker(
                trainer=self.trainer, model_params=model_params
            )
            self.training_worker.progress.connect(self.update_progress)
            self.training_worker.finished.connect(self.handle_training_finished)
            self.training_worker.error.connect(self.handle_training_error)
            self.training_worker.start()

        except Exception as e:
            self.statusBar.showMessage(f"Error initializing training: {str(e)}")
            self.train_button.setEnabled(True)
            if hasattr(self, "training_timer"):
                self.training_timer.stop()

    def update_training_time(self) -> None:
        """
        Update both the status bar and progress dialog with current training time.
        """
        if hasattr(self, "training_start_time"):
            elapsed_time = time.time() - self.training_start_time
            current_progress = (
                self.progress_dialog.value() if hasattr(self, "progress_dialog") else 0
            )

            # Determine the current stage based on progress
            status_prefix = (
                "Loading data..."
                if current_progress <= 10
                else (
                    "Preparing features..."
                    if current_progress <= 30
                    else (
                        "Training model..."
                        if current_progress <= 40
                        else (
                            "Generating predictions..."
                            if current_progress <= 80
                            else "Finishing up..."
                        )
                    )
                )
            )

            # Update both status bar and progress dialog
            status_text = f"{status_prefix} (Elapsed: {elapsed_time:.2f} seconds)"
            self.statusBar.showMessage(status_text)

            if hasattr(self, "progress_dialog") and self.progress_dialog:
                self.progress_dialog.setLabelText(status_text)

    @Slot(int)
    def update_progress(self, value: int) -> None:
        """
        Update the progress dialog and status bar as the training process advances.

        :param value: The current progress percentage.
        """
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.setValue(value)

    @Slot(dict, object)
    def handle_training_finished(self, predictions: dict, model: object) -> None:
        """
        Handle completion of the training process.

        This method updates the status bar with cleaning statistics, stores the trained model,
        and displays predictions in the analytics viewer.
        """
        # Stop the timer
        if hasattr(self, "training_timer"):
            self.training_timer.stop()

        # Close the progress dialog
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.close()

        # Calculate total training time
        total_time = time.time() - self.training_start_time

        # Update status with final message
        self.statusBar.showMessage(
            f"Model training finished in {total_time:.2f} seconds. "
            f"Data cleaning: removed {self.trainer.observations_removed} rows; "
            f"cleaned data: {self.trainer.cleaned_row_count} rows out of {self.trainer.original_row_count}."
        )

        self.trained_model = model
        self.update_tab_states()

        try:
            # Update the display with predictions
            df = self.trainer.data_cleaned
            headers = df.columns.tolist()

            if isinstance(model, BCFModel):
                pred_headers = [
                    "Posterior Mean CATE",
                    "2.5th Percentile CATE",
                    "97.5th Percentile CATE",
                    "Credible Interval Width CATE",
                    "Posterior Mean",
                    "2.5th Percentile",
                    "97.5th Percentile",
                    "Credible Interval Width",
                ]
            else:
                pred_headers = [
                    "Posterior Mean",
                    "2.5th Percentile",
                    "97.5th Percentile",
                    "Credible Interval Width",
                ]
            full_headers = pred_headers + headers
            prediction_data = {}
            if isinstance(model, BCFModel):
                prediction_data.update(
                    {
                        "Posterior Mean CATE": predictions["Posterior Mean CATE"],
                        "2.5th Percentile CATE": predictions["2.5th Percentile CATE"],
                        "97.5th Percentile CATE": predictions["97.5th Percentile CATE"],
                        "Credible Interval Width CATE": predictions[
                            "Credible Interval Width CATE"
                        ],
                    }
                )
            prediction_data.update(
                {
                    "Posterior Average ŷ": predictions["Posterior Mean"],
                    "2.5th percentile ŷ": predictions["2.5th Percentile"],
                    "97.5th percentile ŷ": predictions["97.5th Percentile"],
                    "Credible Interval Width ŷ": predictions["Credible Interval Width"],
                }
            )
            prediction_df = pd.DataFrame(prediction_data)
            combined_df = pd.concat([prediction_df, df], axis=1)
            model_table = PandasTableModel(combined_df, full_headers, predictions)
            self.analytics_viewer.setModel(model_table)
            self.analytics_viewer.resizeColumnsToContents()
            self.highlight_selected_column()
        except Exception as e:
            print(f"Error updating predictions: {e}")
            print("Traceback:", traceback.format_exc())

    @Slot(str)
    def handle_training_error(self, error_message: str) -> None:
        """
        Handle errors encountered during model training.

        Displays a detailed error dialog and resets the training worker.
        """
        try:
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None
            if self.training_worker:
                self.training_worker.stop()
                self.training_worker = None
            self.train_button.setEnabled(True)
            self.statusBar.showMessage("Training error encountered")
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Training Error")
            error_dialog.setText("An error occurred during model training")
            detailed_text = [
                "Error Details:",
                "-------------",
                str(error_message),
                "",
                "Debug Information:",
                "-----------------",
            ]
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
            error_dialog.setMinimumWidth(400)
            error_dialog.exec()
        except Exception as e:
            print(f"Error in error handler: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                "An error occurred while handling the training error.\n"
                f"Original error: {error_message}\nHandler error: {str(e)}",
            )

    def cancel_training(self) -> None:
        """
        Cancel the ongoing model training process.
        """
        if self.training_worker:
            self.training_worker.stop()
            self.training_worker.wait()
            self.training_worker = None

        if hasattr(self, "training_timer"):
            self.training_timer.stop()

        if hasattr(self, "progress_dialog"):
            self.progress_dialog.close()

        self.train_button.setEnabled(True)
        self.statusBar.showMessage("Training cancelled")

    def closeEvent(self, event) -> None:
        """
        Handle the application shutdown by ensuring any running worker thread is stopped.
        """
        if self.training_worker:
            self.training_worker.stop()
            self.training_worker.wait()
        event.accept()

    def save_results(self, file_path: str) -> None:
        """
        Save the predictions and model parameters to the specified file.

        (Implementation placeholder.)
        """
        if hasattr(self, "predictions"):
            # Save implementation goes here.
            pass

    def select_predict_file(self) -> None:
        """
        Open a file dialog to select a dataset for running predictions.
        """
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset for Prediction",
            desktop_path,
            "CSV Files (*.csv);;All Files (*)",
        )
        if file_path:
            self.predict_file_path = file_path
            self.statusBar.showMessage(f"Selected prediction file: {file_path}")
        else:
            self.statusBar.showMessage("No file selected for prediction.")

    def update_last_directory(self) -> None:
        """
        If the "Remember current directory" checkbox is checked,
        store the current directory in QSettings.
        """
        if self.load_ui.rememberDirCheckBox.isChecked():
            settings = QSettings("UT Austin", "Arborist")
            settings.setValue("load/lastDirectory", self.current_directory)
            settings.setValue("load/rememberDir", "true")
        else:
            settings = QSettings("UT Austin", "Arborist")
            settings.setValue("load/rememberDir", "false")


def main() -> None:
    """
    Main function to start the Arborist application.
    """
    app = QApplication(sys.argv)
    main_window = Arborist()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
