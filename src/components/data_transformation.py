import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib


@dataclass
class DataTransformationConfig:
    """Configuration class for data transformation."""

    preprocessor_obj_file_path: str
    categorical_columns: list
    numerical_columns: list
    target_column: str


class DataTransformation:
    """Class for handling data transformation operations."""

    def __init__(self, config: DataTransformationConfig):
        """
        Initialize DataTransformation with configuration.

        Args:
            config: DataTransformationConfig instance containing transformation parameters
        """
        self.config = config
        self.preprocessor = None

        # Create directory for preprocessor object if it doesn't exist
        Path(self.config.preprocessor_obj_file_path).parent.mkdir(
            parents=True, exist_ok=True
        )

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Create a preprocessing pipeline with OneHotEncoder and StandardScaler.

        Returns:
            ColumnTransformer object with categorical and numerical transformation pipelines
        """
        # Pipeline for categorical columns
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Pipeline for numerical columns
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Combine both pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.config.categorical_columns),
                ('num', numerical_transformer, self.config.numerical_columns)
            ]
        )

        return preprocessor

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Load, transform, and save train and test data.

        Args:
            train_path: Path to training data CSV file
            test_path: Path to test data CSV file

        Returns:
            Tuple containing transformed train data, test data, and preprocessor path
        """
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            print(f"Train data loaded. Shape: {train_df.shape}")
            print(f"Test data loaded. Shape: {test_df.shape}")

            # Separate features and target
            X_train = train_df.drop(columns=[self.config.target_column])
            y_train = train_df[self.config.target_column]

            X_test = test_df.drop(columns=[self.config.target_column])
            y_test = test_df[self.config.target_column]

            print(f"Target variable separated from features")

            # Create and fit preprocessor on training data
            self.preprocessor = self.get_data_transformer_object()
            X_train_transformed = self.preprocessor.fit_transform(X_train)
            X_test_transformed = self.preprocessor.transform(X_test)

            print(f"Data transformation completed")
            print(f"Transformed train data shape: {X_train_transformed.shape}")
            print(f"Transformed test data shape: {X_test_transformed.shape}")

            # Save preprocessor object
            joblib.dump(self.preprocessor, self.config.preprocessor_obj_file_path)
            print(f"Preprocessor saved to: {self.config.preprocessor_obj_file_path}")

            return X_train_transformed, X_test_transformed, self.config.preprocessor_obj_file_path

        except FileNotFoundError as e:
            print(f"Error: Data file not found - {e}")
            raise
        except KeyError as e:
            print(f"Error: Column not found - {e}")
            raise
        except Exception as e:
            print(f"Error during data transformation: {e}")
            raise
