import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


@dataclass
class ModelTrainerConfig:
    """Configuration class for model training."""

    model_dir: str
    best_model_path: str
    random_state: int = 42


class ModelTrainer:
    """Class for training and evaluating multiple ML models."""

    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize ModelTrainer with configuration.

        Args:
            config: ModelTrainerConfig instance containing training parameters
        """
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = -float('inf')
        self.best_model_name = None

        # Create model directory if it doesn't exist
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

    def get_model_params(self) -> Dict[str, Tuple]:
        """
        Define models with their hyperparameter grids for tuning.

        Returns:
            Dictionary containing models and their hyperparameter grids
        """
        models_params = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=self.config.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'DecisionTree': {
                'model': DecisionTreeRegressor(random_state=self.config.random_state),
                'params': {
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['squared_error', 'absolute_error']
                }
            },
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False]
                }
            },
            'KNN': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                    'p': [1, 2]
                }
            }
        }

        return models_params

    def train_models(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Train multiple models with hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary containing training results for all models
        """
        try:
            models_params = self.get_model_params()
            results = {}

            for model_name, config in models_params.items():
                print(f"\n{'='*60}")
                print(f"Training {model_name}...")
                print(f"{'='*60}")

                model = config['model']
                params = config['params']

                # GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=5,
                    scoring='r2',
                    n_jobs=-1,
                    verbose=1
                )

                # Fit the model
                grid_search.fit(X_train, y_train)

                # Get best model
                best_model = grid_search.best_estimator_

                # Make predictions
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Calculate metrics
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)

                results[model_name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae
                }

                print(f"\n{model_name} Results:")
                print(f"  Best Parameters: {grid_search.best_params_}")
                print(f"  Train R² Score: {train_r2:.4f}")
                print(f"  Test R² Score: {test_r2:.4f}")
                print(f"  Train RMSE: {train_rmse:.4f}")
                print(f"  Test RMSE: {test_rmse:.4f}")
                print(f"  Train MAE: {train_mae:.4f}")
                print(f"  Test MAE: {test_mae:.4f}")

                # Track best model
                if test_r2 > self.best_score:
                    self.best_score = test_r2
                    self.best_model = best_model
                    self.best_model_name = model_name

            return results

        except Exception as e:
            print(f"Error during model training: {e}")
            raise

    def initiate_model_training(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray
    ) -> str:
        """
        Train all models and save the best one.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            Path to the saved best model
        """
        # Train all models
        results = self.train_models(X_train, y_train, X_test, y_test)

        # Print summary
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")

        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  Test R² Score: {result['test_r2']:.4f}")
            print(f"  Test RMSE: {result['test_rmse']:.4f}")
            print(f"  Test MAE: {result['test_mae']:.4f}")

        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name} (Test R² Score: {self.best_score:.4f})")
        print(f"{'='*60}\n")

        # Save best model
        joblib.dump(self.best_model, self.config.best_model_path)
        print(f"Best model saved to: {self.config.best_model_path}")

        return self.config.best_model_path
