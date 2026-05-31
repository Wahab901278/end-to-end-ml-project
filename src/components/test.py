import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.data_ingestion import DataIngestion, DataIngestionConfig
from components.data_transformation import DataTransformation, DataTransformationConfig
from components.model_trainer import ModelTrainer, ModelTrainerConfig


def main():
    """Test the complete ML workflow: ingestion -> transformation -> training."""

    print("\n" + "="*80)
    print("STARTING END-TO-END ML WORKFLOW TEST")
    print("="*80 + "\n")

    # ==================== STEP 1: DATA INGESTION ====================
    print("\n" + "▶" * 40)
    print("STEP 1: DATA INGESTION")
    print("▶" * 40 + "\n")

    ingestion_config = DataIngestionConfig(
        raw_data_path='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/raw.csv',
        ingested_data_dir='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/ingested',
        train_data_path='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/ingested/train.csv',
        test_data_path='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/ingested/test.csv',
        test_size=0.2
    )

    ingestion = DataIngestion(ingestion_config)
    train_path, test_path = ingestion.initiate_data_ingestion()

    print(f"✓ Data ingestion completed successfully!")
    print(f"  Train file: {train_path}")
    print(f"  Test file: {test_path}\n")

    # ==================== STEP 2: DATA TRANSFORMATION ====================
    print("\n" + "▶" * 40)
    print("STEP 2: DATA TRANSFORMATION")
    print("▶" * 40 + "\n")

    transformation_config = DataTransformationConfig(
        preprocessor_obj_file_path='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/preprocessor.pkl',
        categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'],
        numerical_columns=['reading_score', 'writing_score'],
        target_column='math_score'
    )

    transformation = DataTransformation(transformation_config)
    X_train_transformed, X_test_transformed, preprocessor_path = transformation.initiate_data_transformation(
        train_path, test_path
    )

    # Load target variables
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_train = train_df['math_score'].values
    y_test = test_df['math_score'].values

    print(f"✓ Data transformation completed successfully!")
    print(f"  Preprocessor saved to: {preprocessor_path}")
    print(f"  Training features shape: {X_train_transformed.shape}")
    print(f"  Test features shape: {X_test_transformed.shape}\n")

    # ==================== STEP 3: MODEL TRAINING ====================
    print("\n" + "▶" * 40)
    print("STEP 3: MODEL TRAINING & HYPERPARAMETER TUNING")
    print("▶" * 40 + "\n")

    trainer_config = ModelTrainerConfig(
        model_dir='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/models',
        best_model_path='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/models/best_model.pkl'
    )

    trainer = ModelTrainer(trainer_config)
    best_model_path = trainer.initiate_model_training(
        X_train_transformed, y_train,
        X_test_transformed, y_test
    )

    print(f"✓ Model training completed successfully!")
    print(f"  Best model saved to: {best_model_path}")
    print(f"  Best model type: {trainer.best_model_name}")
    print(f"  Best test R² score: {trainer.best_score:.4f}\n")

    # ==================== STEP 4: FINAL SUMMARY ====================
    print("\n" + "="*80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"""
Pipeline Summary:
  • Target Variable: math_score
  • Training Samples: {len(y_train)}
  • Test Samples: {len(y_test)}

Data Transformation:
  • Categorical Features: {len(transformation_config.categorical_columns)}
  • Numerical Features: {len(transformation_config.numerical_columns)}
  • Total Features after encoding: {X_train_transformed.shape[1]}

Model Training:
  • Models tested: Random Forest, Decision Tree, Linear Regression, KNN
  • Best Model: {trainer.best_model_name}
  • Test R² Score: {trainer.best_score:.4f}

Artifacts Location: {trainer_config.model_dir}
    """)
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
