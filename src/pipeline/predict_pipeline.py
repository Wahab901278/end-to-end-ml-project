import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union


class CustomData:
    """Class to handle custom input data and convert it to DataFrame."""

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        """
        Initialize CustomData with student features.

        Args:
            gender: Student's gender
            race_ethnicity: Student's race/ethnicity
            parental_level_of_education: Parent's education level
            lunch: Type of lunch student receives
            test_preparation_course: Test preparation course completion status
            reading_score: Reading score
            writing_score: Writing score
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert input data to DataFrame.

        Returns:
            DataFrame with single row containing all features
        """
        data_dict = {
            'gender': [self.gender],
            'race_ethnicity': [self.race_ethnicity],
            'parental_level_of_education': [self.parental_level_of_education],
            'lunch': [self.lunch],
            'test_preparation_course': [self.test_preparation_course],
            'reading_score': [self.reading_score],
            'writing_score': [self.writing_score]
        }

        return pd.DataFrame(data_dict)


class PredictPipeline:
    """Class to load trained model and make predictions."""

    def __init__(
        self,
        model_path: str = '/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/models/best_model.pkl',
        preprocessor_path: str = '/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/preprocessor.pkl'
    ):
        """
        Initialize PredictPipeline with model and preprocessor paths.

        Args:
            model_path: Path to the trained model file
            preprocessor_path: Path to the preprocessor/transformer file
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model and preprocessor from disk."""
        try:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            if not Path(self.preprocessor_path).exists():
                raise FileNotFoundError(f"Preprocessor file not found at {self.preprocessor_path}")

            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)

            print(f"✓ Model loaded from: {self.model_path}")
            print(f"✓ Preprocessor loaded from: {self.preprocessor_path}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            raise

    def predict(self, data: Union[CustomData, pd.DataFrame]) -> Union[float, np.ndarray]:
        """
        Make prediction on input data.

        Args:
            data: Either CustomData object or DataFrame with features

        Returns:
            Predicted math_score value(s)
        """
        try:
            # Convert CustomData to DataFrame if needed
            if isinstance(data, CustomData):
                df = data.get_data_as_dataframe()
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise TypeError("Data must be CustomData or pandas DataFrame")

            # Transform data using preprocessor
            data_transformed = self.preprocessor.transform(df)

            # Make prediction
            prediction = self.model.predict(data_transformed)

            # Return scalar if single prediction, array if multiple
            if len(prediction) == 1:
                return float(prediction[0])
            else:
                return prediction

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def predict_with_confidence(self, data: Union[CustomData, pd.DataFrame]) -> dict:
        """
        Make prediction and provide additional information.

        Args:
            data: Either CustomData object or DataFrame with features

        Returns:
            Dictionary containing prediction and metadata
        """
        try:
            prediction = self.predict(data)

            result = {
                'predicted_math_score': prediction,
                'model_type': type(self.model).__name__,
                'status': 'success'
            }

            return result

        except Exception as e:
            print(f"Error: {e}")
            return {
                'predicted_math_score': None,
                'model_type': type(self.model).__name__,
                'status': 'error',
                'error_message': str(e)
            }


if __name__ == '__main__':
    # Example usage
    print("\n" + "="*80)
    print("PREDICTION PIPELINE EXAMPLE")
    print("="*80 + "\n")

    # Create sample data
    sample_data = CustomData(
        gender='female',
        race_ethnicity='group B',
        parental_level_of_education="bachelor's degree",
        lunch='standard',
        test_preparation_course='completed',
        reading_score=85,
        writing_score=88
    )

    print("Sample Student Data:")
    print(sample_data.get_data_as_dataframe())
    print()

    # Initialize pipeline and make prediction
    pipeline = PredictPipeline()
    print("\nMaking Prediction...\n")

    prediction = pipeline.predict(sample_data)
    print(f"Predicted Math Score: {prediction:.2f}\n")

    # Alternative: Get prediction with confidence info
    result = pipeline.predict_with_confidence(sample_data)
    print("Prediction Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\n" + "="*80 + "\n")
