import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class DataIngestionConfig:
    """Configuration class for data ingestion."""

    raw_data_path: str
    ingested_data_dir: str
    train_data_path: str
    test_data_path: str
    test_size: float = 0.2
    random_state: int = 42


class DataIngestion:
    """Class for handling data ingestion operations."""

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize DataIngestion with configuration.

        Args:
            config: DataIngestionConfig instance containing data paths and parameters
        """
        self.config = config
        self.raw_data = None
        self.train_data = None
        self.test_data = None

        # Create ingested data directory if it doesn't exist
        Path(self.config.ingested_data_dir).mkdir(parents=True, exist_ok=True)

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Load raw data and split into train and test sets.

        Returns:
            Tuple containing paths to train and test data CSV files
        """
        try:
            # Load raw data
            self.raw_data = pd.read_csv(self.config.raw_data_path)
            print(f"Raw data loaded successfully. Shape: {self.raw_data.shape}")

            # Split data into train and test sets
            self.train_data = self.raw_data.sample(
                frac=1 - self.config.test_size,
                random_state=self.config.random_state
            )
            self.test_data = self.raw_data.drop(self.train_data.index)

            # Save train and test data
            self.train_data.to_csv(self.config.train_data_path, index=False)
            self.test_data.to_csv(self.config.test_data_path, index=False)

            print(f"Train data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
            print(f"Train data saved to: {self.config.train_data_path}")
            print(f"Test data saved to: {self.config.test_data_path}")

            return self.config.train_data_path, self.config.test_data_path

        except FileNotFoundError as e:
            print(f"Error: Data file not found - {e}")
            raise
        except Exception as e:
            print(f"Error during data ingestion: {e}")
            raise

if __name__=='__main__':
    config = DataIngestionConfig(
        raw_data_path='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/raw.csv',
        ingested_data_dir='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/ingested',
        train_data_path='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/ingested/train.csv',
        test_data_path='/Users/fireflylaptops/Desktop/end-to-end-ml-project/artifacts/ingested/test.csv'
    )
    dt = DataIngestion(config)
    print(dt.initiate_data_ingestion())