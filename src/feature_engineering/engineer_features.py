import logging
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger("src.feature_engineering.engineer_features")


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train and test datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    train_path = "data/preprocessed/train_preprocessed.csv"
    test_path = "data/preprocessed/test_preprocessed.csv"
    logger.info(f"Loading preprocessed data from {train_path} and {test_path}")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def engineer_features(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Apply feature engineering transformations.

    Args:
        train_data (pd.DataFrame): Training dataset
        test_data (pd.DataFrame): Test dataset

    Returns:
        tuple containing:
            pd.DataFrame: Engineered training features
            pd.DataFrame: Engineered test features
            StandardScaler: Fitted scaler
    """
    logger.info("Engineering features...")
    feature_columns = [col for col in train_data.columns if col != "target"]

    scaler = StandardScaler()
    train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
    test_data[feature_columns] = scaler.transform(test_data[feature_columns])

    return train_data, test_data, scaler


def save_artifacts(
    train_data: pd.DataFrame, test_data: pd.DataFrame, scaler: StandardScaler
) -> None:
    """Save engineered features and scaler.

    Args:
        train_data (pd.DataFrame): Engineered training data
        test_data (pd.DataFrame): Engineered test data
        scaler (StandardScaler): Fitted scaler
    """
    # Save processed data
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving engineered features to {output_dir}")

    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path = os.path.join(output_dir, "test_processed.csv")

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    # Save scaler
    scaler_path = os.path.join("artifacts", "[features]_scaler.joblib")
    logger.info(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)


def main() -> None:
    """Main function to orchestrate feature engineering pipeline."""
    train_data, test_data = load_preprocessed_data()
    train_featured, test_featured, scaler = engineer_features(train_data, test_data)
    save_artifacts(train_featured, test_featured, scaler)
    logger.info("Feature engineering completed")


if __name__ == "__main__":
    main()
