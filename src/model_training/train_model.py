import json
import logging
import os
import sys

# =============================
# CONFIGURAÇÕES DE AMBIENTE
# =============================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ["MLFLOW_TRACKING_PRINT_RUN_URL"] = "false"
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

# =============================
# LOGGING
# =============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("src.model_training.train_model")

# =============================
# IMPORTS
# =============================

import joblib
import mlflow


mlflow.set_tracking_uri(
    "file://" + os.path.expanduser("~/mlruns")
)

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")

# =============================
# LOADERS
# =============================

def load_data() -> pd.DataFrame:
    path = "data/processed/train_processed.csv"
    logger.info(f"Loading training data from {path}")
    return pd.read_csv(path)


def load_params() -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["train"]

# =============================
# PREPARAÇÃO DOS DADOS
# =============================

def prepare_data(
    train_data: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, OneHotEncoder]:

    X = train_data.drop(columns=["target"])
    y = train_data["target"]

    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

    return X, y_encoded, encoder

# =============================
# MODELO
# =============================

def create_model(input_dim: int, num_classes: int, params: dict) -> tf.keras.Model:

    model = Sequential(
        [
            Dense(
                params["hidden_layer_1_neurons"],
                activation="relu",
                input_shape=(input_dim,),
            ),
            Dropout(params["dropout_rate"]),
            Dense(
                params["hidden_layer_2_neurons"],
                activation="relu",
            ),
            Dropout(params["dropout_rate"]),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# =============================
# SALVAR ARTEFATOS
# =============================

def save_artifacts(model: tf.keras.Model, encoder: OneHotEncoder) -> None:
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    model_path = "models/model.keras"
    encoder_path = "artifacts/target_one_hot_encoder.joblib"

    logger.info(f"Saving model to {model_path}")
    model.save(model_path)

    logger.info(f"Saving encoder to {encoder_path}")
    joblib.dump(encoder, encoder_path)

# =============================
# TREINAMENTO
# =============================

def train_model(train_data: pd.DataFrame, params: dict) -> None:

    mlflow.set_experiment("exp_ml_classification")

    with mlflow.start_run():

        # Log hiperparâmetros
        mlflow.log_params(params)

        # Seed
        tf.keras.utils.set_random_seed(params["random_seed"])

        # Log artefatos de pré-processamento
        mlflow.log_artifact("artifacts/features_mean_imputer.joblib")
        mlflow.log_artifact("artifacts/features_scaler.joblib")

        # Dados
        X_train, y_train, encoder = prepare_data(train_data)

        # Modelo
        model = create_model(
            input_dim=X_train.shape[1],
            num_classes=y_train.shape[1],
            params=params,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        logger.info("Training model...")
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[early_stopping],
            verbose=0,
        )

        # Salvar artefatos
        save_artifacts(model, encoder)

        mlflow.log_artifact("models/model.keras")
        mlflow.log_artifact("artifacts/target_one_hot_encoder.joblib")

        # Métricas finais
        final_metrics = {
            k: float(v[-1]) for k, v in history.history.items()
        }

        for k, v in final_metrics.items():
            mlflow.log_metric(k, v)

        # Métricas para o DVC
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/training.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

# =============================
# MAIN
# =============================

def main() -> None:
    train_data = load_data()
    params = load_params()
    train_model(train_data, params)
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()
