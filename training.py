from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

MODEL_PATH = Path("output/churn_model.joblib")


def save_churn_model(
    pipeline: Pipeline, metrics: dict[str, float], path: Path = MODEL_PATH
) -> dict:
    bundle = {"pipeline": pipeline, "trained_at": datetime.now(timezone.utc), "metrics": metrics}
    path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def load_churn_model(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None

    return joblib.load(path)


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression())])


def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 2),
        "f1": round(f1_score(y_test, y_pred), 2),
    }
