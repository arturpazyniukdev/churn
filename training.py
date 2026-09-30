from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models import FeatureVectorChurn, TrainingConfigChurn
from preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, to_dataframe

MODEL_PATH = Path("output/churn_model.joblib")

ESTIMATORS = {"logreg": LogisticRegression, "random_forest": RandomForestClassifier}


def save_churn_model(
    pipeline: Pipeline,
    metrics: dict[str, float],
    config: TrainingConfigChurn,
    path: Path | None = None,
) -> dict:
    path = path or MODEL_PATH
    bundle = {
        "pipeline": pipeline,
        "trained_at": datetime.now(timezone.utc),
        "metrics": metrics,
        "model_type": config.model_type,
        "hyperparameters": config.hyperparameters,
    }
    path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def load_churn_model(path: Path | None = None) -> dict | None:
    path = path or MODEL_PATH
    if not path.exists():
        return None

    return joblib.load(path)


def _build_pipeline(config: TrainingConfigChurn) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
    estimator = ESTIMATORS[config.model_type](**config.hyperparameters)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def train_churn_model(
    X_train: pd.DataFrame, y_train: pd.Series, config: TrainingConfigChurn
) -> Pipeline:
    pipeline = _build_pipeline(config)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 2),
        "f1": round(f1_score(y_test, y_pred), 2),
        "roc_auc": round(roc_auc_score(y_test, proba), 2),
    }


def predict_churn(pipeline: Pipeline, rows: list[FeatureVectorChurn]) -> list[dict]:
    df = to_dataframe(rows)
    classes = pipeline.predict(df)
    probas = pipeline.predict_proba(df)
    return [
        {"churn": int(c), "probabilities": {0: float(p[0]), 1: float(p[1])}}
        for c, p in zip(classes, probas)
    ]
