import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


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
    return {"accuracy": accuracy_score(y_test, y_pred), "f1": f1_score(y_test, y_pred)}
