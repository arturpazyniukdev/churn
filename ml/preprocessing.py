import pandas as pd
from sklearn.model_selection import train_test_split

from schemas import DatasetRowChurn

NUMERIC_COLUMNS = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_COLUMNS = ["region", "device_type", "payment_method"]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

TARGET = "churn"


def to_dataframe(rows: list[DatasetRowChurn]) -> pd.DataFrame:
    df = pd.DataFrame([row.model_dump() for row in rows])
    return df[[c for c in FEATURE_COLUMNS + [TARGET] if c in df.columns]]


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.dropna()
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return (X, y)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
