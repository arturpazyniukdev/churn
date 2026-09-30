from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "monthly_fee": 9.99,
                    "usage_hours": 2.0,
                    "support_requests": 5,
                    "account_age_months": 1,
                    "failed_payments": 3,
                    "region": "asia",
                    "device_type": "mobile",
                    "payment_method": "crypto",
                    "autopay_enabled": 0,
                }
            ]
        }
    )
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class DatasetInfo(BaseModel):
    rows_count: int
    columns_count: int
    feature_names: list[str]
    churn_by_class: dict[int, int]


class SplitInfo(BaseModel):
    train_size: int
    test_size: int
    train_churn_by_class: dict[int, int]
    test_churn_by_class: dict[int, int]


class TrainResult(BaseModel):
    train_size: int
    test_size: int
    accuracy: float
    f1: float


class ModelStatus(BaseModel):
    trained: bool
    trained_at: datetime | None
    metrics: dict[str, float] | None


class PredictionResponseChurn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"churn": 1, "probabilities": {0: 0.19, 1: 0.81}}]}
    )
    churn: int
    probabilities: dict[int, float]
