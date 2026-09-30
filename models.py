from pydantic import BaseModel


class FeatureVectorChurn(BaseModel):
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
