from ml.preprocessing import FEATURE_COLUMNS, TARGET, prepare_data, split_data, to_dataframe
from schemas import DatasetRowChurn


def make_rows(n: int) -> list[DatasetRowChurn]:
    return [
        DatasetRowChurn(
            monthly_fee=10.0 + i,
            usage_hours=1.0 * i,
            support_requests=i,
            account_age_months=i,
            failed_payments=i % 3,
            region="asia",
            device_type="mobile",
            payment_method="card",
            autopay_enabled=i % 2,
            churn=i % 2,
        )
        for i in range(n)
    ]


def test_to_dataframe_columns():
    df = to_dataframe(make_rows(4))
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET]
    assert len(df) == 4


def test_prepare_data_separates_target():
    X, y = prepare_data(to_dataframe(make_rows(4)))
    assert TARGET not in X.columns
    assert list(y) == [0, 1, 0, 1]


def test_split_data_is_reproducible():
    X, y = prepare_data(to_dataframe(make_rows(10)))
    a = split_data(X, y, random_state=1)
    b = split_data(X, y, random_state=1)
    assert list(a[0].index) == list(b[0].index)
    assert len(a[0]) + len(a[1]) == 10
