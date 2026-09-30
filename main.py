from collections import Counter

from fastapi import FastAPI

from dataset import load_dataset
from models import DatasetInfo, DatasetRowChurn, FeatureVectorChurn, SplitInfo
from preprocessing import prepare_data, split_data, to_dataframe

app = FastAPI()


DATASET = load_dataset()


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post("/predict")
def predict(data: FeatureVectorChurn) -> FeatureVectorChurn:
    return data


@app.get("/dataset/preview")
def dataset_preview(n: int = 10) -> list[DatasetRowChurn]:
    return DATASET[:n]


@app.get("/dataset/info")
def dataset_info() -> DatasetInfo:
    return {
        "rows_count": len(DATASET),
        "columns_count": len(DatasetRowChurn.model_fields),
        "feature_names": list(FeatureVectorChurn.model_fields),
        "churn_by_class": Counter(row.churn for row in DATASET),
    }


@app.get("/dataset/split-info")
def dataset_split_info() -> SplitInfo:
    X, y = prepare_data(to_dataframe(DATASET))
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(len(X_test))

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_by_class": Counter(y_train),
        "test_churn_by_class": Counter(y_test),
    }
