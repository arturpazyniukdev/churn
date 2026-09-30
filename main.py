from collections import Counter

from fastapi import FastAPI

from dataset import load_dataset
from models import DatasetInfo, DatasetRowChurn, FeatureVectorChurn

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
