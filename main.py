from collections import Counter

from fastapi import FastAPI, HTTPException

from dataset import load_dataset
from models import (
    DatasetInfo,
    DatasetRowChurn,
    FeatureVectorChurn,
    ModelStatus,
    PredictionResponseChurn,
    SplitInfo,
    TrainingConfigChurn,
    TrainResult,
)
from preprocessing import prepare_data, split_data, to_dataframe
from training import evaluate, load_churn_model, predict_churn, save_churn_model, train_churn_model

app = FastAPI()


DATASET = load_dataset()
MODEL = load_churn_model()  # dict | None


def ensure_dataset() -> None:
    if len(DATASET) == 0:
        raise HTTPException(status_code=400, detail="dataset is empty")


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


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
    ensure_dataset()
    X, y = prepare_data(to_dataframe(DATASET))
    X_train, X_test, y_train, y_test = split_data(X, y)

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_by_class": Counter(y_train),
        "test_churn_by_class": Counter(y_test),
    }


@app.post("/model/train")
def model_train(config: TrainingConfigChurn | None = None) -> TrainResult:
    global MODEL

    if config is None:
        config = TrainingConfigChurn()

    ensure_dataset()
    X, y = prepare_data(to_dataframe(DATASET))
    X_train, X_test, y_train, y_test = split_data(X, y)
    try:
        pipeline = train_churn_model(X_train, y_train, config)
    except (TypeError, ValueError) as e:
        raise HTTPException(422, f"bad hyperparameters: {e}")

    metrics = evaluate(pipeline, X_test, y_test)
    MODEL = save_churn_model(pipeline, metrics, config)

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
    }


@app.get("/model/status")
def model_status() -> ModelStatus:
    if MODEL is None:
        return {
            "trained": False,
            "trained_at": None,
            "metrics": None,
            "model_type": None,
            "hyperparameters": None,
        }
    return {
        "trained": True,
        "trained_at": MODEL["trained_at"],
        "metrics": MODEL["metrics"],
        "model_type": MODEL["model_type"],
        "hyperparameters": MODEL["hyperparameters"],
    }


@app.post("/predict")
def predict(data: FeatureVectorChurn | list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
    if MODEL is None:
        raise HTTPException(503, "model is not trained, call POST /model/train")
    rows = data if isinstance(data, list) else [data]
    return predict_churn(MODEL["pipeline"], rows)
