import logging
from collections import Counter

from fastapi import APIRouter, HTTPException, Query

from core import state
from ml.history import append_record, load_history
from ml.preprocessing import prepare_data, split_data, to_dataframe
from ml.training import evaluate, predict_churn, save_churn_model, train_churn_model
from schemas import (
    DatasetInfo,
    DatasetRowChurn,
    ErrorResponse,
    FeatureVectorChurn,
    HealthStatus,
    ModelStatus,
    PredictionResponseChurn,
    SplitInfo,
    TrainingConfigChurn,
    TrainingRecord,
    TrainResult,
)

log = logging.getLogger("churn")
router = APIRouter()


def ensure_dataset() -> None:
    if len(state.DATASET) == 0:
        raise HTTPException(status_code=400, detail="dataset is empty")


@router.get("/")
def root():
    return {"message": "ml churn service is running"}


@router.get("/dataset/preview")
def dataset_preview(n: int = 10) -> list[DatasetRowChurn]:
    return state.DATASET[:n]


@router.get("/dataset/info")
def dataset_info() -> DatasetInfo:
    return {
        "rows_count": len(state.DATASET),
        "columns_count": len(DatasetRowChurn.model_fields),
        "feature_names": list(FeatureVectorChurn.model_fields),
        "churn_by_class": Counter(row.churn for row in state.DATASET),
    }


@router.get("/dataset/split-info")
def dataset_split_info() -> SplitInfo:
    ensure_dataset()
    X, y = prepare_data(to_dataframe(state.DATASET))
    X_train, X_test, y_train, y_test = split_data(X, y)

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_by_class": Counter(y_train),
        "test_churn_by_class": Counter(y_test),
    }


@router.post(
    "/model/train",
    responses={
        400: {"model": ErrorResponse, "description": "dataset is empty"},
        422: {"model": ErrorResponse, "description": "bad config or hyperparameters"},
    },
)
def model_train(config: TrainingConfigChurn | None = None) -> TrainResult:
    if config is None:
        config = TrainingConfigChurn()

    ensure_dataset()
    X, y = prepare_data(to_dataframe(state.DATASET))
    X_train, X_test, y_train, y_test = split_data(X, y)
    try:
        pipeline = train_churn_model(X_train, y_train, config)
    except (TypeError, ValueError) as e:
        raise HTTPException(422, f"bad hyperparameters: {e}")

    metrics = evaluate(pipeline, X_test, y_test)
    log.info("model trained: %s %s metrics=%s", config.model_type, config.hyperparameters, metrics)
    state.MODEL = save_churn_model(pipeline, metrics, config)
    append_record(
        TrainingRecord(
            trained_at=state.MODEL["trained_at"],
            model_type=state.MODEL["model_type"],
            hyperparameters=state.MODEL["hyperparameters"],
            metrics=metrics,
        )
    )

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }


@router.get("/model/status")
def model_status() -> ModelStatus:
    if state.MODEL is None:
        return {
            "trained": False,
            "trained_at": None,
            "metrics": None,
            "model_type": None,
            "hyperparameters": None,
        }
    return {
        "trained": True,
        "trained_at": state.MODEL["trained_at"],
        "metrics": state.MODEL["metrics"],
        "model_type": state.MODEL["model_type"],
        "hyperparameters": state.MODEL["hyperparameters"],
    }


@router.get("/model/metrics")
def model_metrics(
    limit: int = Query(1, ge=1), model_type: str | None = None
) -> list[TrainingRecord]:
    history = load_history()
    if model_type:
        history = [r for r in history if r.model_type == model_type]
    return history[-limit:]


@router.post(
    "/predict",
    responses={
        422: {"model": ErrorResponse, "description": "invalid features"},
        503: {"model": ErrorResponse, "description": "model not trained"},
    },
)
def predict(data: FeatureVectorChurn | list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
    if state.MODEL is None:
        raise HTTPException(503, "model is not trained, call POST /model/train")
    rows = data if isinstance(data, list) else [data]
    log.info("predict: %d rows", len(rows))
    return predict_churn(state.MODEL["pipeline"], rows)


@router.get("/model/schema")
def model_schema() -> dict[str, str]:
    return {
        name: field.annotation.__name__ for name, field in FeatureVectorChurn.model_fields.items()
    }


@router.get("/health")
def health() -> HealthStatus:
    return {
        "status": "ok" if state.MODEL is not None and state.DATASET else "degraded",
        "model_loaded": state.MODEL is not None,
        "dataset_loaded": len(state.DATASET) > 0,
    }
