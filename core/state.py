import logging

from ml.dataset import load_dataset
from ml.training import load_churn_model
from schemas import DatasetRowChurn

log = logging.getLogger("churn")

DATASET: list[DatasetRowChurn] = []
MODEL: dict | None = None


def load() -> None:
    global DATASET, MODEL
    DATASET = load_dataset()
    log.info("dataset loaded: %d rows", len(DATASET))
    MODEL = load_churn_model()
