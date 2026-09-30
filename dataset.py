import csv
from pathlib import Path

from models import DatasetRowChurn

DATA_PATH = Path("data/churn_dataset.csv")


def load_dataset(path: Path = DATA_PATH) -> list[DatasetRowChurn]:
    if not path.exists():
        return []

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [DatasetRowChurn(**row) for row in reader]
