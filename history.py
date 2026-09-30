import json
from pathlib import Path

from models import TrainingRecord

HISTORY_PATH = Path("output/training_history.json")


def load_history(path=HISTORY_PATH) -> list[TrainingRecord]:
    if not path.exists():
        return []
    return [TrainingRecord(**d) for d in json.loads(path.read_text())]


def append_record(record: TrainingRecord, path: Path = HISTORY_PATH) -> None:
    records = load_history(path) + [record]
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps([r.model_dump(mode="json") for r in records], indent=2))
