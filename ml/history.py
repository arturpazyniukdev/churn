import json
from pathlib import Path

from core.config import HISTORY_PATH
from schemas import TrainingRecord


def load_history(path: Path | None = None) -> list[TrainingRecord]:
    path = path or HISTORY_PATH
    if not path.exists():
        return []
    return [TrainingRecord(**d) for d in json.loads(path.read_text())]


def append_record(record: TrainingRecord, path: Path | None = None) -> None:
    path = path or HISTORY_PATH
    records = load_history(path) + [record]
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps([r.model_dump(mode="json") for r in records], indent=2))
