import pytest
from fastapi.testclient import TestClient

import history
import main
import training
from tests.test_preprocessing import make_rows


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(history, "HISTORY_PATH", tmp_path / "h.json")
    monkeypatch.setattr(main, "DATASET", make_rows(40))
    monkeypatch.setattr(main, "MODEL", None)
    return TestClient(main.app)


def test_dataset_csv_loads():
    from dataset import load_dataset

    rows = load_dataset()
    assert len(rows) > 0
    assert rows[0].churn in (0, 1)


def test_train_status_predict(client):
    r = client.post("/model/train")
    assert r.status_code == 200
    assert set(r.json()) >= {"accuracy", "f1", "roc_auc"}

    r = client.get("/model/status")
    assert r.status_code == 200
    assert r.json()["trained"] is True
    assert r.json()["model_type"] == "logreg"

    features = make_rows(1)[0].model_dump(exclude={"churn"})
    r = client.post("/predict", json=[features])
    assert r.status_code == 200
    assert r.json()[0]["churn"] in (0, 1)


def test_predict_without_model(client):
    features = make_rows(1)[0].model_dump(exclude={"churn"})
    r = client.post("/predict", json=[features])
    assert r.status_code == 503
    assert r.json()["code"] == "http_error"


def test_predict_bad_body(client):
    r = client.post("/predict", json={"monthly_fee": "abc"})
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"
    assert r.json()["details"]


def test_train_empty_dataset(client, monkeypatch):
    monkeypatch.setattr(main, "DATASET", [])
    r = client.post("/model/train")
    assert r.status_code == 400
    assert r.json()["message"] == "dataset is empty"
