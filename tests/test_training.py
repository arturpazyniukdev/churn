from ml.preprocessing import prepare_data, to_dataframe
from ml.training import (
    evaluate,
    load_churn_model,
    predict_churn,
    save_churn_model,
    train_churn_model,
)
from schemas import FeatureVectorChurn, TrainingConfigChurn
from tests.test_preprocessing import make_rows


def _trained():
    X, y = prepare_data(to_dataframe(make_rows(20)))
    config = TrainingConfigChurn()
    return X, y, config, train_churn_model(X, y, config)


def test_train_and_evaluate():
    X, y, _, pipeline = _trained()
    metrics = evaluate(pipeline, X, y)
    assert set(metrics) == {"accuracy", "f1", "roc_auc"}
    assert 0 <= metrics["accuracy"] <= 1


def test_predict_churn_shape():
    _, _, _, pipeline = _trained()
    rows = [FeatureVectorChurn(**r.model_dump(exclude={"churn"})) for r in make_rows(2)]
    result = predict_churn(pipeline, rows)
    assert len(result) == 2
    for r in result:
        assert r["churn"] in (0, 1)
        assert abs(sum(r["probabilities"].values()) - 1) < 1e-6


def test_save_and_load_roundtrip(tmp_path):
    X, y, config, pipeline = _trained()
    path = tmp_path / "m.joblib"
    save_churn_model(pipeline, evaluate(pipeline, X, y), config, path)
    bundle = load_churn_model(path)
    assert bundle["model_type"] == "logreg"
    assert len(bundle["pipeline"].predict(X)) == 20
