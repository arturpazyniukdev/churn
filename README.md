# Churn prediction service

FastAPI service that trains a customer churn classifier on `data/churn_dataset.csv`
and serves predictions. Built step by step as a learning project.

## Dataset

`data/churn_dataset.csv`, one row per customer, 2000 rows.

| column | type | meaning |
|---|---|---|
| monthly_fee | float | subscription price |
| usage_hours | float | hours of usage per month |
| support_requests | int | tickets opened |
| account_age_months | int | months since signup |
| failed_payments | int | failed charges |
| region | str | africa, america, asia, europe |
| device_type | str | desktop, mobile, tablet |
| payment_method | str | card, crypto, paypal |
| autopay_enabled | int | 0 or 1 |
| churn | int | target: 1 = customer left |

## Project layout

```
main.py          app, logging, error handlers
api/routes.py    endpoints
schemas.py       pydantic request / response models
ml/              dataset loading, preprocessing, training, history
core/config.py   file paths
core/state.py    in-memory dataset and model
tests/           pytest
output/          runtime files (model, history), gitignored
```

## Run locally

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --reload
```

Docs: http://127.0.0.1:8000/docs

## Run in Docker

```sh
docker build -t churn-service .
docker run --rm -p 8000:8000 churn-service
```

To keep the trained model between runs: `-v "$(pwd)/output:/app/output"`.

## Endpoints

| method | path | purpose |
|---|---|---|
| GET | /health | model / dataset loaded |
| GET | /dataset/info | row count, features, class balance |
| GET | /dataset/preview?n=10 | first rows |
| GET | /dataset/split-info | train / test sizes |
| POST | /model/train | train, save, record metrics |
| GET | /model/status | current model info |
| GET | /model/metrics?limit=5&model_type=logreg | training history |
| GET | /model/schema | expected features and types |
| POST | /predict | churn prediction for one or many customers |

## Examples

Train with defaults (logistic regression):

```sh
curl -X POST http://127.0.0.1:8000/model/train
```

Train a random forest with hyperparameters:

```sh
curl -X POST http://127.0.0.1:8000/model/train \
  -H 'Content-Type: application/json' \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 200, "class_weight": "balanced", "random_state": 42}}'
```

Response:

```json
{"train_size": 1600, "test_size": 400, "accuracy": 0.79, "f1": 0.17, "roc_auc": 0.59}
```

Predict for one customer (a list of customers also works):

```sh
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"monthly_fee": 9.99, "usage_hours": 2, "support_requests": 5, "account_age_months": 1, "failed_payments": 3, "region": "asia", "device_type": "mobile", "payment_method": "crypto", "autopay_enabled": 0}'
```

Response:

```json
[{"churn": 1, "probabilities": {"0": 0.19, "1": 0.81}}]
```

Errors come back as JSON with `code`, `message`, `details`:

```json
{"code": "http_error", "message": "model is not trained, call POST /model/train", "details": null}
```

## Tests

```sh
.venv/bin/pytest -q
```

Tests use synthetic rows and temp files, nothing in `output/` is touched.
