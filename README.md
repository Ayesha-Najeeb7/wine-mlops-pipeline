# Wine MLOps Pipeline

![CI](https://github.com/Ayesha-Najeeb7/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

Reproducible MLOps pipeline for `sklearn.datasets.load_wine` (178 samples, 13 features, 3 classes):
Makefile automation, MLflow tracking + Model Registry, and a GitHub Actions quality gate.
All randomness uses seed **42**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install     # upgrade pip + install pinned requirements
make lint        # flake8, max line length 100
make test        # pytest -v (unit tests + model quality gate)
make train       # 6 MLflow runs, register best as WineClassifier@champion
make evaluate    # load champion, report test-split metrics
make clean       # remove bytecode / caches
```

View experiments: `mlflow ui --backend-store-uri sqlite:///mlflow.db` then open http://127.0.0.1:5000

## Structure

```
src/data.py       load, validate (no nulls, 13 features), stratified 80/20 split
src/train.py      RandomForest + GradientBoosting grids, 5-fold CV, MLflow logging, registry
src/evaluate.py   load models:/WineClassifier@champion and score on test split
tests/            data tests + model quality gate
.github/workflows/ci.yml   install -> lint -> test on PR/push to main
```

## Quality gate (`tests/test_model_gate.py`)

- Validation macro F1 >= 0.88
- Batch inference latency <= 30 ms
- Predictions are class indices in {0, 1, 2}

## Branching

`main` stays clean; work happens on `feature/*` branches merged via pull request.

MLflow runs are tracked in a local SQLite store (mlflow.db).
