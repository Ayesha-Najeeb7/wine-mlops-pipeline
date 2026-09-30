"""MLOps quality gate: blocks degrading models from reaching main."""
import os
import time

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from src.data import get_train_test_split
from src.evaluate import load_champion
from src.train import CHAMPION_ALIAS, MODEL_NAME, run_training

MIN_VAL_F1 = 0.88
MAX_LATENCY_MS = 30.0
VALID_CLASSES = {0, 1, 2}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Train + register in an isolated MLflow store so the gate is self-contained."""
    workdir = tmp_path_factory.mktemp("mlflow_gate")
    previous = os.getcwd()
    os.chdir(workdir)
    uri = f"sqlite:///{workdir / 'mlflow.db'}"
    try:
        champion = run_training(tracking_uri=uri)
        model = load_champion(uri)
        _, X_test, _, _ = get_train_test_split()
        yield {"champion": champion, "model": model, "X_test": X_test, "uri": uri}
    finally:
        os.chdir(previous)


def test_champion_alias_registered(pipeline):
    version = MlflowClient(pipeline["uri"]).get_model_version_by_alias(
        MODEL_NAME, CHAMPION_ALIAS
    )
    assert version.version == pipeline["champion"]["registered_version"]


def test_metric_threshold_gate(pipeline):
    f1 = pipeline["champion"]["val_f1_macro"]
    assert f1 >= MIN_VAL_F1, f"Validation macro F1 {f1:.4f} < {MIN_VAL_F1}"


def test_inference_latency_gate(pipeline):
    model, X_test = pipeline["model"], pipeline["X_test"]
    model.predict(X_test)  # warm-up
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        model.predict(X_test)
        timings.append((time.perf_counter() - start) * 1000)
    median_ms = float(np.median(timings))
    assert median_ms <= MAX_LATENCY_MS, f"Batch latency {median_ms:.2f} ms > {MAX_LATENCY_MS} ms"


def test_output_schema_gate(pipeline):
    preds = pipeline["model"].predict(pipeline["X_test"])
    assert len(preds) == len(pipeline["X_test"])
    assert np.issubdtype(np.asarray(preds).dtype, np.integer)
    assert set(np.unique(preds)) <= VALID_CLASSES
