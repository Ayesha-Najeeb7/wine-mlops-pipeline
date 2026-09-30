"""Inference check: load the registered champion and score it on the test split."""
import argparse
from typing import Dict

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import SEED, get_train_test_split
from src.train import CHAMPION_ALIAS, DEFAULT_TRACKING_URI, MODEL_NAME


def load_champion(tracking_uri: str = DEFAULT_TRACKING_URI):
    """Load models:/WineClassifier@champion from the MLflow Model Registry."""
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")


def evaluate_champion(tracking_uri: str = DEFAULT_TRACKING_URI, seed: int = SEED) -> Dict:
    """Compute final test-split metrics for the champion model."""
    model = load_champion(tracking_uri)
    _, X_test, _, y_test = get_train_test_split(seed=seed)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    return {
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_f1_macro": float(f1_score(y_test, preds, average="macro")),
        "test_log_loss": float(log_loss(y_test, proba, labels=[0, 1, 2])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the champion model on the test split.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    metrics = evaluate_champion(args.tracking_uri, args.seed)
    print(f"Champion: {MODEL_NAME}@{CHAMPION_ALIAS}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
