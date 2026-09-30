"""Hyperparameter search for two tree-based classifiers, tracked with MLflow."""
import argparse
from typing import Dict, List

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import SEED, get_train_test_split

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
CV_FOLDS = 10

# Model Family A
RF_GRID = [
    {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
    {"n_estimators": 100, "max_depth": None, "min_samples_split": 4},
]

# Model Family B
GBM_GRID = [
    {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
]

MODEL_FAMILIES = {
    "RandomForest": (RandomForestClassifier, RF_GRID),
    "GradientBoosting": (GradientBoostingClassifier, GBM_GRID),
}

SCORING = {"f1_macro": "f1_macro", "accuracy": "accuracy", "log_loss": "neg_log_loss"}


def build_model(family: str, params: Dict, seed: int = SEED):
    """Instantiate a classifier of the given family with a fixed random_state."""
    model_cls, _ = MODEL_FAMILIES[family]
    return model_cls(random_state=seed, **params)


def cross_validate_model(model, X_train, y_train, seed: int = SEED) -> Dict[str, float]:
    """5-fold stratified CV on the training split; returns mean train/val metrics."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    scores = cross_validate(
        model, X_train, y_train, cv=cv, scoring=SCORING, return_train_score=True
    )
    metrics = {}
    for split in ("train", "val"):
        key = "train" if split == "train" else "test"
        metrics[f"{split}_f1_macro"] = float(scores[f"{key}_f1_macro"].mean())
        metrics[f"{split}_accuracy"] = float(scores[f"{key}_accuracy"].mean())
        metrics[f"{split}_log_loss"] = float(-scores[f"{key}_log_loss"].mean())
    return metrics


def run_experiments(tracking_uri: str = DEFAULT_TRACKING_URI, seed: int = SEED) -> List[Dict]:
    """Train every configuration of both families, one MLflow run each."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, _, y_train, _ = get_train_test_split(seed=seed)

    results = []
    for family, (_, grid) in MODEL_FAMILIES.items():
        for idx, params in enumerate(grid, start=1):
            model = build_model(family, params, seed)
            with mlflow.start_run(run_name=f"{family}-cfg{idx}") as run:
                mlflow.set_tags(
                    {"model_family": family, "config_id": str(idx), "dataset": "sklearn-wine"}
                )
                mlflow.log_params(
                    {"model_family": family, "seed": seed, "cv_folds": CV_FOLDS, **params}
                )
                metrics = cross_validate_model(model, X_train, y_train, seed)
                mlflow.log_metrics(metrics)

                model.fit(X_train, y_train)
                signature = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_train.head(5),
                )
                results.append(
                    {
                        "run_id": run.info.run_id,
                        "run_name": f"{family}-cfg{idx}",
                        "model_family": family,
                        **{k: (str(v) if v is None else v) for k, v in params.items()},
                        **metrics,
                    }
                )
    return results


def register_champion(results: List[Dict], tracking_uri: str = DEFAULT_TRACKING_URI) -> Dict:
    """Register the run with the best validation macro F1 and alias it 'champion'."""
    mlflow.set_tracking_uri(tracking_uri)
    best = max(results, key=lambda r: r["val_f1_macro"])
    version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version.version)
    return {**best, "registered_version": version.version}


def run_training(tracking_uri: str = DEFAULT_TRACKING_URI, seed: int = SEED) -> Dict:
    """Full pipeline: experiments -> champion registration. Returns champion info."""
    results = run_experiments(tracking_uri, seed)
    champion = register_champion(results, tracking_uri)
    champion["all_results"] = results
    return champion


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and register the Wine classifier.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    champion = run_training(args.tracking_uri, args.seed)
    table = pd.DataFrame(champion["all_results"]).drop(columns=["run_id"])
    table.to_csv("search_results.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(
        f"\nChampion: {champion['run_name']} "
        f"(val macro F1 = {champion['val_f1_macro']:.4f}) -> "
        f"{MODEL_NAME} v{champion['registered_version']} @{CHAMPION_ALIAS}"
    )


if __name__ == "__main__":
    main()
