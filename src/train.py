import os
from pathlib import Path
from typing import List

import joblib
from tqdm import tqdm

from preprocess import load_data, get_feature_target_split, build_preprocessor, train_test_split_data
from models import build_model_pipelines
from evaluate import evaluate_model, save_metrics
from sklearn.model_selection import ParameterGrid


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "metrics.csv"


def ensure_dirs() -> None:
    for d in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)


def main() -> None:
    ensure_dirs()

    # Load and split data
    df = load_data(str(DATA_PATH))
    X, y = get_feature_target_split(df, target_col="hospital_death")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Build preprocessors: scaled for linear models, unscaled for tree-based models
    preprocessor_scaled = build_preprocessor(X_train, scale_numeric=True)
    preprocessor_unscaled = build_preprocessor(X_train, scale_numeric=False)

    # compute class imbalance ratio (neg / pos) to pass to boosting models
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = None
    if pos > 0:
        scale_pos_weight = neg / pos
        print(f"Estimated scale_pos_weight (neg/pos): {scale_pos_weight:.2f}")

    model_searches = build_model_pipelines(
        preprocessor_scaled, preprocessor_unscaled, scale_pos_weight=scale_pos_weight
    )

    metrics_list: List[dict] = []

    # Train and evaluate each model
    for name, search in tqdm(list(model_searches.items()), desc="Models"):
        # estimate number of grid candidates and total fits (candidates * cv)
        try:
            param_grid = search.param_grid
            n_candidates = sum(1 for _ in ParameterGrid(param_grid))
        except Exception:
            n_candidates = None

        cv = getattr(search, 'cv', None)
        if n_candidates is not None and cv is not None:
            print(f"Training {name}: {n_candidates} candidates x {cv} folds = {n_candidates * cv} fits")
        else:
            print(f"Training {name}...")

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")

        # Save model
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(best_model, model_path)

        # Evaluate and save confusion matrix figure
        fig_path = FIGURES_DIR / f"confusion_matrix_{name}.png"
        metrics = evaluate_model(
            name,
            best_model,
            X_test,
            y_test,
            save_fig=True,
            fig_path=str(fig_path),
        )
        metrics_list.append(metrics)

    # Save metrics
    save_metrics(metrics_list, str(METRICS_PATH))


if __name__ == "__main__":
    main()
