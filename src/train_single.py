"""
Train a single model by specifying the model name.

Usage:
    python train_single.py --model log_reg
    python train_single.py --model random_forest
    python train_single.py --model grad_boost
    python train_single.py --model xgboost
    python train_single.py --model lightgbm
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import joblib
import numpy as np

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

from preprocess import load_data, get_feature_target_split, build_preprocessor, train_test_split_data
from models import build_single_model_pipeline
from evaluate import evaluate_model, save_metrics


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "metrics.csv"


def ensure_dirs() -> None:
    """Ensure that required directories exist."""
    for d in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)



# ============================================================
#   Manual Grid Search with TRUE fold-level tqdm updates
# ============================================================
class ManualGridSearch:
    """
    A drop-in replacement for GridSearchCV that:
    - Evaluates each parameter combination manually
    - Produces EXACT per-fold tqdm updates
    - Supports estimator pipelines
    """

    def __init__(self, estimator, param_grid, scoring="roc_auc", cv=3, n_jobs=1, refit=True):
        self.estimator = estimator
        self.param_grid = list(ParameterGrid(param_grid))
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs  # Not used (kept for compatibility)
        self.refit = refit

        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None

    def fit(self, X, y):
        """Run manual grid search with per-fold tqdm."""
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        total_fits = len(self.param_grid) * self.cv
        pbar = tqdm(total=total_fits, desc="Training Progress", unit="fit", ncols=100)

        best_score = -np.inf
        best_params = None
        best_estimator = None

        # Loop over all parameter combinations
        for params in self.param_grid:
            fold_scores = []

            # For each fold
            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Clone estimator and set parameters
                model = clone(self.estimator)
                model.set_params(**params)

                # Fit
                model.fit(X_tr, y_tr)

                # Predict probabilities
                if hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred = model.predict(X_val)

                score = roc_auc_score(y_val, y_pred)
                fold_scores.append(score)

                # Update progress bar
                pbar.update(1)

            # Average CV score
            mean_cv_score = np.mean(fold_scores)

            # Track best
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                best_params = params
                best_estimator = clone(self.estimator).set_params(**params)
                best_estimator.fit(X, y)

        pbar.close()

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.best_estimator_ = best_estimator
        return self



# ============================================================
#                     Main training script
# ============================================================
def main(model_name: str) -> None:
    ensure_dirs()

    # Load data
    print(f"Loading data from {DATA_PATH}...")
    df = load_data(str(DATA_PATH))
    X, y = get_feature_target_split(df, target_col="hospital_death")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Preprocessors
    preprocessor_scaled = build_preprocessor(X_train, scale_numeric=True)
    preprocessor_unscaled = build_preprocessor(X_train, scale_numeric=False)

    # Class imbalance
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = None
    if pos > 0:
        scale_pos_weight = neg / pos
        print(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}")

    # Build pipeline
    print(f"\nBuilding {model_name} pipeline...")
    try:
        search_original = build_single_model_pipeline(
            model_name,
            preprocessor_scaled,
            preprocessor_unscaled,
            scale_pos_weight=scale_pos_weight
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("\nAvailable models: log_reg, random_forest, grad_boost, xgboost, lightgbm")
        return

    # Replace GridSearchCV with our manual search
    search = ManualGridSearch(
        estimator=search_original.estimator,
        param_grid=search_original.param_grid,
        scoring=search_original.scoring,
        cv=search_original.cv,
        n_jobs=search_original.n_jobs,
        refit=True,
    )

    # Estimated fits
    n_candidates = len(search.param_grid)
    cv = search.cv
    total_fits = n_candidates * cv
    print(f"Training {model_name}: {n_candidates} candidates × {cv} folds = {total_fits} fits")

    # Train
    print(f"\nTraining {model_name}...")
    search.fit(X_train, y_train)

    # Best model
    print(f"\nBest parameters for {model_name}:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV score (ROC-AUC): {search.best_score_:.4f}")

    best_model = search.best_estimator_

    # Save model
    model_path = MODELS_DIR / f"{model_name}.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to {model_path}")

    # Evaluate
    fig_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    metrics = evaluate_model(
        model_name,
        best_model,
        X_test,
        y_test,
        save_fig=True,
        fig_path=str(fig_path),
    )

    print(f"\nTest Set Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

    # Save metrics
    save_metrics([metrics], str(METRICS_PATH))
    print(f"\nMetrics saved to {METRICS_PATH}")
    print(f"Confusion matrix saved to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name: log_reg, random_forest, grad_boost, xgboost, or lightgbm"
    )
    args = parser.parse_args()
    main(args.model)

