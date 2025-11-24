from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore


def build_model_pipelines(preprocessor_scaled, preprocessor_unscaled, scale_pos_weight: float | None = None) -> Dict[str, GridSearchCV]:
    """Create model pipelines with expanded hyperparameter grids.

    Two preprocessors are accepted:
    - preprocessor_scaled: for models that benefit from numeric scaling (e.g., logistic regression)
    - preprocessor_unscaled: for tree-based models (no numeric scaling)
    """

    models = {}

    # Logistic Regression (use scaled preprocessor)
    log_reg = Pipeline(
        steps=[("preprocess", preprocessor_scaled), ("clf", LogisticRegression(max_iter=2000))]
    )
    log_reg_param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__class_weight": [None, "balanced"],
    }
    models["log_reg"] = GridSearchCV(
        log_reg,
        param_grid=log_reg_param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
    )

    # Random Forest (tree-based, use unscaled preprocessor)
    rf = Pipeline(
        steps=[("preprocess", preprocessor_unscaled), ("clf", RandomForestClassifier(random_state=42))]
    )
    rf_param_grid = {
        "clf__n_estimators": [50, 100, 300],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__class_weight": [None, "balanced"],
    }
    models["random_forest"] = GridSearchCV(
        rf,
        param_grid=rf_param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
    )

    # Gradient Boosting 
    gb = Pipeline(
        steps=[("preprocess", preprocessor_unscaled), ("clf", GradientBoostingClassifier(random_state=42))]
    )
    gb_param_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__max_depth": [3, 5, 7],
        "clf__subsample": [0.8, 1.0],
    }
    models["grad_boost"] = GridSearchCV(
        gb,
        param_grid=gb_param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
    )

    # XGBoost
    if XGBClassifier is not None:
        xgb = Pipeline(
            steps=[
                ("preprocess", preprocessor_unscaled),
                (
                    "clf",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        use_label_encoder=False,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        xgb_param_grid = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
        }
        # if a scale_pos_weight value is provided (ratio neg/pos), include it for tuning
        if scale_pos_weight is not None:
            xgb_param_grid["clf__scale_pos_weight"] = [1, float(scale_pos_weight)]
        models["xgboost"] = GridSearchCV(
            xgb,
            param_grid=xgb_param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=1,
        )

    # LightGBM (optional tree-based)
    if LGBMClassifier is not None:
        lgbm = Pipeline(
            steps=[
                ("preprocess", preprocessor_unscaled),
                (
                    "clf",
                    LGBMClassifier(
                        objective="binary",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        lgbm_param_grid = {
            "clf__n_estimators": [100, 200, 400],
            "clf__max_depth": [-1, 5, 10],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__num_leaves": [31, 63],
        }
        if scale_pos_weight is not None:
            lgbm_param_grid["clf__scale_pos_weight"] = [1, float(scale_pos_weight)]
        models["lightgbm"] = GridSearchCV(
            lgbm,
            param_grid=lgbm_param_grid,
            cv=5,
            scoring="roc_auc",
            n_jobs=1,
        )

    return models
