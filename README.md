# Patient Hospital Death Prediction

This repository implements a reproducible pipeline to explore and predict in-hospital mortality using an ICU dataset. The code provides:

- A reproducible EDA notebook to inspect data quality, distributions, and group-level mortality.
- A preprocessing module with missing-value handling and optional numeric scaling.
- Model pipelines and cross-validation for linear and tree-based classifiers (Logistic Regression, Random Forest, Gradient Boosting, and optional XGBoost / LightGBM).
- Evaluation utilities that compute metrics and save confusion-matrix figures.

## Project structure

- `data/dataset.csv` — input dataset
- `notebooks/eda.ipynb` — exploratory data analysis (EDA)
- `src/preprocess.py` — data loading, feature/target split, preprocessing builder, and stratified train/test split
- `src/models.py` — model pipelines and hyperparameter grids (GridSearchCV)
- `src/evaluate.py` — metrics calculation and figure saving
- `src/train.py` — end-to-end runner: train, save models, evaluate, and save metrics/figures
- `results/models/` — saved model artifacts (`.joblib`)
- `results/figures/` — saved plots (confusion matrices, etc.)
- `results/metrics.csv` — aggregated metrics for each trained model
- `requirements.txt` — Python dependencies

## Requirements & setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## EDA notebook

Open `notebooks/eda.ipynb` to explore:

- Data shape, dtypes, and a missingness table.
- Target class balance and counts.
- Categorical distributions and mortality split (per-category bar charts).
- Numerical feature KDEs and histograms by outcome and boxplots.
- Correlation heatmap for numeric features (selected subset) to inspect collinearity.
- ICU / hospital-level mortality rates (helpful for client-level analyses or federated setups).

The notebook also includes quick recommendations for preprocessing (columns to drop, class imbalance suggestions, and collinearity checks).

## Run training

From the project root:

```bash
cd src
python train.py
```

Actions performed by the runner:

- Load `data/dataset.csv`.
- Drop clear identifier and leakage columns when creating features (this is done in `src/preprocess.py`).
- Perform a stratified train/test split on `hospital_death`.
- Build two preprocessors from the training set:
  - `preprocessor_scaled`: numeric median imputation + StandardScaler; used for linear models.
  - `preprocessor_unscaled`: numeric median imputation only (no scaling); used for tree-based models.
- Create model pipelines and run cross-validation (GridSearchCV) to tune hyperparameters. 
- Models included:
  - Logistic Regression (uses scaled features)
  - Random Forest (unscaled)
  - Gradient Boosting (sklearn) (unscaled)
  - XGBoost (optional, unscaled)
  - LightGBM (optional, unscaled)
- Save the best estimator for each model to `results/models/{model_name}.joblib`.
- Evaluate the best estimators on the held-out test set, compute metrics (accuracy, precision, recall, F1, ROC AUC, confusion matrix), save metrics to `results/metrics.csv`, and save confusion matrix images to `results/figures/`.

## Preprocessing details

- Numeric features: missing values are imputed with the median. You can toggle scaling; the code builds both scaled and unscaled preprocessors and uses them appropriately per model.
- Categorical features: missing values are imputed with the most frequent value and one-hot encoded (with `handle_unknown='ignore'`).

Recommended preprocessing actions (also suggested in the notebook):
- Drop model-based mortality estimates (e.g., `apache_4a_hospital_death_prob`, `apache_4a_icu_death_prob`) — these are treated as leakage and are removed from the feature set by default.
- Drop obvious useless columns (e.g., `Unnamed` index columns) or columns with extremely high missingness.

## Models and class imbalance handling

- Logistic Regression: L1/L2 penalty options, `class_weight` tuning (None or `'balanced'`). Uses scaled numeric features.
- Random Forest: expanded grid including `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `class_weight`.
- Gradient Boosting (sklearn): expanded grid including `n_estimators`, `learning_rate`, `max_depth`, `subsample`, and `min_samples_leaf`.
- XGBoost / LightGBM: included with additional hyperparameters (`colsample_bytree`, `reg_alpha`, etc.). The training script computes an estimated `scale_pos_weight = (#neg) / (#pos)` from the training set and includes it in the search grid to help address class imbalance when appropriate.

For an imbalanced dataset, the pipeline will try `class_weight='balanced'` for applicable estimators and tune `scale_pos_weight` for boosting algorithms.

## Results

- Models are written to `results/models/` as `.joblib` files.
- Per-model confusion matrix PNGs are saved to `results/figures/`.
- Per-model numeric metrics are aggregated into `results/metrics.csv`.

## Performance and runtime notes

- The provided hyperparameter grids are intentionally broad; full GridSearchCV over all models may be time-consuming. For quicker experimentation consider:
  - Reducing the size of grids
  - Using `RandomizedSearchCV` instead of `GridSearchCV`
  - Decreasing `cv` folds (e.g., to 3) or running on a smaller sample

## Extending the pipeline

Suggested next steps you can ask me to implement:

- Add statistical tests (t-test / chi-square) and annotate top features in the EDA notebook.
- Save key EDA plots automatically to `results/figures/` when the notebook runs.
- Add a `--debug` or `FAST` mode to `src/train.py` that runs smaller grids / fewer CV folds for quick validation.

## Notes on target leakage

The dataset includes `apache_4a_hospital_death_prob` and `apache_4a_icu_death_prob`, which are model-based mortality estimates. To avoid overly optimistic performance and target leakage, these columns are excluded when forming the feature matrix.

We also drop pure identifiers (`encounter_id`, `patient_id`, `hospital_id`, `icu_id`) that don't generalize and could cause data leakage across splits.
