# Patient Hospital Death Prediction

This project trains several machine learning models (logistic regression and tree-based models) to predict in-hospital death using an ICU dataset.

## Project structure

- `data/dataset.csv` – input dataset
- `notebooks/eda.ipynb` – exploratory data analysis notebook
- `src/preprocess.py` – data loading, feature/target split, preprocessing, and train/test split
- `src/models.py` – model and hyperparameter grid definitions
- `src/evaluate.py` – evaluation utilities and metrics saving
- `src/train.py` – end-to-end training, model saving, and metrics writing
- `results/models/` – saved model artifacts (`.joblib`)
- `results/figures/` – (optional) figures
- `results/metrics.csv` – evaluation metrics for each model
- `requirements.txt` – Python dependencies

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run training

From the project root:

```bash
python -m src.train
```

This will:

- Load `data/dataset.csv`
- Drop obvious identifier/leakage columns (e.g., `encounter_id`, `patient_id`, `hospital_id`, `icu_id`, `apache_4a_hospital_death_prob`, `apache_4a_icu_death_prob`)
- Perform a stratified train/test split on `hospital_death`
- Build preprocessing (scaling numeric features, one-hot encoding categoricals)
- Train and cross-validate:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Save the best model for each family to `results/models/*.joblib`
- Evaluate on the held-out test set and save metrics to `results/metrics.csv`

## Notes on target leakage

The dataset includes `apache_4a_hospital_death_prob` and `apache_4a_icu_death_prob`, which are model-based mortality estimates. To avoid overly optimistic performance and target leakage, these columns are excluded when forming the feature matrix.

We also drop pure identifiers (`encounter_id`, `patient_id`, `hospital_id`, `icu_id`) that don't generalize and could cause data leakage across splits.
