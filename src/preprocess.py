import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Tuple


def load_data(path: str) -> pd.DataFrame:
    """Load the CSV dataset."""
    return pd.read_csv(path)


def get_feature_target_split(df: pd.DataFrame, target_col: str = "hospital_death") -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features X and target y, dropping obvious ID/leakage columns.

    We drop encounter_id, patient_id, hospital_id, icu_id as pure identifiers.
    We also drop apache_4a_hospital_death_prob and apache_4a_icu_death_prob because they are model-based estimates of death (leakage-like).
    """
    drop_cols = [
        "encounter_id",
        "patient_id",
        "hospital_id",
        "icu_id",
        "apache_4a_hospital_death_prob",
        "apache_4a_icu_death_prob",
    ]
    df = df.copy()
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Build a preprocessing pipeline.

    If scale_numeric is True, numeric features will be median-imputed then StandardScaled.
    If False (useful for tree-based models), numeric features are median-imputed but not scaled.

    Categorical features are imputed with most frequent value then one-hot encoded.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numeric: impute missing values (median) then scale
    if scale_numeric:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        # For tree-based models we generally don't scale numeric features
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    # Categorical: impute most frequent then one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train-test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
