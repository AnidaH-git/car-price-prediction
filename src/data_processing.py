import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#df = pd.read_csv("data/cars_dataset_cleaned_with_features.csv")
# Define filepath
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cars_dataset_cleaned_with_features.csv"
df = pd.read_csv(DATA_PATH)

# Define target
TARGET_COLUMN = "priceusd"

# Define numerical columns
numeric_features = [
    "year",
    "mileage_kilometers",
    "volume_cm3",
    "car_age",
    "engine_volume_liters",
    "mileage_per_year",
    "is_high_mileage",
    "is_newer_car",
]

# Define categorical columns
categorical_features = [
    "make",
    "model",
    "fuel_type",
    "condition",
    "color",
    "transmission",
    "drive_unit",
    "segment",
    "brand_model",
]

# Create helper function which returns a combined list of all numeric and categorical feature columns
def get_all_feature_columns() -> list[str]:
    return numeric_features + categorical_features

# Create helper function which splits a DataFrame into feature matrix X and target series y using the predefined feature columns and target column
def split_features_and_target(df: pd.DataFrame,) -> tuple[pd.DataFrame, pd.Series]:
    X = df[get_all_feature_columns()].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


# Numeric pipeline: Imputes missing values with median, then scales features
def _build_numeric_transformer() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


# Categorical pipeline: Fills missing entries with 'missing', then applies one-hot encoding
def _build_categorical_transformer() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    missing_values=np.nan,
                    strategy="constant",
                    fill_value="missing",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    max_categories=20,
                ),
            ),
        ]
    )


# Create ColumnTransformer orchestrator
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", _build_numeric_transformer(), numeric_features),
            ("categorical",_build_categorical_transformer(),categorical_features,),
        ]
    )

