import re
import pandas as pd

RAW_DATA_PATH = "data/cars.csv"
CLEANED_DATA_PATH = "data/cars_dataset_cleaned.csv"

MISSING_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE",
}

# Standardize all the column names
def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_columns = []
    for col in df.columns:
        clean_col = col.strip().lower()
        clean_col = clean_col.replace("(", "_").replace(")", "")
        clean_col = clean_col.replace("-", "_").replace("/", "_")
        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")
        new_columns.append(clean_col)
    df.columns = new_columns
    return df

# Remove blanks from the string and object ends
def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
    return df

# replace missing values with NA
def _replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)
    return df

# standardize numeric columns with numbers
def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_columns = [
        "priceusd",
        "year",
        "mileage_kilometers",
        "volume_cm3",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# remove target rows with missing values
def _remove_rows_with_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["priceusd"])
    return df

# keep columns with price over zero
def _price_over_zero(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["priceusd"] > 0]
    return df

# select rows with year over 1980
def _year_definition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["year"] > 1980]
    return df

# remove detected duplicates
def _remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    return df

# final pipeline function
def clean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(_standardize_column_names)
        .pipe(_strip_string_values)
        .pipe(_replace_missing_like_values)
        .pipe(_convert_numeric_columns)
        .pipe(_remove_rows_with_missing_target)
        .pipe(_price_over_zero)
        .pipe(_year_definition)
        .pipe(_remove_duplicate_rows)
        .reset_index(drop=True)
    )


def main() -> None:
    print("Loading raw dataset...")
    df_raw = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning dataset...")
    df_cleaned = clean(df_raw)

    print("Saving cleaned dataset...")
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)


if __name__ == "__main__":
    main()