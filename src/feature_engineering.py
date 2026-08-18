from datetime import datetime
import pandas as pd

# set current year
CURRENT_YEAR = datetime.now().year

# create additional columns
def create_features(df):

    df["car_age"] = CURRENT_YEAR - df["year"]

    df["engine_volume_liters"] = df["volume_cm3"] / 1000

    df["mileage_per_year"] = (
        df["mileage_kilometers"] /
        df["car_age"].replace(0,1)
    )

    df["is_high_mileage"] = (
        df["mileage_kilometers"] > 200000
    ).astype(int)

    df["is_newer_car"] = (
        df["year"] >= 2018
    ).astype(int)

    df["brand_model"] = (
        df["make"] + "_" + df["model"]
    )

    return df

def main() -> None:

    RAW_DATA_PATH = "data/cars.csv"
    CLEANED_DATA_PATH = "data/cars_dataset_cleaned.csv"
    FEATURES_DATA_PATH = "data/cars_dataset_cleaned_with_features.csv"


    """Load cleaned data, build features, and save the feature-engineered dataset."""
    print("Loading cleaned dataset...")
    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)

    print("Building features...")
    df_features = create_features(df_cleaned)

    print("Saving feature-engineered dataset...")
    df_features.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"Feature-engineered dataset saved to: {FEATURES_DATA_PATH}")



if __name__ == "__main__":
    main()