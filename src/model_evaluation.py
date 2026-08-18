import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from data_processing import split_features_and_target

DATA_PATH = "data/cars_dataset_cleaned_with_features.csv"
MODEL_PATH = "models/linear_regression_model.joblib"

# Model testing
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Splitting features and target...")
X, y = split_features_and_target(df)

# Split data
print("Creating the same train/test split...")
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

# Read model 
print("Loading trained model...")
loaded_model = joblib.load(MODEL_PATH)

# Predictions on test sample
print("Making predictions...")
y_pred = loaded_model.predict(X_test)
print(y_pred[:10])

# Show metrics evaluation
print("Calculating regression metrics...")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Show metrics results
metrics = pd.DataFrame({
"metric": ["MAE", "MSE", "RMSE", "R2"],
"value": [mae, mse, rmse, r2],
})
print("\nRegression metrics:")
print(metrics)

# Show comparsion between actual and predicted results
comparison = X_test.copy()

comparison["Actual"] = y_test.values

comparison["Predicted"] = y_pred

comparison["Error"] = (
    comparison["Actual"] -
    comparison["Predicted"]).abs()

print(comparison[[
    "Actual",
    "Predicted",
    "Error"
]].head(10))