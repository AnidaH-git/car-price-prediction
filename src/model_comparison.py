import pandas as pd
from sklearn.ensemble import (RandomForestRegressor,HistGradientBoostingRegressor,)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from data_processing import (
split_features_and_target,
build_preprocessor,
)

DATA_PATH = "data/cars_dataset_cleaned_with_features.csv"

df = pd.read_csv(DATA_PATH)

# Define initial and target values
X, y = split_features_and_target(df)

# Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

# Models we want to evaluate and compare
models = {
"Linear Regression": LinearRegression(),
"Decision Tree": DecisionTreeRegressor(random_state=42),
"Random Forest": RandomForestRegressor(n_estimators=50,random_state=42,n_jobs=-1),
"Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=100,learning_rate=0.1,max_leaf_nodes=31,random_state=42),
}

results = [] # create empty list for results
# Loop over models from dictionary
for model_name, regressor in models.items():
    model = Pipeline(
    steps=[
    ("preprocessor", build_preprocessor()),
    ("regressor", regressor),])
    # Train model
    model.fit(X_train, y_train)
    # Model prediction
    y_pred = model.predict(X_test)
    # Evaluate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    # Append results into list
    results.append({
    "model": model_name,
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "r2": r2,
    })

# Show the results in a Data Frame 
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="mae")
results_df = results_df.reset_index(drop=True)

print(results_df)

# Show the best model
best_model = results_df.iloc[0]

print(f"\nBest model: {best_model['model']}")
print(f"MAE: {best_model['mae']:.2f}")
print(f"RMSE: {best_model['rmse']:.2f}")
print(f"R²: {best_model['r2']:.4f}")