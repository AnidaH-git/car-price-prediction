# Car Price Prediction Project
A complete machine learning regression project for predicting the prices of used cars based on their technical specifications and characteristics.

## Project Goal
The goal of this project is to build a machine learning model that predicts the price in USD (priceUSD) of a used car.
Unlike a classification problem, where a model predicts a category, this is a regression problem because the model predicts a continuous numerical value.
The model uses information such as:
make and model,
year,
condition,
mileage (kilometers),
fuel type,
engine volume,
color,
transmission,
drive unit,
segment.
The main objective is not to predict the exact price of every car, but to minimize the difference between the predicted and actual prices.

## Exploratory Data Analysis
The exploratory data analysis (EDA) is used to understand the structure and quality of the dataset before training the models.
The analysis focuses on:
dataset structure and dimensions,
data types,
missing values,
duplicate records,
numerical features,
categorical features,
potential extremes and unusual values.
This step helps identify potential data-quality problems and better understand which features may be useful for predicting car prices.

## Data Cleaning
Several cleaning steps are performed before the data is used for machine learning.
The cleaning process includes:
standardizing column names,
handling missing values,
removing duplicate rows,
checking and correcting data types,
checking numerical values for inconsistencies,
preparing the dataset for feature engineering.
The cleaned dataset is then used as the basis for the machine learning pipeline.

## Feature Engineering
Additional features are created to provide the models with more useful information.
Examples include:
car age — calculated from the car's year,
engine volume in liters — converted from cubic centimeters,
mileage per year — estimated from mileage and car age,
high-mileage indicator — identifies vehicles with more than 200,000 km,
newer-car indicator — identifies vehicles from 2018 or later.
These features can help the models identify relationships between a vehicle's characteristics and its price.

## Preprocessing
The dataset contains both numerical and categorical features, so different preprocessing techniques are required.
Numerical features are:imputed when necessary using the median,standardized using StandardScaler.
Categorical features such as make, model, fuel type, and transmission are: filled when values are missing, converted into numerical representations using One-Hot Encoding.
The preprocessing steps are implemented using a scikit-learn ColumnTransformer and Pipeline.
This ensures that the same preprocessing transformations are applied consistently during training and prediction and helps prevent data leakage.

## Model Training
Several regression algorithms are trained and compared to determine which approach performs best on the dataset.
The initial model comparison includes:
Linear Regression — provides a simple baseline,
Decision Tree Regressor — captures nonlinear relationships,
Random Forest Regressor — combines multiple decision trees,
HistGradientBoosting Regressor — uses gradient boosting to model more complex relationships.
Each model uses the same training and test split and the same preprocessing pipeline to ensure a fair comparison.

## Model Evaluation
The models are evaluated using several regression metrics.
**MAE — Mean Absolute Error**-
Measures the average absolute difference between the actual and predicted prices.
A lower MAE means that the model's predictions are closer to the actual prices on average.
**MSE — Mean Squared Error**-
Measures the average squared difference between actual and predicted prices.
Because the errors are squared, larger prediction errors have a greater impact on this metric.
**RMSE — Root Mean Squared Error**-
RMSE is the square root of MSE and is expressed in the same units as the target variable.
For this project, RMSE is expressed in USD, making it easier to interpret the size of prediction errors.
**R² — R-squared**-
R² measures how much of the variation in car prices is explained by the model.
A higher R² generally indicates better predictive performance.
The models are compared using all of these metrics rather than relying on a single score.

## Running the Project
1. Clone or download the repository:
Download the repository to your local machine.
2. Install the required dependencies:
Install the Python packages required by the project.
3. Run the Jupyter Notebook.
Open:
notebook/02_car_price_prediction.ipynb.
The notebook contains the main exploratory analysis, model training, evaluation, and model comparison.
4. Source files:
The src folder contains the individual scripts used for:
Data cleaning,
Feature engineering,
Data preprocessing,
Model training,
Model evaluation,
Model comparison,
The processed CSV datasets are stored in the data folder, while trained models are stored in the models folder.

## Result
The final model is selected based on its performance on the test dataset, with particular attention to MAE, RMSE, and R².
The purpose of the project is to demonstrate a complete end-to-end machine learning workflow for a real-world regression problem, from raw data exploration and cleaning through feature engineering, preprocessing, model comparison, and final model selection.

## Conclusion
The results show that tree-based ensemble models significantly outperform the Decision Tree and Linear Regression models. **Random Forest** achieved the lowest MAE of 1209.10, indicating that it provides the most accurate predictions on average. However, **Hist Gradient Boosting** achieved the lowest RMSE (2999.41) and the highest R² (0.8781), explaining approximately 87.8% of the variation in the target variable.
**Linear Regression** performed the worst, suggesting that the relationship between the input features and target is nonlinear.
Overall, the ensemble models demonstrate that more complex nonlinear learning methods are better suited to this dataset.





