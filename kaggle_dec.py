import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Lasso, LassoCV, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Ignore all warnings
warnings.filterwarnings('ignore')

# Load the datasets
data = pd.read_csv('A1_stock_volatility_labeled.csv')

# Submission Set
submission_data = pd.read_csv('A1_stock_volatility_submission.csv')

# Remove the outlier from the dataset
data_1 = data[data['Volatility'] <= 140]


# List of features for which to create lagged variables
lagged_features = ['Open', 'Close', 'High', 'Low', 'Volume', 'Amount', 'Avg_Price', 'Return']

# Creating lag2 features for each stock separately
for feature in lagged_features:
    # Creating lag2 (2 months back)
    data_1[f'{feature}_lag2'] = data_1.groupby('Stock')[feature].shift(2)

# Drop rows with any NaN values that were introduced due to shifting
data_cleaned = data_1.dropna()

# Reset the index to ensure it is continuous after dropping rows
data_cleaned.reset_index(drop=True, inplace=True)

# Filter the training data for the specific dates
data_lag2 = data_cleaned[data_cleaned['Date'] == '2023-10-01']

# Set the index to 'Stock' for both datasets for easy lookup
data_lag2.set_index('Stock', inplace=True)

# Select the necessary columns for the lagged features
lagged_features = ['Open', 'Close', 'High', 'Low', 'Volume', 'Amount', 'Avg_Price', 'Return']

# Set the index to 'Stock' for easy assignment in submission data
submission_data.set_index('Stock', inplace=True)

# Add lag1 (from 2023-10-01) and lag2 (from 2023-09-01) features to submission data
for feature in lagged_features:
    submission_data[f'{feature}_lag2'] = submission_data.index.map(data_lag2[feature])

# Reset the index in submission data after mapping
submission_data.reset_index(inplace=True)

# Extracting month, quarter, and year for both datasets
for df in [data_cleaned, submission_data]:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year

# Creating cyclical features for month and quarter for both datasets
for df in [data_cleaned, submission_data]:
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
    df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)

# Create 'Is Quarter End' and 'Is Fiscal Year End' features
for df in [data_cleaned, submission_data]:
    df['Is_Quarter_End'] = df['Month'].isin([3, 6, 9, 12]).astype(int)
    df['Is_Fiscal_Year_End'] = (df['Month'] == 12).astype(int)

# Define financial features
financial_features = ['Revenue', 'Net Income', 'Gross Profit', 'EPS', 'Total Assets', 
                      'Total Liabilities', 'Total Equity', 'Cash and Cash Equivalents', 
                      'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow']

# Filter the latest quarter data
latest_quarter_data = data_cleaned[data_cleaned['Date'] == '2023-10-01'].set_index('Stock')

# Update submission dataset with latest quarter data
submission_data.set_index('Stock', inplace=True)
for feature in financial_features:
    submission_data[feature] = submission_data.index.map(latest_quarter_data[feature])
submission_data.reset_index(inplace=True)

# Create bins for 'Total Assets', 'Revenue', and 'Net Income' for both datasets
bin_features = {
    'Total Assets': ['Small', 'Medium', 'Large'],
    'Revenue': ['Low Revenue', 'Medium Revenue', 'High Revenue'],
    'Net Income': ['Loss', 'Low Profit', 'High Profit']
}

for df in [data_cleaned, submission_data]:
    for feature, labels in bin_features.items():
        if feature == 'Net Income':
            # Special binning for Net Income (with 0 separating profit and loss)
            bins = [df[feature].min(), 0, df[feature].quantile(0.5), df[feature].max()]
        else:
            bins = [df[feature].min(), df[feature].quantile(0.33), df[feature].quantile(0.66), df[feature].max()]
        
        df[f'{feature}_Category'] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)

# Drop the sfuture predictors and Date
data_selected = data_cleaned.drop(columns=['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 'Amount', 'Avg_Price', 'Return'])
 
# Drop Date in Submission Data
submission_data = submission_data.drop(columns=['Date', 'Volatility'])

# List of features for log transformation
log_transform_features = [
    'Revenue', 'Total Assets', 'Total Liabilities',
    'Amount_lag2', 'High_lag2', 'Low_lag2', 'Close_lag2', 'Open_lag2', 'Avg_Price_lag2'
]

# Applying log transformation to the selected features
for feature in log_transform_features:
    # Adding a small constant to avoid log(0) errors
    data_selected[f'{feature}_log'] = np.log1p(data_selected[feature])
    submission_data[f'{feature}_log'] = np.log1p(submission_data[feature])

# List of features for squared and cubic transformation
squared_cubic_features = [
    'Return_lag2', 'EPS', 'Avg_Price_lag2', 'Open_lag2', 'Close_lag2',
    'High_lag2', 'Low_lag2'
]

# Applying squared and cubic transformations for both data_selected and submission_data
for feature in squared_cubic_features:
    for df in [data_selected, submission_data]:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_cubed'] = df[feature] ** 3



# Step 1: Drop 'Volatility' from X and keep it as y (target variable)
X = data_selected.drop(columns=['Volatility'])
y = data_selected['Volatility']

# Ensure submission_data has the same column order as X
submission_data = submission_data[X.columns]

# Step 2: Identify categorical and numerical columns
categorical_cols = list(X.select_dtypes(exclude=['number']).columns)  # Categorical columns
numerical_cols = list(X.select_dtypes(include=['number']).columns)    # Numerical columns

# Step 3: Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Standardize numerical columns
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)  # One-hot encode categorical columns
    ])

# Step 4: Fit the preprocessor on X (training data)
X_preprocessed = preprocessor.fit_transform(X)

# Step 5: Define the RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Step 6: Create a pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('rf', rf)
])

# Step 7: Define RMSE scorer
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Create a custom scorer for RMSE
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

# Step 8: Define hyperparameter grid for tuning
param_grid = {
    'rf__n_estimators': [150, 200, 300, 400],        # Increased range of estimators
    'rf__max_depth': [8, 10, 12, 15],           # Fine-tuning around best value
    'rf__min_samples_split': [5, 10],           # Narrowing split values
    'rf__min_samples_leaf': [1, 2],             # Focus on best leaf values
}

# Step 9: Use GridSearchCV with n_jobs=-1 and verbose=3 to track progress
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring=rmse_scorer, n_jobs=-1, verbose=3)

# Step 10: Fit the model and search for the best parameters
grid_search.fit(X, y)

# Step 11: Best hyperparameters
best_params = grid_search.best_params_
print("Best parameters found by GridSearchCV:", best_params)

# Step 12: Using the best hyperparameters to fit the final model
best_rf = grid_search.best_estimator_

# Step 13: Preprocess the submission data using the fitted preprocessor
submission_data_preprocessed = preprocessor.transform(submission_data)

# Step 14: Predict on the test set (submission_data should have the same columns as X)
predictions = best_rf.predict(submission_data_preprocessed)

# Step 15: Ensure submission_data has 'Stock' column and add 'Volatility' predictions
submission_output = submission_data[['Stock']].copy()
submission_output['Volatility'] = predictions

# Step 16: Save the output to 'kaggle_rf.csv' as per the required format
submission_output.to_csv('kaggle_rf.csv', index=False)
