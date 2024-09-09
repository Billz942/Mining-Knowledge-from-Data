import pandas as pd
import warnings
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold

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
data_lag2 = data_cleaned[data_cleaned['Date'] == '2023-09-01']

# Set the index to 'Stock' for both datasets for easy lookup
data_lag2.set_index('Stock', inplace=True)

# Select the necessary columns for the lagged features
lagged_features = ['Open', 'Close', 'High', 'Low', 'Volume', 'Amount', 'Avg_Price', 'Return']

# Set the index to 'Stock' for easy assignment in submission data
submission_data.set_index('Stock', inplace=True)

# Add lag2 (from 2023-09-01) features to submission data
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


# Drop the 'Volatility' column from X (features) and keep it as y (target)
X = data_selected.drop(columns=['Volatility'])
y = data_selected['Volatility']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KFold with 10 folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(exclude=['number']).columns
numerical_cols = X_train.select_dtypes(include=['number']).columns

# Create column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

# Define a range of alphas (lambdas) to test
alphas = np.logspace(-2, 2, 100)

# Initialize a list to store RMSE for each alpha across all folds
rmse_scores = []

# Perform 10-fold cross-validation manually to compute RMSE for each alpha
for alpha in alphas:
    fold_rmse = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Create pipeline for each fold
        ridge_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('ridge', Ridge(alpha=alpha))
        ])
        
        # Fit the model
        ridge_pipeline.fit(X_train_fold, y_train_fold)
        
        # Predict and calculate RMSE for the validation fold
        y_val_pred = ridge_pipeline.predict(X_val_fold)
        fold_rmse.append(np.sqrt(mean_squared_error(y_val_fold, y_val_pred)))
    
    # Average RMSE for this alpha
    rmse_scores.append(np.mean(fold_rmse))

# Find the best alpha based on RMSE
best_alpha_ridge = alphas[np.argmin(rmse_scores)]

# Create the final pipeline with the best alpha
final_ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('ridge', Ridge(alpha=best_alpha_ridge))
])

# Fit the final pipeline to the training data
final_ridge_pipeline.fit(X, y)

# Apply the same preprocessing to the submission data
submission_data_preprocessed = preprocessor.transform(submission_data)

# Predict on the preprocessed submission data
predictions = final_ridge_pipeline.named_steps['ridge'].predict(submission_data_preprocessed)

# Ensure submission_data has 'Stock' column and add 'Volatility' predictions
submission_output = submission_data[['Stock']].copy()
submission_output['Volatility'] = predictions

# Save the output to 'pred_values.csv' as per the required format
submission_output.to_csv('pred_values.csv', index=False)

print(submission_output)