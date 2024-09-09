# Mining-Knowledge-from-Data

Python Version 3.12.5

#### Libraries Imported:
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, KFold

#### Code:
Run the code in your favourite IDE with the required datasets ('A1_stock_volatility_labeled.csv', 'A1_stock_volatility_submission.csv')  present in your current working directory. 

An output file containing the predictions will be generated.