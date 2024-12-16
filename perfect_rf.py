import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

train = pd.read_csv('D:/Code/Python4/ML_CMI/train.csv')
df_test = pd.read_csv('D:/Code/Python4/ML_CMI/test.csv')

# Get the intersection of columns in train and df_test
common_columns = train.columns.intersection(df_test.columns)

# Create the new DataFrame 'df_train' with only the common columns
df_train = train[common_columns].copy()

# Add the 'sii' column
if 'sii' in train.columns:
    df_train.loc[:, 'sii'] = train['sii']

# Identify columns in 'train' that are missing in 'df_test'
missing_columns = [col for col in train.columns if col not in df_test.columns]

# Create a DataFrame 'missing_data' for all data in 'train' columns missing from 'df_test'
missing_data = train[missing_columns]

# Identify all columns that contribute to 'PCIAT-PCIAT_Total', excluding 'PCIAT-PCIAT_Total' itself
pciat_columns = [col for col in train.columns if col.startswith('PCIAT-PCIAT_') and col != 'PCIAT-PCIAT_Total']

# Create a new boolean column to check the correctness of 'PCIAT-PCIAT_Total'
missing_data.loc[:, 'PCIAT_Total_Correct'] = train[pciat_columns].fillna(0).sum(axis=1) == train['PCIAT-PCIAT_Total']

# Identify the PCIAT-PCIAT_number columns (excluding 'PCIAT-PCIAT_Total')
pciat_columns = [col for col in train.columns if col.startswith('PCIAT-PCIAT_') and col != 'PCIAT-PCIAT_Total']

# Initialize the KNNImputer with a reasonable number of neighbors (e.g., 5)
imputer = KNNImputer(n_neighbors=5)

# Apply the imputer to the PCIAT-PCIAT_number columns
imputed_data = imputer.fit_transform(train[pciat_columns])

# Round the imputed values to the nearest integer and cast them to int type
train[pciat_columns] = imputed_data.round().astype(int)

# Confirm that missing values have been imputed
print(train[pciat_columns].isnull().sum())

# Update the 'PCIAT-PCIAT_Total' column with the new sum
train['PCIAT-PCIAT_Total'] = train[pciat_columns].sum(axis=1)

# Confirm the updated values
print(train['PCIAT-PCIAT_Total'].head())

# Update the 'sii' column based on the rules defined in the data dictionary
train['sii'] = 0  # Default to 0 (None)
train.loc[(train['PCIAT-PCIAT_Total'] > 30) & (train['PCIAT-PCIAT_Total'] <= 49), 'sii'] = 1  # Mild
train.loc[(train['PCIAT-PCIAT_Total'] > 49) & (train['PCIAT-PCIAT_Total'] <= 79), 'sii'] = 2  # Moderate
train.loc[train['PCIAT-PCIAT_Total'] >= 80, 'sii'] = 3  # Severe

# Confirm the updated values
print(train['sii'].value_counts())

df_train['sii'] = train['sii']

df_train.T.head(10)

cat_columns = df_train.select_dtypes(include=['object', 'category']).columns
df_train[cat_columns].head()

df_train.drop('id', axis=1, inplace=True)

# Dividing data on categorical and numerical (without target column)

cat_columns = df_train.select_dtypes(include=['object', 'category']).columns
num_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.drop('sii')

cat_columns = (
    [  # 'Basic_Demos-Enroll_Season',
        'CGAS-Season',
        'Physical-Season',
        'Fitness_Endurance-Season',
        'FGC-Season', 'BIA-Season',
        'PAQ_A-Season',
        'PAQ_C-Season',
        'SDS-Season',
        'PreInt_EduHx-Season'])

# Define transformers for numerical and categorical columns
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_columns),
        ('cat', cat_transformer, cat_columns)
    ], remainder='drop')

# Create a pipeline with the preprocessor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)])

# Apply the pipeline
X = df_train.drop('sii', axis=1)
y = df_train['sii']  # normalize dependent variable np.log(
X_preprocessed = pipeline.fit_transform(X)

# Apply the pipline to test data

df_test_preprocessed = pipeline.transform(df_test)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to the data
X_train_smote, y_train_smote = smote.fit_resample(X_preprocessed, y)

# Check the distribution of the resampled data
from collections import Counter

print("Original training distribution:", Counter(y))
print("Resampled training distribution:", Counter(y_train_smote))

# Split the data into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train_smote, y_train_smote, test_size=0.25, random_state=123)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Random Forest
rf = RandomForestClassifier(random_state=123)
rf_params = {
    'n_estimators': [300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 3]
}

rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print("Best Parameters for Random Forest:", rf_grid.best_params_)
print("Random Forest Performance:\n", classification_report(y_valid, rf_grid.best_estimator_.predict(X_valid)))


# Ensure X_train, X_valid, y_train, y_valid are converted to pandas objects if they're NumPy arrays
X_train = pd.DataFrame(X_train) if isinstance(X_train, (np.ndarray)) else X_train
X_valid = pd.DataFrame(X_valid) if isinstance(X_valid, (np.ndarray)) else X_valid
y_train = pd.Series(y_train) if isinstance(y_train, (np.ndarray)) else y_train
y_valid = pd.Series(y_valid) if isinstance(y_valid, (np.ndarray)) else y_valid

# Concatenate Train and Validation datasets
X_final = pd.concat([X_train, X_valid])
y_final = pd.concat([y_train, y_valid])

# Random Forest
rf = RandomForestClassifier(random_state=123)
rf_params = {
    'n_estimators': [200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 3]
}

rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_final, y_final)

print("Best Parameters for Random Forest:", rf_grid.best_params_)

# Evaluate on the final dataset using cross-validation
y_pred_final = cross_val_predict(rf_grid.best_estimator_, X_final, y_final, cv=5)

# Performance Metrics
print("Final Random Forest Performance:\n", classification_report(y_final, y_pred_final))

# Confusion Matrix
cm_final = confusion_matrix(y_final, y_pred_final)


# Predict target for test data
df_test_preprocessed = pd.DataFrame(df_test_preprocessed) if isinstance(df_test_preprocessed,
                                                                        (np.ndarray)) else df_test_preprocessed
y_test_pred = rf_grid.best_estimator_.predict(df_test_preprocessed)

# Output predictions
print("Test Data Predictions:", y_test_pred)

# submission

df_submit = df_test[['id']].copy()
df_submit['sii'] = y_test_pred

df_submit.to_csv('submission.csv', index=False)
