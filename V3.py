import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline



# Load data
train = pd.read_csv('res/train.csv')
df_test = pd.read_csv('res/test.csv')

# Preprocessing: Select common columns
common_columns = train.columns.intersection(df_test.columns)
df_train = train[common_columns].copy()
if 'sii' in train.columns:
    df_train['sii'] = train['sii']

# Remove ID column
df_train.drop('id', axis=1, inplace=True)

# Separate numerical and categorical columns
cat_columns = df_train.select_dtypes(include=['object', 'category']).columns
num_columns = df_train.select_dtypes(include=['int64', 'float64']).columns.drop('sii')

# Define preprocessing for numerical and categorical features
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_columns),
        ('cat', cat_transformer, cat_columns)
    ]
)

# Drop rows with missing target values
df_train = df_train.dropna(subset=['sii'])

# Separate features and target
X = df_train.drop('sii', axis=1)
y = df_train['sii']

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=123)

# Define the pipeline with SMOTE
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=123)),
    ('classifier', RandomForestClassifier(random_state=123))
])

# Define the parameter grid for RandomForestClassifier
param_grid = {
    'classifier__n_estimators': [50, 100, 150],  # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20],    # Maximum depth of the trees
    'classifier__min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'classifier__min_samples_leaf': [1, 2, 4]     # Minimum samples required at each leaf node
}

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)

# Evaluate the best model on the validation set
best_model = grid_search.best_estimator_
y_valid_pred = best_model.predict(X_valid)

print("Validation Performance:\n", classification_report(y_valid, y_valid_pred))

# Preprocess and predict on the test set
y_test_pred = best_model.predict(df_test)

# Save predictions to CSV
df_submit = df_test[['id']].copy()
df_submit['sii'] = y_test_pred
df_submit.to_csv('submission.csv', index=False)