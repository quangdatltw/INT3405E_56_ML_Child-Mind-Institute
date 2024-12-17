import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

# Preprocess training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Apply SMOTE
smote = SMOTE(random_state=123)
X_resampled, y_resampled = smote.fit_resample(X_train_preprocessed, y_train)

# Train a classifier using resampled data
classifier = RandomForestClassifier(random_state=123, n_estimators=100)
classifier.fit(X_resampled, y_resampled)

# Preprocess validation data and evaluate the model
X_valid_preprocessed = preprocessor.transform(X_valid)
y_valid_pred = classifier.predict(X_valid_preprocessed)

print("Validation Performance:\n", classification_report(y_valid, y_valid_pred))

# Preprocess and predict on the test set
df_test_preprocessed = preprocessor.transform(df_test)
y_test_pred = classifier.predict(df_test_preprocessed)

# Save predictions to CSV
df_submit = df_test[['id']].copy()
df_submit['sii'] = y_test_pred
df_submit.to_csv('submission.csv', index=False)
