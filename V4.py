import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

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

# Define base models with parameters
lightgbm_model = LGBMClassifier(
    learning_rate=0.046,
    max_depth=12,
    num_leaves=478,
    min_child_samples=13,
    feature_fraction=0.893,
    bagging_fraction=0.784,
    bagging_freq=4,
    reg_alpha=10,
    reg_lambda=0.01,
    random_state=123
)

xgboost_model = XGBClassifier(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=5,
    random_state=123,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

catboost_model = CatBoostClassifier(
    learning_rate=0.05,
    depth=6,
    iterations=200,
    l2_leaf_reg=10,
    random_seed=123,
    verbose=False
)

random_forest_model = RandomForestClassifier(random_state=123)

# Ensemble model using VotingClassifier
ensemble_model = VotingClassifier(
    estimators=[
        ('lightgbm', lightgbm_model),
        ('xgboost', xgboost_model),
        ('catboost', catboost_model),
        ('randomforest', random_forest_model)
    ],
    voting='soft'
)

# Define the pipeline with SMOTE
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=123)),
    ('classifier', ensemble_model)
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate on validation set
y_valid_pred = pipeline.predict(X_valid)
print("Validation Performance:\n", classification_report(y_valid, y_valid_pred))

# Preprocess and predict on the test set
y_test_pred = pipeline.predict(df_test)

# Save predictions to CSV
df_submit = df_test[['id']].copy()
df_submit['sii'] = y_test_pred
df_submit.to_csv('submission.csv', index=False)