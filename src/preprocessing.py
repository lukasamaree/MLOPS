import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, f_regression

# Load data
df = pd.read_csv("data/restaurant_data.csv")

# Extract target
y = df["Revenue"].values.reshape(-1, 1)

# Select features (exclude ID column and Revenue)
features = df.columns[1:-1]
X = df[features]

# Impute target (in case of missing revenue)
impy = SimpleImputer(strategy="mean")
impy.fit(y)
y = impy.transform(y)

# Split into train/test (optional – here we use all data for processing)
# If you want to split, do it here using train_test_split

# Define numeric preprocessing: median imputation + standard scaling
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Define categorical preprocessing: one-hot encoding + f_regression feature selection
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine into full preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_include=["int64", "float64"])),
        ("cat", categorical_transformer, make_column_selector(dtype_exclude=["int64", "float64"]))
    ]
)

# Wrap preprocessor into a pipeline (can add model later)
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor)
    ]
)

# Fit and transform
clf.fit(X, y)
X_new = clf.transform(X)

# Convert to DataFrame and attach the target
X_new = pd.DataFrame(X_new)
X_new["Revenue"] = y

# Save to CSV
X_new.to_csv("data/processed_restaurant_data.csv", index=False)

# Save the pipeline
with open("data/pipeline_restaurant.pkl", "wb") as f:
    pickle.dump(clf, f)
