import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
import joblib

df = pd.read_csv("weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---

# Q1

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.mean(axis=0))

# We fit the scaler on X_train only to prevent data leakage from the test set.

# Q2

encoder = OneHotEncoder(sparse_output=False)

season_encoded = encoder.fit_transform(df[["season"]])

print(season_encoded.shape)
print(encoder.get_feature_names_out())

# The season column became 4 columns because there are 4 unique season categories.


# Q3

X_all = df[numeric + ["season"]]

X_train_q3, X_test_q3, y_train_q3, y_test_q3 = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(), ["season"])
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

pipeline.fit(X_train_q3, y_train_q3)

y_pred = pipeline.predict(X_test_q3)

print(accuracy_score(y_test_q3, y_pred))

# The pipeline automatically fits preprocessing on the training data and applies the same transformations before prediction.


# --- KNN ---

# Q1

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

knn_pipeline.fit(X_train, y_train)

y_pred_knn = knn_pipeline.predict(X_test)

print(accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Q2

k_values = [1, 3, 5, 7, 9, 11, 15, 21]

for k in k_values:
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k))
    ])

    cv_scores = cross_val_score(
        knn_pipeline, X_train, y_train, cv=5, scoring="accuracy"
    )

    print(f"k = {k}, mean CV accuracy = {cv_scores.mean():.4f}")

# I would choose k = 11 because it has the highest mean 5-fold cross-validation accuracy (94.17%).


# Q3

knn_raw_model = KNeighborsClassifier(n_neighbors=5)
knn_raw_model.fit(X_train, y_train)

raw_predictions = knn_raw_model.predict(X_test)
raw_accuracy = accuracy_score(y_test, raw_predictions)

knn_scaled_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

knn_scaled_pipeline.fit(X_train, y_train)

scaled_predictions = knn_scaled_pipeline.predict(X_test)
scaled_accuracy = accuracy_score(y_test, scaled_predictions)

print(f"Accuracy without scaling: {raw_accuracy:.4f}")
print(f"Accuracy with scaling: {scaled_accuracy:.4f}")

# Scaling improved accuracy because KNN uses distances, and scaling prevents features with larger numerical ranges from dominating those distances.