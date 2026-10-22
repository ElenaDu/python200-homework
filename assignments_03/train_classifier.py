import requests
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import json
import os
import platform
import joblib
import sklearn

# Task 1: Fetch the Data

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 30.63,
    "longitude": -97.68,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Chicago",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])

df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Data shape:", df.shape)
print(df.head())

# I used Georgetown, Texas, for the weather data.

#Task 2: Engineer the Label

# Adjusted the maximum temperature from 26°C to 30°C to better reflect Georgetown's warmer climate.

df["good_for_running"] = (
    (df["temperature_2m_max"] >= 7) &
    (df["temperature_2m_max"] <= 30) &
    (df["temperature_2m_min"] >= 0) &
    (df["precipitation_sum"] < 3) &
    (df["wind_speed_10m_max"] < 30)
).astype(int)

print(df["good_for_running"].value_counts().sort_index())
good_days_fraction = df["good_for_running"].mean()

print("Fraction of good days:", round(good_days_fraction, 3))

# About 36.7% of days were good for running in Georgetown.
# This seems reasonable given the hot Texas summers and our temperature, precipitation, and wind requirements.
# Since there are more skip days than good days, precision, recall, and F1 score are important metrics to evaluate the model.


# Task 3: Build and Evaluate a KNN Classifier

X = df[
    [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]
]

y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Test different values of k using 5-fold cross-validation.
neighbor_counts = [1, 3, 5, 7, 9, 11, 15]

for k in neighbor_counts:
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k)),
    ])

    scores = cross_val_score(
        knn_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    print(f"k={k}, mean CV accuracy={scores.mean():.3f}")

# Choose the k with the highest mean cross-validation accuracy.
best_k = None
best_score = 0

for k in neighbor_counts:
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=k)),
    ])

    scores = cross_val_score(
        knn_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    mean_score = scores.mean()

    if mean_score > best_score:
        best_score = mean_score
        best_k = k

print("Best k:", best_k)

# Train the final model using the best k.
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=best_k)),
])

knn_pipeline.fit(X_train, y_train)

# Evaluate the model on the test set.
predictions = knn_pipeline.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, predictions))
print(classification_report(
    y_test,
    predictions,
    target_names=["skip", "good"],
))

# The best k was 5, with a mean cross-validation accuracy of 89.7%.
# The model achieved 91.8% test accuracy.
# Recall for good days was 96%, meaning the model identified most of the days that met our running criteria.


# Task 4: Build and Evaluate a Logistic Regression Classifier

logistic_regression_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

logistic_regression_pipeline.fit(X_train, y_train)
logistic_predictions = logistic_regression_pipeline.predict(X_test)

# Print evaluation metrics.
print("Test accuracy:", accuracy_score(y_test, logistic_predictions))
print("Precision:", precision_score(y_test, logistic_predictions, pos_label=1))
print("Recall:", recall_score(y_test, logistic_predictions, pos_label=1))
print("F1:", f1_score(y_test, logistic_predictions, pos_label=1))

print("\nClassification report:")
print(classification_report(
    y_test,
    logistic_predictions,
    target_names=["skip", "good"],
))

# Print each feature's coefficient.
feature_names = X_train.columns
feature_coefficients = logistic_regression_pipeline.named_steps["classifier"].coef_[0]

print("\nFeature coefficients:")
for index in range(len(feature_names)):
    print(f"{feature_names[index]}: {feature_coefficients[index]:.4f}")

# Precipitation has the strongest influence toward "skip", followed by minimum temperature and wind speed.
# Maximum temperature has very little influence in this model.


# Task 5: Compare and Choose

# KNN had higher test accuracy (91.78%) than Logistic Regression (80.82%).
# For a running app, precision and recall for "good" days are important.
# Low precision means recommending running on a day labeled "skip" (false positive).
# Low recall means missing a day labeled "good" false negative).
# KNN had 96% recall for good days, compared with 59% for Logistic Regression, so it missed fewer good running days.
# I would deploy KNN because it had higher accuracy and recall for good days on this test set. 
# Both models are relatively small and fast, but Logistic Regression is easier to interpret.


# Task 6: Save the Model

os.makedirs("models", exist_ok=True)

# Save the fitted KNN pipeline.
joblib.dump(knn_pipeline, "models/weather_classifier.pkl")

knn_predictions = knn_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, knn_predictions)
test_f1 = f1_score(y_test, knn_predictions, pos_label=1)

# Save model metadata.
metadata = {
    "python_version": platform.python_version(),
    "scikit_learn_version": sklearn.__version__,
    "feature_names": list(X.columns),
    "label_rule": (
        "1 = good if maximum temperature is between 7 and 30°C, "
        "minimum temperature is at least 0°C, precipitation is under "
        "3 mm, and maximum wind speed is under 30 km/h; otherwise 0 = skip."
    ),
    "city": "Georgetown, Texas",
    "test_accuracy": test_accuracy,
    "test_f1": test_f1,
}

with open("models/weather_classifier_metadata.json", "w") as metadata_file:
    json.dump(metadata, metadata_file, indent=4)

print("Model and metadata saved successfully.")
