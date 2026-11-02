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


# --- Logistic Regression ---
# Q1

logistic_regression_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_regression_pipeline.fit(X_train, y_train)

logistic_predictions = logistic_regression_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, logistic_predictions))
print("Precision:", precision_score(y_test, logistic_predictions, pos_label=1))
print("Recall:", recall_score(y_test, logistic_predictions, pos_label=1))
print("F1:", f1_score(y_test, logistic_predictions, pos_label=1))

# The logistic regression model achieved 86.67% accuracy.
# Precision was 85.71%, recall was 88.52%, and F1 was 87.10%.
# The model correctly predicts most good-for-running days, with slightly more false positives than false negatives.


# Q2

feature_names = X_train.columns
feature_coefficients = logistic_regression_pipeline.named_steps["classifier"].coef_[0]

for index in range(len(feature_names)):
    print(f"{feature_names[index]}: {feature_coefficients[index]:.4f}")

# Negative coefficients push the prediction toward "skip" (class 0).
# Wind speed and precipitation push most strongly toward "skip" for this example.
# This matches my intuition because windy and rainy weather can make running less comfortable.

# Q3
good_probabilities = logistic_regression_pipeline.predict_proba(X_test)

for index in range(5):
    print(f"Day {index + 1}: {good_probabilities[index][1]:.2f}")

# A probability near 0.5 means the model is less certain whether the day is good or not.


# --- Evaluation ---
# Q1

confusion_matrix_values = confusion_matrix(y_test, logistic_predictions)

confusion_matrix_display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix_values,
    display_labels=["skip", "good"]
)

confusion_matrix_display.plot()

plt.savefig("outputs/logreg_confusion_matrix.png", bbox_inches="tight")
plt.show()

# The model made 9 false positives and 7 false negatives.
# False positives may matter more because users could be encouraged to run when the weather is unsuitable.


# --- Joblib ---
# Q1
import os

os.makedirs("models", exist_ok=True)

joblib.dump(logistic_regression_pipeline, "models/warmup_model.pkl")

loaded = joblib.load("models/warmup_model.pkl")

assert (loaded.predict(X_test) == logistic_regression_pipeline.predict(X_test)).all()

print("Predictions match.")

#Saving only LogisticRegression without the scaler would mean the input data
# would not be scaled before prediction, which could lead to different results.
