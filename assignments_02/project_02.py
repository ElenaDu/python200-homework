import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Task 1: Fetch the Data

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 30.6333,
    "longitude": -97.6770,
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

print("Dataset shape:", df.shape)
print(df.head())

# I chose Georgetown, Texas.


# Task 2: Look at the Features

# Exploring the data
print("\nSummary statistics:")
print(df.describe())

print("\nMissing values in each column:")
print(df.isnull().sum())

# The dataset has no missing values, so no additional actions are needed.

# Find the correlation of each numeric column with temperature_2m_max, sorted
correlations = df.corr(numeric_only=True)["temperature_2m_max"].sort_values(ascending=False)
print("\nCorrelations with daily high temperature:")
print(correlations)

# Plot daily low versus daily high

plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"])
plt.title("Daily High vs. Low Temperature in Georgetown, TX")
plt.xlabel("Daily Low Temperature (°C)")
plt.ylabel("Daily High Temperature (°C)")
plt.savefig("outputs/project_02_high_vs_low.png")
plt.show()

# The plot shows how daily high temperatures generally increase as daily low temperatures increase.

# Task 3: Engineer a Feature

df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

print("\nSummer and non-summer days:")
print(df["is_summer"].value_counts().sort_index().rename({0: "Non-summer", 1: "Summer"}))

# I expect is_summer to help predict the daily high because Georgetown, Texas, generally has hotter temperatures in summer than in other seasons

# Task 4: Baseline Model

X = df[["temperature_2m_min"]]
y = df["temperature_2m_max"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("\nTask 4: Baseline Model")
print("Slope:", model.coef_[0])
print("RMSE:", rmse)
print("R²:", r2)

# RMSE is in degrees Celsius. It tells us that a typical prediction is off by approximately 2.9°C degrees from the actual daily high.

# Task 5: Full Model

feature_cols = ["temperature_2m_min", "precipitation_sum", "wind_speed_10m_max", "is_summer"]
                
X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("\nTask 5: Full Model")
print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("Test RMSE:", rmse)

# Compare the full model with the Task 4 baseline.
baseline_test_r2 = 0.8469780287323851
print("Baseline test R²:", baseline_test_r2)
print("Change in test R²:", test_r2 - baseline_test_r2)

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# The daily low temperature and precipitation/wind have different effects:
# higher daily lows raise the predicted high, while more precipitation and higher wind speeds lower it slightly.

# The negative summer coefficient is surprising because Georgetown is hotter in summer.
# However, Central Texas is also hot in spring and fall, so the difference between summer and non-summer days may be smaller than expected.

# Train R² (0.849) and test R² (0.858) are very close.
# This means the model performs similarly on the training and test data, with no obvious sign of overfitting.


# Task 6: Evaluate and Summarize

plt.scatter(y_test_pred, y_test)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.title("Predicted vs Actual (Full Model) in Georgetown, TX")
plt.xlabel("Predicted High Temperature (°C)")
plt.ylabel("Actual High Temperature (°C)")
plt.savefig("outputs/predicted_vs_actual_high.png")
plt.show()

# The model's errors appear somewhat larger at the low end of the temperature range.
# Some cooler days have predictions that are noticeably different from the actual highs, while most predictions at the high end are closer
# to the diagonal reference line.

#---Summary:---
# The dataset contains 365 daily weather observations, and the test set contains 73 days (20% of the dataset).

# The full model has a test RMSE of about 2.79°C, meaning its predictions are typically off by roughly 2.8°C from the actual daily high.

# Its test R² is about 0.858, meaning it explains approximately 85.8% of the variation in daily high temperatures in the test set.

# Daily low temperature has the largest effect on the predicted high.
# Its positive coefficient (+0.941) means higher daily lows are associated with higher predicted daily highs.

# One surprising result is the negative summer coefficient (-0.992), because Georgetown is hotter in summer. Central Texas can also be hot
# in spring and fall, so the difference between summer and non-summer days may be smaller than expected.








