import requests
import pandas as pd

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
import matplotlib.pyplot as plt

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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

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

