import requests
import pandas as pd

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

