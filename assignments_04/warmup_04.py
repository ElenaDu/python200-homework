import json


# Question 1
class ModelInfo:
    def __init__(self, path):
        with open(path, "r") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        return len(self.data["features"])


# Create a small JSON file
sample_data = {
    "features": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max"
    ]
}

with open("sample_metadata.json", "w") as file:
    json.dump(sample_data, file)

info = ModelInfo("sample_metadata.json")

print("Feature count:", info.feature_count())

# Loading the JSON in __init__ is more efficient because the file is read only once, even if feature_count() is called many times.



# Question 2
from dataclasses import dataclass


@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        if self.will_rain:
            rain_status = "rain expected"
        else:
            rain_status = "no rain expected"

        return f"{self.date}: high {self.high_c}C, {rain_status}"


# Create two DayForecast objects
day1 = DayForecast("2023-06-15", 24.1, True)
day2 = DayForecast("2023-06-16", 28.5, False)

print(day1.summary())
print(day2.summary())

# The @dataclass decorator automatically generates __init__(), so we do not have to write the constructor ourselves.



# Question 3

import pytest


def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Invalid temperature: {text}")


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Without match, pytest.raises(ValueError) checks only that a
# ValueError is raised, not whether its message contains the expected word "temperature".

