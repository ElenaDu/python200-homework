import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from weatherkit import WeatherResponse


def test_real_weather_file():
    # An absolute path based on this file works regardless of the terminal's directory.
    path = Path(__file__).parent.parent / "weather_raw.json"

    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)
    assert len(response.hourly.time) == 168


def test_invalid_latitude():
    data = {
        "latitude": 200.0,
        "longitude": -80.0,
        "timezone": "America/New_York",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [10.0],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_mismatched_lists():
    data = {
        "latitude": 35.0,
        "longitude": -80.0,
        "timezone": "America/New_York",
        "elevation": 100.0,
        "hourly": {
            "time": ["a", "b"],
            "temperature_2m": [10.0],
            "precipitation": [0.0, 0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)


def test_null_temperature():
    data = {
        "latitude": 35.0,
        "longitude": -80.0,
        "timezone": "America/New_York",
        "elevation": 100.0,
        "hourly": {
            "time": ["a"],
            "temperature_2m": [None],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(data)