# Task 4

import joblib
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import WeatherClassifier, Prediction


@pytest.fixture
def classifier(tmp_path):
    """Create and save a small weather model for testing."""

    X = pd.DataFrame(
        [
            [25, 15, 0, 5],
            [28, 18, 0, 8],
            [30, 20, 0, 6],
            [10, 2, 15, 30],
            [8, 1, 20, 35],
            [12, 4, 10, 25],
        ],
        columns=[
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
    )

    y = [1, 1, 1, 0, 0, 0]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )

    model.fit(X, y)

    model_path = tmp_path / "weather_model.pkl"
    joblib.dump(model, model_path)

    return WeatherClassifier(model_path)


# Test 1: A good weather day returns "good".

def test_good_day_returns_good(classifier):
    day = {
        "temperature_2m_max": 25,
        "temperature_2m_min": 15,
        "precipitation_sum": 0,
        "wind_speed_10m_max": 5,
    }

    result = classifier.predict([day])

    assert result[0].label == "good"


# Test 2: A bad weather day returns "skip".

def test_bad_day_returns_skip(classifier):
    day = {
        "temperature_2m_max": 10,
        "temperature_2m_min": 2,
        "precipitation_sum": 15,
        "wind_speed_10m_max": 30,
    }

    result = classifier.predict([day])

    assert result[0].label == "skip"


# Test 3: The probability is between 0 and 1.

def test_probability_is_between_zero_and_one(classifier):
    day = {
        "temperature_2m_max": 25,
        "temperature_2m_min": 15,
        "precipitation_sum": 0,
        "wind_speed_10m_max": 5,
    }

    result = classifier.predict([day])

    assert 0 <= result[0].probability <= 1


# Test 4: Three input days return three Prediction objects in order.

def test_three_days_return_predictions_in_order(classifier):
    days = [
        {
            "temperature_2m_max": 25,
            "temperature_2m_min": 15,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 5,
        },
        {
            "temperature_2m_max": 10,
            "temperature_2m_min": 2,
            "precipitation_sum": 15,
            "wind_speed_10m_max": 30,
        },
        {
            "temperature_2m_max": 28,
            "temperature_2m_min": 18,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 8,
        },
    ]

    results = classifier.predict(days)

    assert len(results) == 3
    assert all(isinstance(result, Prediction) for result in results)
    assert results[0].label == "good"
    assert results[1].label == "skip"
    assert results[2].label == "good"


# Test 5: Empty input returns an empty list.

def test_empty_days_returns_empty_list(classifier):
    assert classifier.predict([]) == []


# Test 6: Missing weather features raise ValueError.

def test_missing_feature_raises_value_error(classifier):
    day = {
        "temperature_2m_max": 25,
        "temperature_2m_min": 15,
        "precipitation_sum": 0,
    }

    with pytest.raises(ValueError, match="wind_speed_10m_max"):
        classifier.predict([day])


# Test 7: A missing model file raises FileNotFoundError.

def test_missing_model_file_raises_error(tmp_path):
    missing_path = tmp_path / "missing_model.pkl"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        WeatherClassifier(missing_path)


# Parametrize test: Check multiple valid weather predictions.

@pytest.mark.parametrize(
    "temperature, precipitation, wind_speed, expected_label",
    [
        (25, 0, 5, "good"),
        (10, 15, 30, "skip"),
    ],
)
def test_parametrized_weather_predictions(
    classifier, temperature, precipitation, wind_speed, expected_label
):
    day = {
        "temperature_2m_max": temperature,
        "temperature_2m_min": 15,
        "precipitation_sum": precipitation,
        "wind_speed_10m_max": wind_speed,
    }

    result = classifier.predict([day])

    assert result[0].label == expected_label


# Deliberately changing the prediction label caused three tests to fail:
# test_good_day_returns_good,
# test_three_days_return_predictions_in_order, and
# test_parametrized_weather_predictions[25-0-5-good].
# This confirmed that the tests catch incorrect prediction labels.