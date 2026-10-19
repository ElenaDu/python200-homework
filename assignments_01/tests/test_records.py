from weatherkit import WeatherResponse, to_readings, HourlyReading


def make_response():
    """Create a small example response for testing."""
    return WeatherResponse.model_validate({
        "latitude": 35.0,
        "longitude": -80.0,
        "timezone": "America/New_York",
        "elevation": 100.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00"],
            "temperature_2m": [10.0, 12.0],
            "precipitation": [0.1, 0.2],
        },
    })


def test_readings_count_and_order():
    readings = to_readings(make_response())

    assert len(readings) == 2
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T01:00"


def test_reading_values_match_input():
    readings = to_readings(make_response())

    assert readings[1].temperature_c == 12.0
    assert readings[1].precipitation_mm == 0.2


def test_identical_readings_are_equal():
    reading1 = HourlyReading("2026-04-08T00:00", 10.0, 0.1)
    reading2 = HourlyReading("2026-04-08T00:00", 10.0, 0.1)

    assert reading1 == reading2