# Task 5

from weather_model import WeatherClassifier


def main() -> None:
    """Load the saved weather model and predict weather conditions."""

    classifier = WeatherClassifier("models/weather_classifier.pkl")

    # Four hypothetical days: good, bad, and borderline conditions.
    days = [
        {
            "temperature_2m_max": 24,
            "temperature_2m_min": 15,
            "precipitation_sum": 0,
            "wind_speed_10m_max": 5,
        },
        {
            "temperature_2m_max": 8,
            "temperature_2m_min": 2,
            "precipitation_sum": 20,
            "wind_speed_10m_max": 30,
        },
        {
            "temperature_2m_max": 19,
            "temperature_2m_min": 12,
            "precipitation_sum": 2,
            "wind_speed_10m_max": 12,
        },
        {
            "temperature_2m_max": 28,
            "temperature_2m_min": 20,
            "precipitation_sum": 5,
            "wind_speed_10m_max": 18,
        },
    ]

    predictions = classifier.predict(days)

    for day, prediction in zip(days, predictions):
        print(
            f"Weather inputs: {day} | "
            f"Prediction: {prediction.label} | "
            f"Probability: {prediction.probability:.2%}"
        )


# Without this guard, importing predict_weather.py from another module would execute main() immediately,
# causing the predictions to run even when the other module only wanted to import and reuse a function.

if __name__ == "__main__":
    main()


# Task 6: Reflection

# 1. Fixed feature order:
# Selecting features in a fixed order prevents the model from receiving values in the wrong columns.
# For example, if temperature and wind speed were swapped, the model could interpret wind speed as temperature,
# producing incorrect predictions.

# 2. Testing with a tiny model:
# A small model created in a test fixture makes tests independent of the real saved model file. 
# Tests are more predictable, faster, and easier to reproduce because they create their own test data and model.


# 3. Preparing for the Week 10 pipeline:
# WeatherClassifier accepts batches of weather records and returns predictions in the same order, making it useful for a database pipeline.
# Before production, I would add logging and error handling to make it easier to monitor prediction failures and unexpected input data.
