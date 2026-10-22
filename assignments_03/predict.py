import json
import joblib
import pandas as pd

# Task 7: Load and Predict

# Load the saved model and metadata.
model = joblib.load("models/weather_classifier.pkl")

with open("models/weather_classifier_metadata.json", "r") as metadata_file:
    metadata = json.load(metadata_file)

print("City:", metadata["city"])
print("Features:", metadata["feature_names"])
print("Test accuracy:", round(metadata["test_accuracy"], 3))

# Create four hypothetical weather days.
weather_days = pd.DataFrame([
    {
        "temperature_2m_max": 22.0,
        "temperature_2m_min": 12.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 10.0,
    },
    {
        "temperature_2m_max": 38.0,
        "temperature_2m_min": 25.0,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 15.0,
    },
    {
        "temperature_2m_max": 28.0,
        "temperature_2m_min": 18.0,
        "precipitation_sum": 1.0,
        "wind_speed_10m_max": 20.0,
    },
    {
        "temperature_2m_max": 24.0,
        "temperature_2m_min": 10.0,
        "precipitation_sum": 2.5,
        "wind_speed_10m_max": 29.0,
    },
])


weather_days = weather_days[metadata["feature_names"]]
predictions = model.predict(weather_days)
good_probabilities = model.predict_proba(weather_days)[:, 1]

# Print the inputs, prediction, and confidence for each day.
for index in range(len(weather_days)):
    if predictions[index] == 1:
        label = "good"
    else:
        label = "skip"

    print(f"\nDay {index + 1}:")
    print(weather_days.iloc[index].to_dict())
    print("Prediction:", label)
    print(f"Confidence of 'good': {good_probabilities[index]:.1%}")


# Task 8: Reflect

# 1. Day 4 was my borderline day. The model predicted "good" with 80% confidence. 
#    I would consider this less certain than the predictions with 100% confidence. 
#    If the model predicted 0.52, I would consider it uncertain because the probability is close
#    to 50%, and I would check the weather conditions before deciding.

# 2. If predict.py ran before train_classifier.py, the saved model and metadata files might not exist, causing a FileNotFoundError.   
#    I could make the error message more helpful by checking whether the files exist and explaining that train_classifier.py must
#    be run first.

# 3. For a cloud pipeline, predict.py would need to receive new weather data automatically instead of using manually entered hypothetical data.
   