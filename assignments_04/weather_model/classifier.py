from dataclasses import dataclass
from pathlib import Path
import joblib


# Task 1
@dataclass
class Prediction:
    """
    Represents a weather prediction for one day.

    Attributes:
        label: Either "good" or "skip".
        probability: Probability from 0 to 1 that the day is good for running.
    """

    label: str
    probability: float


# Task 2

class WeatherClassifier:
    """
    Loads a trained weather classification model and predicts
    whether days are good for running.
    """

    FEATURES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    def __init__(self, model_path: str) -> None:
        """
        Load the trained model from a .pkl file.

        Args:
            model_path: Path to the saved model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = Path(model_path)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        self.model = joblib.load(model_path)

    def predict(self, days: list[dict]) -> list[Prediction]:
        """
        Predict whether each day is good for running.

        Args:
            days: A list of dictionaries containing weather features.

        Returns:
            A list of Prediction objects in the same order as the input.

        Raises:
            ValueError: If any required weather features are missing.
        """
        if not days:
            return []

        for day in days:
            missing_features = [
                feature
                for feature in self.FEATURES
                if feature not in day
            ]

            if missing_features:
                raise ValueError(
                    f"Missing required features: {', '.join(missing_features)}"
                )

        # Select features in the required fixed order.
        features = [
            [day[feature] for feature in self.FEATURES]
            for day in days
        ]

        labels = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]

        predictions = []

        for label, probability in zip(labels, probabilities):
            prediction_label = "good" if label == 1 else "skip"

            predictions.append(
                Prediction(
                    label=prediction_label,
                    probability=float(probability),
                )
            )

        return predictions

