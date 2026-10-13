# --- Classes ---

#Q1
class Thermometer:
    def __init__(self, location, readings=None):
        self.location = location

        if readings is None:
            self.readings = []
        else:
            self.readings = readings

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if len(self.readings) == 0:
            return None

        total = sum(self.readings)
        count = len(self.readings)
        return total / count

    def hottest(self):
        if len(self.readings) == 0:
            return None

        return max(self.readings)

    def __repr__(self):
        return (
            f"Thermometer(location='{self.location}', "
            f"n_readings={len(self.readings)}, "
            f"average={self.average()})"
        )


# Create a thermometer for Georgetown, TX
thermometer = Thermometer("Georgetown")

# Add four temperature readings
thermometer.add(85)
thermometer.add(100)
thermometer.add(95)
thermometer.add(78)

# Print the average and hottest temperatures
print("Average temperature:", thermometer.average())
print("Hottest temperature:", thermometer.hottest())

# average() must handle an empty list because dividing by zero causes a ZeroDivisionError. Returning None avoids this error.

#Q2
#Print the thermometer object
print(thermometer)

thermometer2 = Thermometer("Dallas", [90, 95])
print([thermometer, thermometer2])

# Without __repr__, Python displays a default representation that usually includes the class name and memory address.
# This is not helpful for debugging because it does not show the thermometer's location or temperature readings.


# Q3
class TemperatureAlert:
    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        alert_readings = []

        for reading in thermometer.readings:
            if reading > self.threshold:
                alert_readings.append(reading)

        return alert_readings

alert1 = TemperatureAlert(90.0)
alert2 = TemperatureAlert(80.0)

print("Readings above 90:", alert1.breaches(thermometer))
print("Readings above 80:", alert2.breaches(thermometer))

# The threshold is stored in TemperatureAlert so each alert remembers its own threshold. If we have twenty thermometers, we can reuse the
# same alert object to check all of them without passing the threshold every time we call breaches().


# --- Dataclass ---

#Q1

from dataclasses import dataclass


@dataclass
class Station:
    """Represents a weather station and its geographic location."""

    station_id: int
    name: str
    latitude: float
    longitude: float
    elevation: float


# Create two stations with identical values
station_a = Station(1, "Austin", 30.2672, -97.7431, 149.0)
station_b = Station(1, "Austin", 30.2672, -97.7431, 149.0)

print(station_a == station_b)

# The result is True because a dataclass compares the values of its fields.
# With the original class, the result would be False because the two separately created objects would not be equal by default.

#Q2

from dataclasses import FrozenInstanceError

@dataclass(frozen=True)
class FrozenStation:
    station_id: int
    name: str
    latitude: float
    longitude: float
    elevation: float


# Create three stations, where two are identical
station_a = FrozenStation(1, "Austin", 30.2672, -97.7431, 149.0)
station_b = FrozenStation(2, "Houston", 29.7604, -95.3698, 15.0)
station_c = FrozenStation(2, "Houston", 29.7604, -95.3698, 15.0)

# Try to change a field
try:
    station_a.name = "Dallas"
except FrozenInstanceError as error:
    print("Error:", error)

# Put all three objects in a set
stations = {station_a, station_b, station_c}
print("Number of stations:", len(stations))

# frozen=True prevents field changes and makes objects hashable,
# so they can be stored in a set. Identical objects count only once.

from dataclasses import dataclass, field


# First attempt (intentionally incorrect):
# stations: list[FrozenStation] = []
#
# Error:
# ValueError: mutable default <class 'list'> for field stations
# is not allowed: use default_factory
#
# Python refuses this because a list is mutable. If multiple
# StationBatch objects shared the same default list, adding a
# station to one batch could affect another batch.


@dataclass
class StationBatch:
    """Represents a group of weather stations in one region."""

    region: str
    stations: list[FrozenStation] = field(default_factory=list)

    def add(self, station: FrozenStation) -> None:
        """Add a station to the batch."""
        self.stations.append(station)

    def highest(self) -> FrozenStation | None:
        """Return the station with the greatest elevation, or None if empty."""
        if len(self.stations) == 0:
            return None

        return max(self.stations, key=lambda station: station.elevation)


# Create a batch and add stations
batch = StationBatch("Texas")
batch.add(station_a)
batch.add(station_b)

print(batch.highest())

# --- Pydantic ---

#Q1

from pydantic import BaseModel, Field, model_validator

class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def check_sensor(self):
        """Reject a combination that indicates a failed sensor."""
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("Possible failed sensor")
        return self

# Create one valid Reading
reading = Reading(
    station_id="ST001",
    timestamp="2026-10-12 10:00:00",
    temperature_c=25.5,
    humidity=60.0
)

# Print the reading
print(reading)


#Q2

from pydantic import ValidationError

# 1. Missing required field
try:
    reading1 = Reading(
        station_id="ST001",
        temperature_c=21.5,
        humidity=40
    )
except ValidationError as error:
    print("Error 1 - Missing field:")
    print(error)

# 2. Temperature is too high
try:
    reading2 = Reading(
        station_id="ST001",
        timestamp="2026-10-12 10:00:00",
        temperature_c=150.0,
        humidity=40
    )
except ValidationError as error:
    print("Error 2 - Invalid temperature:")
    print(error)

# 3. Humidity is not a valid number
try:
    reading3 = Reading(
        station_id="ST001",
        timestamp="2026-10-12 10:00:00",
        temperature_c=21.5,
        humidity="very humid"
    )
except ValidationError as error:
    print("Error 3 - Invalid humidity:")
    print(error)

# Pydantic converts compatible values to the required type when possible.
# "21.5" can be converted to a float, but "very humid" cannot.
# Values must be convertible to the field's type and satisfy its constraints.

# Convert compatible input values
reading4 = Reading(
    station_id="ST001",
    timestamp="2026-10-12 10:00:00",
    temperature_c="21.5",
    humidity=40
)

print("Valid reading:")
print(reading4)

print("Temperature type:", type(reading4.temperature_c))
print("Humidity type:", type(reading4.humidity))

#Q3

try:
    reading5 = Reading(
        station_id="AB",
        temperature_c="hot",
        humidity=40
    )
except ValidationError as e:
    for error in e.errors():
        print("Location:", error["loc"])
        print("Message:", error["msg"])
        print()

# Three errors are reported: station_id is too short, timestamp is missing,
# and temperature_c is not a valid number. Reporting all errors at once
# helps us fix multiple problems without running the code repeatedly.

# Q4

valid_reading = Reading(
    station_id="ST001",
    timestamp="2026-10-12 10:00:00",
    temperature_c=20.0,
    humidity=50.0
)
print("Valid reading:", valid_reading)

# This combination indicates a failed sensor
try:
    bad_reading = Reading(
        station_id="ST001",
        timestamp="2026-10-12 10:00:00",
        temperature_c=-45.0,
        humidity=0.0
    )
except ValidationError as e:
    print("Invalid reading:")
    print(e)

# Field constraints check individual values. This rule depends on the
# relationship between humidity and temperature, so we need a model validator.


# --- pytest ---

#Q1

import pytest


def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return celsius * 9 / 5 + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212

    # Floating-point calculations can produce tiny precision differences,
    # so pytest.approx checks whether the result is sufficiently close.
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

#Q2

def mean(values: list[float]) -> float:
    """Calculate the mean of a list of numbers."""
    if len(values) == 0:
        raise ValueError("List is empty")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

    # pytest.raises(ValueError) checks the error type,
    # but match= also checks the error message.


#Q3

@pytest.mark.parametrize("values, expected", [
    ([2, 4], 3),
    ([5], 5),
    ([-2, 2], 0),
    ([1, 2, 3], 2),
])
def test_mean_values(values, expected):
    assert mean(values) == expected

# Summary: 6 passed (1 test from Q1, 1 from Q2, and 4 from Q3).
# Parametrize avoids writing four nearly identical test functions.


#Q4

# Failure output when I changed 9 / 5 to 9 / 4:
# E       assert 257.0 == 212
# 1 failed, 5 passed in 0.12s
#
# Pytest showed the expected value (212) and actual value (257.0).
# This is more useful than "assertion failed" because I can see exactly which values do not match.
