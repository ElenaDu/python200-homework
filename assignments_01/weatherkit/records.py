from dataclasses import dataclass
from weatherkit.schemas import WeatherResponse

@dataclass
class HourlyReading:
    """One reading with temperature in Celsius and precipitation in millimeters."""

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert hourly API lists into individual readings.

    Args:
        response: A validated weather API response.

    Returns:
        A list of hourly readings in their original order.
    """
    readings = []

    for i in range(len(response.hourly.time)):
        reading = HourlyReading(
            timestamp=response.hourly.time[i],
            temperature_c=response.hourly.temperature_2m[i],
            precipitation_mm=response.hourly.precipitation[i],
        )
        readings.append(reading)

    return readings


# WeatherResponse validates untrusted API data at the boundary.
# HourlyReading is a simple dataclass for data already validated.