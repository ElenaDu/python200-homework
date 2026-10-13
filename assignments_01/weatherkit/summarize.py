from dataclasses import dataclass
from weatherkit.records import HourlyReading


@dataclass
class DailySummary:
    """Daily weather summary with temperatures in Celsius and precipitation in mm."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between maximum and minimum temperatures."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Group hourly readings into daily summaries."""

    def __init__(self, min_hours: int = 24):
        """Set the minimum number of readings required for a day."""
        self.min_hours = min_hours

    def summarize(
        self, readings: list[HourlyReading]
    ) -> list[DailySummary]:
        """Create daily summaries and skip days with too few readings."""
        days = {}

        for reading in readings:
            date = reading.timestamp[:10]

            if date not in days:
                days[date] = []

            days[date].append(reading)

        summaries = []

        for date in sorted(days):
            day_readings = days[date]

            if len(day_readings) < self.min_hours:
                continue

            temperatures = [r.temperature_c for r in day_readings]
            precipitation = [r.precipitation_mm for r in day_readings]

            summary = DailySummary(
                date=date,
                temp_max=max(temperatures),
                temp_min=min(temperatures),
                precipitation_sum=sum(precipitation),
                hours_observed=len(day_readings),
            )
            summaries.append(summary)

        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return dates with fewer than the required number of readings."""
        days = {}

        for reading in readings:
            date = reading.timestamp[:10]
            days[date] = days.get(date, 0) + 1

        incomplete = []

        for date in sorted(days):
            if days[date] < self.min_hours:
                incomplete.append(date)

        return incomplete