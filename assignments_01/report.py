import json
from pathlib import Path

from weatherkit import WeatherResponse, to_readings, DailyAggregator


def main():
    """Load, validate, convert, summarize, and print weather data."""
    file_path = Path(__file__).parent / "weather_raw.json"

    with open(file_path, encoding="utf-8") as file:
        data = json.load(file)

    response = WeatherResponse.model_validate(data)

    print("Latitude:", response.latitude)
    print("Timezone:", response.timezone)
    print("Hourly observations:", len(response.hourly.time))

    readings = to_readings(response)
    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)

    print("\nDate         High   Low   Precipitation   Range")
    for day in summaries:
        print(
            f"{day.date}  {day.temp_max:5.1f}  {day.temp_min:5.1f}"
            f"  {day.precipitation_sum:12.1f}  {day.temp_range():5.1f}"
        )

    incomplete = aggregator.incomplete_days(readings)
    if incomplete:
        print("Warning: incomplete days dropped:", ", ".join(incomplete))


# Without this guard, importing report.py would run the whole pipeline.
if __name__ == "__main__":
    main()


# Reflection:
# 1. Rejecting null temperatures is useful when complete data is required.
#    For partial data, temperature_2m could allow None, using list[float | None],
#    and the conversion and aggregation code would need to handle missing values.
#
# 2. At noon, a day has only about 12 hours, so min_hours=24 drops it.
#    incomplete_days() tells us which dates were dropped.
#
# 3. A package makes it easier for another pipeline to import and reuse
#    the validation, conversion, and aggregation code separately.