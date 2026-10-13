import pytest
from weatherkit import HourlyReading, DailyAggregator


@pytest.fixture
def readings():
    """Return readings covering two dates, including one incomplete day."""
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.1),
        HourlyReading("2026-04-08T01:00", 14.0, 0.2),
        HourlyReading("2026-04-09T00:00", -2.0, 0.3),
        HourlyReading("2026-04-09T01:00", 4.0, 0.4),
    ]


def test_grouping_and_temperatures(readings):
    summaries = DailyAggregator(min_hours=1).summarize(readings)

    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-08"
    assert summaries[0].temp_max == 14.0
    assert summaries[0].temp_min == 10.0


def test_precipitation_sum(readings):
    summaries = DailyAggregator(min_hours=1).summarize(readings)

    assert summaries[0].precipitation_sum == pytest.approx(0.3)


def test_incomplete_day_is_dropped(readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(readings)

    assert summaries == []
    assert aggregator.incomplete_days(readings) == [
        "2026-04-08",
        "2026-04-09",
    ]


def test_lower_min_hours_keeps_day(readings):
    summaries = DailyAggregator(min_hours=2).summarize(readings)

    assert len(summaries) == 2


@pytest.mark.parametrize(
    "minimum, expected_count",
    [(1, 2), (2, 2), (3, 0)],
)
def test_min_hours(minimum, expected_count, readings):
    summaries = DailyAggregator(min_hours=minimum).summarize(readings)

    assert len(summaries) == expected_count


# The test_grouping_and_temperatures test caught the bug
# when max() was accidentally changed to min().
# FAILED tests/test_summarize.py::test_grouping_and_temperatures - AssertionError: assert 10.0 == 14.0