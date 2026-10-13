from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """Hourly weather measurements stored in parallel lists."""

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def check_lengths(self):
        """Make sure all lists have the same length."""
        if not (
            len(self.time)
            == len(self.temperature_2m)
            == len(self.precipitation)
        ):
            raise ValueError("All hourly lists must have the same length")
        return self


class WeatherResponse(BaseModel):
    """Weather API response with location information and hourly data."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock