from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")  # NBA game dates are US Eastern calendar dates


def eastern_today(now: datetime | None = None) -> date:
    return (now or datetime.now(UTC)).astimezone(EASTERN).date()
