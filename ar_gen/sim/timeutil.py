"""Time helpers. Internally the simulator uses integer milliseconds since the Unix epoch (UTC)."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

EPOCH = datetime(1970, 1, 1)
SEC_MS = 1_000
MIN_MS = 60_000
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
INTERVALS_PER_YEAR = 35_040


def dt_to_ms(dt: datetime) -> int:
    """Naive datetimes are treated as UTC."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return int(round((dt - EPOCH).total_seconds() * 1000))


def ms_to_dt(ms: int) -> datetime:
    """Naive UTC datetime with millisecond precision."""
    return EPOCH + timedelta(milliseconds=int(ms))


def floor_ms(ms: int, step: int) -> int:
    return ms - ms % step


@lru_cache(maxsize=8)
def zone(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def to_local(ms: int, tz: str) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=zone(tz))


def local_date(ms: int, tz: str) -> date:
    return to_local(ms, tz).date()


def local_to_ms(d: date, hour: float, tz: str) -> int:
    """Milliseconds for a local date + fractional hour."""
    h = int(hour)
    m = int(round((hour - h) * 60))
    if m == 60:
        h, m = h + 1, 0
    local = datetime(d.year, d.month, d.day, tzinfo=zone(tz)) + timedelta(hours=h, minutes=m)
    return dt_to_ms(local)


def date_str(d: date) -> str:
    return d.isoformat()


def parse_date(s: str) -> date:
    return date.fromisoformat(s)
