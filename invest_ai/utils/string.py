import re
from datetime import timedelta


def format_time(seconds: int) -> str:
    """Convert time from seconds to a readable string format."""
    delta = timedelta(seconds=round(seconds))
    days, remainder = divmod(delta.total_seconds(), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_str = ""
    if days > 0:
        time_str += f"{int(days)}d "
    if hours > 0:
        time_str += f"{int(hours)}h "
    if minutes > 0:
        time_str += f"{int(minutes)}m "
    time_str += f"{int(seconds)}s"

    return time_str


def validate_date_format(date_str: str) -> None:
    """Validate if the given date string is in the format 'YYYY-MM-DD'."""
    assert isinstance(date_str, str)
    pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    assert pattern.fullmatch(
        date_str
    ), f"The date {date_str} is not in the 'YYYY-MM-DD' format."
