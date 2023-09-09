from datetime import date

import pytest

from invest_ai.utils.string import date_to_str, str_to_date


def test_str_to_date():
    assert str_to_date("2023-09-09") == date(2023, 9, 9)
    assert str_to_date("2000-01-01") == date(2000, 1, 1)
    with pytest.raises(ValueError):
        str_to_date("01-01-2000")


def test_date_to_str():
    assert date_to_str(date(2023, 9, 9)) == "2023-09-09"
    assert date_to_str(date(2000, 1, 1)) == "2000-01-01"


def test_str_date_conversion():
    """
    Test the conversion from string to datetime.date and back, as well as from datetime.date to string and back,
    to ensure consistency and accuracy.
    """
    # Define a sample date string and datetime.date object
    sample_str = "2023-09-09"
    sample_date = date(2023, 9, 9)

    # Convert string to date and back to string
    converted_date = str_to_date(sample_str)
    converted_str_back = date_to_str(converted_date)

    # Convert date to string and back to date
    converted_str = date_to_str(sample_date)
    converted_date_back = str_to_date(converted_str)

    assert converted_date == sample_date, "Failed: Conversion from string to date"
    assert converted_str_back == sample_str, "Failed: Conversion from date to string"
    assert (
        converted_str == sample_str
    ), "Failed: Conversion from date to string and back to date"
    assert (
        converted_date_back == sample_date
    ), "Failed: Conversion from string to date and back to string"
