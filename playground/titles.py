import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from invest_ai import configs as cfg


def _get_urls(date):
    """Given a date, return the list of URLs for articles from TechCrunch for that date."""
    url = "https://techcrunch.com/" + date.strftime("%Y/%m/%d")
    content = requests.get(url, timeout=120).text
    urls = [
        a["href"]
        for a in BeautifulSoup(content, features="html.parser").find_all(
            "a", {"class": "post-block__title__link"}
        )
    ]
    return urls


def _get_title(url):
    """Given a URL, fetch the title of the article."""
    try:
        content = requests.get(url).text
        soup = BeautifulSoup(content, "html.parser")
        title = soup.find("h1").text  # assuming the title is in an h1 tag
    except Exception as e:
        print(str(e))
        return None
    return title


def _save_to_csv(titles, dates, urls, file_path):
    """
    Save the data to a CSV file.

    Parameters:
    - titles (list): List of article titles.
    - dates (list): List of dates corresponding to the titles.
    - urls (list): List of URLs corresponding to the titles.
    - file_path (str): The file path where the CSV will be saved.
    """

    # Create the DataFrame
    df = pd.DataFrame({"title": titles, "date": dates, "url": urls})

    # Save the DataFrame to the specified file path
    df.to_csv(file_path, index=False)


def _load_from_interim(file_path, default_start_date):
    """
    Load data from an interim CSV file if it exists.

    Parameters:
    - file_path (str): The file path where the interim CSV might be saved.
    - default_start_date (str): The default start date in the format 'YYYY-MM-DD' to be used if interim data isn't found.

    Returns:
    - titles (list or None): List of article titles if the interim file exists, None otherwise.
    - dates (list or None): List of dates if the interim file exists, None otherwise.
    - urls (list or None): List of URLs if the interim file exists, None otherwise.
    - datetime.date: The date from which the collection should resume.
    """

    if os.path.exists(file_path):
        # Load the interim data
        interim_df = pd.read_csv(file_path)
        titles = interim_df["title"].tolist()
        dates = interim_df["date"].tolist()
        urls = interim_df["url"].tolist()
        # Set the new start date to the day after the last date in the interim data
        last_date_in_interim = datetime.strptime(dates[-1], "%Y-%m-%d")
        resume_date = last_date_in_interim + timedelta(days=1)
        return titles, dates, urls, resume_date.date()
    else:
        return (
            None,
            None,
            None,
            datetime.strptime(default_start_date, "%Y-%m-%d").date(),
        )


# Update the main function to incorporate the changes
def collect_titles_for_dates(start_date, end_date, destination_folder="/mnt/data"):
    """
    Collect article titles from TechCrunch between the given start and end dates (using dummy data).
    If an interim save exists, the function resumes from the last collected date.

    Parameters:
    - start_date (str): The starting date in the format 'YYYY-MM-DD'.
    - end_date (str): The ending date in the format 'YYYY-MM-DD'.
    - destination_folder (str): The folder path where the final CSV will be saved.

    Returns:
    - pd.DataFrame: A dataframe containing titles, dates, and URLs.
    """

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Compute the paths for interim and final files
    interim_file_path = os.path.join(
        destination_folder, f"titles_{start_date}_to_{end_date}_interim.csv"
    )
    final_file_path = os.path.join(
        destination_folder, f"titles_{start_date}_to_{end_date}.csv"
    )

    # Load data from interim file if it exists and set the new start date
    titles, dates, urls, resume_date = _load_from_interim(interim_file_path, start_date)
    if titles is None:
        titles, dates, urls = [], [], []

    # Calculate total days for progress estimation
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()
    total_days = (end_date_obj - resume_date).days + 1

    start_time = time.time()

    # Loop through each day and collect data
    for idx, single_date in enumerate(
        [resume_date + timedelta(days=i) for i in range(total_days)], start=1
    ):
        urls_for_date = _get_urls(single_date)
        date_str = single_date.strftime("%Y-%m-%d")
        titles_for_date = []
        for url in urls_for_date:
            title = _get_title(url)
            if title is not None:  # Only append if title is not None
                urls.append(url)
                dates.append(date_str)
                titles_for_date.append(title)

        titles += titles_for_date

        # Print progress info
        elapsed_time = time.time() - start_time
        avg_time_per_day = elapsed_time / idx
        estimated_completion_time = avg_time_per_day * (total_days - idx)
        print(
            f"Date: {single_date.strftime('%Y-%m-%d')}, Titles collected: {len(titles_for_date)}, Total collected: {len(titles)}, Elapsed time: {elapsed_time:.2f}s, Estimated completion time: {estimated_completion_time:.2f}s"
        )

        _save_to_csv(titles, dates, urls, interim_file_path)

    print(
        f"Titles from {start_date} to {end_date} have been saved to '{final_file_path}'"
    )

    return final_file_path


if __name__ == "__main__":
    df_demo = collect_titles_for_dates(
        "2020-01-01", "2023-09-01", destination_folder=cfg.DATA_PATH
    )
