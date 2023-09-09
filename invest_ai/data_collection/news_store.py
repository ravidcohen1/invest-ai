# Implementation of the NewsStore class based on the provided requirements and code snippets

import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from shutil import copy2
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from invest_ai import configs as cfg
from invest_ai.utils.string import format_time, validate_date_format

tqdm.pandas()


class NewsStore:
    """
    A class to manage the storage and retrieval of news articles and their metadata.
    """

    def __init__(self, csv_file_path: Optional[Path] = cfg.NEW_DATA_PATH, backup=True):
        self.csv_file_path = csv_file_path
        self.df = self._load_data(backup)

    def _load_data(self, backup) -> pd.DataFrame:
        """Load data from the CSV file into a DataFrame."""
        if os.path.exists(self.csv_file_path):
            if backup:
                self._backup_data()
            return pd.read_csv(self.csv_file_path)
        else:
            return pd.DataFrame(columns=["date", "url", "source", "title", "article"])

    def _save_data(self) -> None:
        """Save the DataFrame to a CSV file."""
        self.df.to_csv(self.csv_file_path, index=False)

    def _backup_data(self) -> None:
        """Create a backup of the current CSV file."""
        data_folder = self.csv_file_path.parent
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_filename = data_folder / f"news_store_backup_{timestamp}.csv"

        copy2(self.csv_file_path, backup_filename)

    def _validate_inputs(self, start_date_str: str, end_date_str: str):
        validate_date_format(start_date_str)
        validate_date_format(end_date_str)
        assert end_date_str > start_date_str

    def get_news_for_dates(
        self, start_date_str: str, end_date_str: str, fetch_missing_dates: bool = False
    ) -> pd.DataFrame:
        """Fetch or retrieve news for a date range."""
        self._validate_inputs(start_date_str, end_date_str)

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

        # Filter existing records for the date range

        result_df = self.df[
            (self.df["date"] >= start_date_str) & (self.df["date"] <= end_date_str)
        ]
        if not fetch_missing_dates:
            return result_df

        # Identify missing dates and fetch records for them
        existing_dates = set(result_df["date"])
        all_dates = {
            str(start_date + timedelta(days=i))
            for i in range((end_date - start_date).days + 1)
        }
        missing_dates = sorted(
            all_dates - existing_dates, reverse=True
        )  # Most recent to oldest
        print(f"{len(missing_dates)} days missing")
        # Initialize progress tracking variables
        start_time = time.time()
        total_fetched = 0

        for day_idx, date in enumerate(missing_dates):
            print(f"Fetching data for: {date}")
            daily_df = self._get_urls(date)

            daily_content = self.fetch_content_in_parallel(
                daily_df["url"]
            )  # .progress_apply(self._get_content_for_url)
            daily_df = pd.concat([daily_df, daily_content], axis=1)
            daily_df["date"] = date
            total_fetched += len(daily_df)
            if len(daily_df) > 0:
                self.df = pd.concat([self.df, daily_df])

            # Save data and print progress
            self._save_data()

            elapsed_time = time.time() - start_time
            avg_time_per_day = elapsed_time / (day_idx + 1)
            estimated_completion_time = avg_time_per_day * (
                len(missing_dates) - (day_idx + 1)
            )

            print(
                f"Articles collected for the day: {len(daily_df)}, Total collected: {total_fetched}, Total store size: {len(self.df)}, Elapsed time: {format_time(elapsed_time)}, Estimated completion time: {format_time(estimated_completion_time)}"
            )

        result_df = self.df[
            (self.df["date"] >= start_date_str) & (self.df["date"] <= end_date_str)
        ]
        return result_df

    def fetch_content_in_parallel(
        self, urls: pd.Series, num_workers: int = 4
    ) -> pd.DataFrame:
        """
        Fetch content for a list of URLs in parallel.

        :param urls: A Pandas Series of URLs to fetch content for.
        :param num_workers: Number of worker threads for parallel execution.
        :return: A Pandas Series containing the fetched content for each URL.
        """
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self._get_content_for_url, urls))
        return pd.DataFrame(results)

    @staticmethod
    def _get_urls(date: str) -> pd.DataFrame:
        """Fetch URLs of articles for a given date."""
        date_obj = datetime.strptime(date, "%Y-%m-%d").date()
        url = f"https://techcrunch.com/{date_obj.strftime('%Y/%m/%d')}"
        content = requests.get(url, timeout=120).text
        urls = [
            a["href"]
            for a in BeautifulSoup(content, features="html.parser").find_all(
                "a", {"class": "post-block__title__link"}
            )
        ]
        return pd.DataFrame({"url": urls, "source": ["TechCrunch"] * len(urls)})

    @staticmethod
    def _get_content_for_url(url: str) -> pd.Series:
        """Fetch the title and article content for a given URL."""
        try:
            content = requests.get(url).text
            soup = BeautifulSoup(content, features="html.parser")
        except Exception as e:
            print(str(e))
            return pd.Series({"title": None, "article": None})
        try:
            article_div = soup.find("div", {"class": "article-content"})
            article = " ".join(
                [p.text for p in article_div.find_all("p", recursive=False)]
            )
        except Exception as e:
            print(str(e))
            article = None

        # Extract title
        try:
            title = soup.find("h1").text  # assuming the title is in an h1 tag
        except Exception as e:
            print(str(e))
            title = None

        return pd.Series({"title": title, "article": article})


if __name__ == "__main__":
    news_store = NewsStore()
    news_store.get_news_for_dates("2000-01-01", "2023-09-01", fetch_missing_dates=True)
