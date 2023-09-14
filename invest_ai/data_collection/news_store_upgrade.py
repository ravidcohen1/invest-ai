# Implementation of the NewsStore class based on the provided requirements and code snippets

import datetime
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from shutil import copy2
from typing import Optional, Union

import pandas as pd
import pytz
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

    def __init__(self, data_path: Optional[Path] = cfg.NEW_DATA_PATH, backup=True):
        self.data_path = data_path
        self.df = self._load_data(backup)

    def _load_data(self, backup) -> pd.DataFrame:
        """Load data from the pkl file into a DataFrame."""
        if os.path.exists(self.data_path):
            if backup:
                self._backup_data()
            df = pd.read_pickle(self.data_path)
            print(f"Data loaded. DataFrame shape: {df.shape}")
            return df
        else:
            return pd.DataFrame(
                columns=["title", "article", "time_utc", "time_et", "date_utc", "date"]
            )

    def _save_data(self) -> None:
        """Save the DataFrame to a pkl file."""
        self.df.to_pickle(self.data_path, index=False)
        print(f"Data saved to {self.data_path}. DataFrame shape: {self.df.shape}")

    def _backup_data(self) -> None:
        """Create a backup of the current pkl file."""
        data_folder = self.data_path.parent
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        backup_filename = data_folder / f"news_store_backup_{timestamp}.pkl"

        copy2(self.data_path, backup_filename)

    @staticmethod
    def _validate_inputs(start_date: datetime.date, end_date: datetime.date):
        assert isinstance(start_date, datetime.date)
        assert isinstance(end_date, datetime.date)
        assert end_date > start_date

    def get_news_for_dates(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        fetch_missing_dates: bool = False,
        drop_articles: bool = False,
    ) -> pd.DataFrame:
        """Fetch or retrieve news for a date range."""

        self._validate_inputs(start_date, end_date)

        # Filter existing records for the date range
        result_df = self.df[
            (self.df["date"] >= start_date) & (self.df["date"] <= end_date)
        ]

        if not fetch_missing_dates:
            if drop_articles:
                result_df.drop(columns=["article"], inplace=True)
            return result_df

        # Identify missing dates and fetch records for them
        existing_dates = set(result_df["date"])
        all_dates = {
            start_date + timedelta(days=i)
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
            (self.df["date"] >= start_date) & (self.df["date"] <= end_date)
        ]
        if drop_articles:
            result_df.drop(columns=["article"], inplace=True)
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
    def _get_urls(date: datetime.date) -> pd.DataFrame:
        """Fetch URLs of articles for a given date."""
        url = f"https://techcrunch.com/{date.strftime('%Y/%m/%d')}"
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
        """
        Fetch the title, article content, and publication time for a given URL.

        :param url: The URL of the webpage to scrape.
        :type url: str
        :return: A Pandas Series containing the title, article, and various time and date formats.
        :rtype: pd.Series
        """
        try:
            content = requests.get(url).text
            with open("content.txt", "w", encoding="utf-8") as file:
                file.write(content)
            soup = BeautifulSoup(content, features="html.parser")
        except Exception as e:
            print(str(e))
            return pd.Series(
                {
                    "title": None,
                    "article": None,
                    "time_utc": None,
                    "time_et": None,
                    "date_utc": None,
                    "date": None,
                }
            )

        try:
            article_div = soup.find("div", {"class": "article-content"})
            article = " ".join(
                [p.text for p in article_div.find_all("p", recursive=False)]
            )
        except Exception as e:
            print(str(e))
            article = None

        try:
            title = soup.find("h1").text
        except Exception as e:
            print(str(e))
            title = None

        try:
            pub_time_meta = soup.find("meta", {"property": "article:published_time"})
            pub_time_utc_str = pub_time_meta["content"]
            time_utc = datetime.datetime.fromisoformat(pub_time_utc_str)
            time_et = time_utc.astimezone(pytz.timezone("US/Eastern"))

            date_utc = datetime.date(time_utc.year, time_utc.month, time_utc.day)
            date_et = datetime.date(time_et.year, time_et.month, time_et.day)

        except Exception as e:
            print(str(e))
            time_utc = None
            time_et = None
            date_utc = None
            date_et = None

        return pd.Series(
            {
                "title": title,
                "article": article,
                "time_utc": time_utc,
                "time_et": time_et,
                "date_utc": date_utc,
                "date": date_et,
            }
        )

    # Example usage (Replace with actual URL)
    # result = _get_content_for_url("https://example.com")
    # print(result)


if __name__ == "__main__":
    """
    todo
    fix tests
    change date col to str to be backward compatible

    """
    news_store = NewsStore(
        data_path=Path(
            "/Users/user/PycharmProjects/invest-ai/data/news_store_with_time.pkl"
        )
    )
    news_store.get_news_for_dates("2007-01-01", "2007-01-05", fetch_missing_dates=True)
