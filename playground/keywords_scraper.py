from datetime import datetime

import requests
from bs4 import BeautifulSoup


def get_news_titles(keyword, start_date, end_date, websites=None):
    base_url = "https://www.google.com/search?q={keyword}+site:{sites}&tbs=cdr:1,cd_min:{start},cd_max:{end}&tbm=nws"

    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Prepare the sites string
    sites = "+OR+site:".join(websites) if websites else ""

    # Prepare the url
    url = base_url.format(
        keyword=keyword,
        sites=sites,
        start=start_date.strftime("%m/%d/%Y"),
        end=end_date.strftime("%m/%d/%Y"),
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Failed to retrieve the webpage.")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    # Extract titles
    articles = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
    titles = [article.get_text() for article in articles]

    return titles


websites = ["bbc.co.uk", "cnn.com", "nytimes.com"]
titles = get_news_titles("climate change", "2022-01-01", "2022-12-31", websites)
print(titles)
