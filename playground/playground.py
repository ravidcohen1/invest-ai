# import yfinance as yf


def get_stock_prices(tickers, start_date, end_date):
    """

    :param tickers: str, list
    :param start_date: YYYY-MM-DD
    :param end_date: YYYY-MM-DD
    :return:
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    return data


# stock_symbols = ["GOOGL", "MSFT", "AAPL", "AMZN", "TSLA", "FB", "NFLX", "BABA", "JNJ", "JPM", "V", "PG", "BAC", "INTC", "KO", "MCD", "NKE", "T", "PFE", "ORCL"]
#
#
# df=get_stock_prices(stock_symbols, "1980-01-01", "2023-05-01")
# print(df)
#
# from matplotlib import pyplot as plt
# for s in stock_symbols:
#     plt.plot(df['Close'][s], label=s)
# plt.legend()
# plt.show()


from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm, trange


def get_urls(date):
    url = "https://techcrunch.com/" + date.strftime("%Y/%m/%d")
    content = requests.get(url).text
    return [
        a["href"]
        for a in BeautifulSoup(content).find_all(
            "a", {"class": "post-block__title__link"}
        )
    ]


def get_title(url):
    content = requests.get(url).text
    soup = BeautifulSoup(content, "html.parser")

    # Extract title
    try:
        title = soup.find("h1").text  # assuming the title is in an h1 tag
    except:
        title = ""

    return title


def get_article(url):
    content = requests.get(url).text
    soup = BeautifulSoup(content, features="html.parser")

    # Extract article content
    try:
        article_div = soup.find("div", {"class": "article-content"})
        article = [p.text for p in article_div.find_all("p", recursive=False)]
    except:
        article = []

    # Extract title
    try:
        title = soup.find("h1").text  # assuming the title is in an h1 tag
    except:
        title = ""

    return article, title


# Initialize empty lists to store data
all_urls = []
all_dates = (
    []
)  # Loop through the past 7 days to gather URLs and their corresponding dates
all_titles = []
for i in trange(10):
    date = datetime.now() - timedelta(days=i)
    urls_for_date = get_urls(date)
    titles_for_date = [get_title(url) for url in urls_for_date]
    for url in urls_for_date:
        all_urls.append(url)
        all_dates.append(date.strftime("%Y/%m/%d"))
    all_titles += titles_for_date

    if i % 100 == 0:
        df = pd.DataFrame(
            {
                "title": all_titles,
                "date": all_dates,
                "url": all_urls,
                # "article": articles
            }
        )
        df.to_csv("data/titles.csv", index=False)

df = pd.DataFrame(
    {
        "title": all_titles,
        "date": all_dates,
        "url": all_urls,
        # "article": articles
    }
)
df.to_csv("data/titles.csv", index=False)

# Fetch articles and titles for the gathered URLs
# articles = []
# titles = []
# for url in tqdm(all_urls):
#     article, title = get_article(url)
#     articles.append(article)
#     titles.append(title)


df.groupby("date").url.count().sort_values().plot()
plt.show()
exit()
# def get_article(url):
#     content = requests.get(url).text
#     article = BeautifulSoup(content).find_all("div", {"class": "article-content"})[0]
#     return [p.text for p in article.find_all("p", recursive=False)]
# urls = sum([get_urls(datetime.now() - timedelta(days=i)) for i in trange(7)], [])
# articles = pd.DataFrame({"url": urls, "article": [get_article(url) for url in tqdm(urls)]})
#
# paragraphs = articles.explode("article").rename(columns={"article": "paragraph"})
# paragraphs = paragraphs[paragraphs["paragraph"].str.split().map(len) > 10]

with open("api_key", "r") as f:
    openai.api_key = f.read().strip()


def get_embedding(texts, model="text-embedding-ada-002"):
    texts = [text.replace("\n", " ") for text in texts]
    return [
        res["embedding"]
        for res in openai.Embedding.create(input=texts, model=model)["data"]
    ]


batch_size = 100
embeddings = []

for i in trange(0, len(paragraphs), batch_size):
    embeddings += get_embedding(paragraphs.iloc[i : i + batch_size]["paragraph"])

paragraphs["embedding"] = embeddings


def find_similar_paragraph(query):
    query_embedding = get_embedding([query])[0]

    best_idx = (
        paragraphs["embedding"]
        .map(
            lambda emb: np.dot(emb, query_embedding)
            / (np.linalg.norm(emb) * np.linalg.norm(query_embedding))
        )
        .argmax()
    )

    best_paragraph = paragraphs.iloc[best_idx]["paragraph"]
    return best_paragraph
