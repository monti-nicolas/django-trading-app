import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
from transformers import pipeline

# url = f'https://finance.yahoo.com/quote/UNH?p=UNH'
# url = f'https://finance.yahoo.com/quote/UNH/news'


def load_sentiment_model():

    return pipeline('sentiment-analysis', model='ProsusAI/finbert')


def get_yahoo_finance_headlines(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}?p={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        headlines = [h.text for h in soup.select('h3')]
        cleaned = [h for h in headlines if h.strip() not in {
            'News', 'Life', 'Entertainment', 'Finance', 'Sports',
            'New on Yahoo', 'Performance Overview', ''
        }]
        return cleaned[:8]
    except Exception:
        return []

# def get_yahoo_finance_headlines(ticker):
#     url = f'https://finance.yahoo.com/quote/{ticker}/news'
#     headers = {'User-Agent': 'Mozilla/5.0'}
#
#     try:
#         response = requests.get(url, headers=headers, timeout=5)
#         soup = BeautifulSoup(response.content, 'html.parser')
#
#         # Find article headline containers
#         articles = soup.select('li.js-stream-content h3, li.js-stream-content h3 a')
#
#         headlines = [a.get_text(strip=True) for a in articles if a.get_text(strip=True)]
#         cleaned = [
#             h for h in headlines
#             if h not in {
#                 'News', 'Life', 'Entertainment', 'Finance', 'Sports',
#                 'New on Yahoo', 'Performance Overview', ''
#             }
#         ]
#         return cleaned[:8]
#
#     except Exception as e:
#         print(f"Error fetching headlines for {ticker}: {e}")
#         return []


def analyze_ticker_sentiment(model, ticker):
    print(f'[INFO] Analysing news sentiment for {ticker}...')
    headlines = get_yahoo_finance_headlines(ticker)
    if not headlines:
        return {'ticker': ticker, 'avg_sentiment': None, 'sentiment_counts': {}, 'headlines': []}

    results = model(headlines)
    sentiment_labels = [r['label'] for r in results]
    counts = Counter(sentiment_labels)

    # Assign numeric values to sentiments
    score_map = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
    numeric_scores = [score_map.get(label, 0) for label in sentiment_labels]
    avg_sentiment = sum(numeric_scores) / len(numeric_scores)

    return {
        'ticker': ticker,
        'avg_sentiment': avg_sentiment,
        'sentiment_counts': dict(counts),
        'sentiment_scores': numeric_scores,
        'sentiment_labels': sentiment_labels,
        'headlines': headlines
    }


def run_sentiment_analysis(ticker_list):
    sentiment_model = load_sentiment_model()
    all_results = [analyze_ticker_sentiment(sentiment_model, ticker) for ticker in ticker_list]
    return pd.DataFrame(all_results)

#
# from transformers import pipeline
#
# def load_sentiment_model():
#     return pipeline('sentiment-analysis', model='ProsusAI/finbert')
#
# def analyze_sentiment(model, headlines):
#     return model(headlines)
