import requests
import pandas as pd
from transformers import pipeline
from collections import Counter
from datetime import datetime, timedelta
import time
from django.conf import settings


def get_finnhub_headlines(ticker, n=8, days=30):
    """Fetch the latest company news headlines from Finnhub within the last `days` days."""
    to_date = datetime.today().date()
    from_date = to_date - timedelta(days=days)

    url = (
        f'https://finnhub.io/api/v1/company-news'
        f'?symbol={ticker}&from={from_date}&to={to_date}&token={settings.FINNHUB_API_KEY}'
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        news = response.json()
        return [article['headline'] for article in news[:n]]
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []


def get_finnhub_sentiment(ticker):
    url = f"https://finnhub.io/api/v1/news-sentiment?symbol={ticker}&token={settings.FINNHUB_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        sentiment = response.json()
        return {
            'finnhub_sentiment': {
                'bullish_score': sentiment.get('bullishPercent'),
                'bearish_score': sentiment.get('bearishPercent'),
                'company_news_score': sentiment.get('companyNewsScore'),
                'sector_avg_bullish': sentiment.get('sectorAverageBullishPercent'),
            }
        }
    except Exception as e:
        print(f'[ERROR] Error fetching sentiment for {ticker}: {e}')
        return {'finnhub_sentiment': None}


def load_finbert_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")


def analyze_with_finbert(model, headlines):
    if not headlines:
        return 0.0, {}, [], []
    results = model(headlines)
    labels = [r['label'] for r in results]
    normalized_labels = [label.upper() for label in labels]
    score_map = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
    numeric_scores = [score_map.get(label, 0) for label in normalized_labels]
    avg_score = sum(numeric_scores) / len(numeric_scores)
    counts = dict(Counter(normalized_labels))
    return avg_score, counts, labels, numeric_scores


def analyze_ticker_sentiment(ticker, finbert_model):
    # print(f'[INFO] Analysing news sentiment for ticker {ticker}...')
    headlines = get_finnhub_headlines(ticker)
    # finhub_sentiment = get_finnhub_sentiment(ticker)
    avg_sentiment, label_counts, labels, scores = analyze_with_finbert(finbert_model, headlines)

    return {
        'ticker': ticker,
        'avg_sentiment': avg_sentiment,
        'sentiment_counts': label_counts,
        'sentiment_labels': labels,
        'sentiment_scores': scores,
        'headlines': headlines,
        # **finhub_sentiment
    }


def run_sentiment_pipeline(ticker_list):
    model = load_finbert_model()
    results = []
    for k, ticker in enumerate(ticker_list):
        print(f'[INFO] Extracting news sentiment... processing ticker {k+1} out of {len(ticker_list)}... ')
        result = analyze_ticker_sentiment(ticker, model)
        results.append(result)
        time.sleep(1)  # Finnhub free tier allows 60 API calls/min
    print('[INFO] News sentiment analysis completed.')
    return pd.DataFrame(results)

