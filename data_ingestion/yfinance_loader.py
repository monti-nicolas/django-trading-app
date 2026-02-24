import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from decouple import config

def fetch_data(tickers, raw_data_path):

    if not os.path.exists(raw_data_path):
        hist_period = config('HIST_DATA', default='1y', cast=str)
        print(f'[INFO] No existing raw data. Fetching {hist_period} year(s) of data...')
        combined_df = _fetch_multiple_tickers(tickers, period=hist_period)
        combined_df.to_csv(raw_data_path, index=False)
        return combined_df
    else:
        print("[INFO] Found existing raw data. Fetching updates...")
        existing_df = pd.read_csv(raw_data_path)
        existing_df['date'] = pd.to_datetime(existing_df['date'], format='mixed').dt.date

        updated_dfs = []

        for k, ticker in enumerate(tickers):
            print(f'[INFO] Extracting price data... processing ticker {k+1} out of  {len(tickers)}...')
            ticker_df = existing_df[existing_df['ticker'] == ticker]
            if ticker_df.empty:
                last_date = datetime.today().date() - timedelta(days=365)
            else:
                last_date = max(ticker_df['date'])

            new_df = fetch_price_data(ticker, period='3mo')
            if new_df is not None:
                new_df = new_df[new_df['date'] > last_date]
                updated_dfs.append(new_df)

        if updated_dfs:
            new_data_df = pd.concat(updated_dfs, ignore_index=True)
            combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            combined_df.drop_duplicates(subset=['ticker', 'date'], keep='last', inplace=True)
            combined_df.sort_values(['ticker', 'date'], inplace=True)
            combined_df.to_csv(raw_data_path, index=False)
            return combined_df
        else:
            print('[INFO] No new data found. Returning existing data.')
            return existing_df


def _fetch_multiple_tickers(tickers, period='1y'):
    all_dfs = []
    for k, ticker in enumerate(tickers):
        print(f'[INFO] Extracting price data... processing ticker {k + 1} out of  {len(tickers)}...')
        df = fetch_price_data(ticker, period=period)
        if df is not None:
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def fetch_price_data(ticker, period='1mo', interval='1d'):
    try:
        # print(f'[INFO] Extracting data for ticker {ticker}...')
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None

        df = df.reset_index()
        if isinstance(df.columns[0], tuple):
            df.columns = [col[0] for col in df.columns]

        if 'Adj Close' in df.columns:
            df = df.drop('Adj Close', axis=1)

        df.columns = [col.lower() for col in df.columns]
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df['date']).dt.date

        return df

    except Exception as e:
        print(f'[ERROR] Failed to fetch data for {ticker}: {e}')
        return None

