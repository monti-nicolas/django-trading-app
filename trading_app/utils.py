import pandas as pd
import ast
from pathlib import Path
from django.conf import settings
import logging

logger = logging.getLogger('trading_app')


class DataManager:
    """
    Manager class for handling data file operations.
    """

    @staticmethod
    def load_ticker_details():
        """Load ticker details from Excel file."""
        try:
            return pd.read_csv(settings.TICKERS_DETAILS_PATH)
        except Exception as e:
            logger.error(f"Error loading ticker details: {e}")
            return pd.DataFrame()

    @staticmethod
    def load_tickers():
        """Load tickers list from CSV."""
        try:
            return pd.read_csv(settings.TICKERS_PATH)
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            return pd.DataFrame()

    @staticmethod
    def load_summary_df():
        """Load summary dataframe."""
        try:
            return pd.read_csv(settings.SUMMARY_DF_PATH)
        except Exception as e:
            logger.error(f"Error loading summary data: {e}")
            return pd.DataFrame()

    @staticmethod
    def load_full_df():
        """Load full dataframe."""
        try:
            df = pd.read_csv(settings.FULL_DF_PATH)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df
        except Exception as e:
            logger.error(f"Error loading full data: {e}")
            return pd.DataFrame()

    @staticmethod
    def load_top_n():
        """Load top N tickers."""
        try:
            return pd.read_csv(settings.TOPN_PATH)
        except Exception as e:
            logger.error(f"Error loading top N data: {e}")
            return pd.DataFrame()

    @staticmethod
    def save_data(summary_df, full_df, top_n):
        """Save dataframes to CSV files."""
        try:
            full_df.to_csv(settings.FULL_DF_PATH, index=False)
            summary_df.to_csv(settings.SUMMARY_DF_PATH, index=False)
            top_n.to_csv(settings.TOPN_PATH, index=False)
            return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False


class TickerDataProcessor:
    """
    Process ticker-specific data for display.
    """

    @staticmethod
    def get_ticker_summary(summary_df, ticker):
        """Get summary data for a specific ticker."""
        return summary_df[summary_df['ticker'] == ticker].copy()

    @staticmethod
    def get_ticker_full_data(full_df, ticker):
        """Get full data for a specific ticker."""
        return full_df[full_df['ticker'] == ticker].copy()

    @staticmethod
    def get_company_name(ticker_details, ticker):
        """Get company name for a ticker."""
        try:
            return ticker_details[ticker_details['ticker'] == ticker]['company_name'].values[0]
        except (IndexError, KeyError):
            return ticker

    @staticmethod
    def get_last_updated_date(full_df, ticker):
        """Get the last updated date for actual data."""
        ticker_data = full_df[
            (full_df['ticker'] == ticker) &
            (full_df['data_type'] == 'actual')
            ]
        if not ticker_data.empty:
            return ticker_data['date'].max().strftime('%Y-%m-%d')
        return "N/A"

    @staticmethod
    def calculate_shares_to_purchase(current_price, investment_amount):
        """Calculate number of shares to purchase."""
        if current_price > 0:
            return int(investment_amount / current_price)
        return 0

    @staticmethod
    def parse_list_field(value):
        """Parse string representation of list/dict fields."""
        if isinstance(value, str):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
        return value

    @staticmethod
    def prepare_sentiment_data(selected_summary):
        """Prepare sentiment data for display."""
        sentiment_data = {}

        try:
            # Average sentiment
            sentiment_data['avg_sentiment'] = selected_summary['avg_sentiment'].iloc[0]

            # Sentiment counts
            sentiment_counts = TickerDataProcessor.parse_list_field(
                selected_summary['sentiment_counts'].iloc[0]
            )
            sentiment_data['sentiment_counts'] = sentiment_counts

            # Parse individual lists
            headlines = TickerDataProcessor.parse_list_field(
                selected_summary['headlines'].iloc[0]
            )
            sentiment_scores = TickerDataProcessor.parse_list_field(
                selected_summary['sentiment_scores'].iloc[0]
            )
            sentiment_labels = TickerDataProcessor.parse_list_field(
                selected_summary['sentiment_labels'].iloc[0]
            )

            # Convert to lists if not already
            headlines = headlines if headlines else []
            sentiment_scores = sentiment_scores if sentiment_scores else []
            sentiment_labels = sentiment_labels if sentiment_labels else []

            # Zip them together
            sentiment_data['headlines_with_scores'] = list(zip(
                headlines,
                sentiment_labels,
                sentiment_scores
            ))

        except Exception as e:
            logger.error(f"Error preparing sentiment data: {e}")
            sentiment_data = {
                'avg_sentiment': 0,
                'sentiment_counts': {},
                'headlines_with_scores': []
            }

        return sentiment_data


class ChartDataProcessor:
    """
    Process data for chart rendering.
    """

    @staticmethod
    def prepare_price_forecast_data(full_df):
        """Prepare data for price forecast chart."""
        hist_df = full_df[full_df['data_type'] == 'actual']
        forecast_df = full_df[full_df['data_type'] == 'forecast']

        data = {
            'historical': {
                'dates': hist_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'close': hist_df['close'].tolist(),
            },
            'forecast': {
                'dates': forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'close': forecast_df['close'].tolist(),
            },
            'ma_10': {
                'dates': full_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'values': full_df['ma_10'].tolist(),
            }
        }

        if 'forecast_lower' in forecast_df.columns and 'forecast_upper' in forecast_df.columns:
            data['confidence_intervals'] = {
                'dates': forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'lower': forecast_df['forecast_lower'].tolist(),
                'upper': forecast_df['forecast_upper'].tolist(),
            }
        else:
            # Fallback: use forecast price as both bounds
            data['confidence_intervals'] = {
                'dates': forecast_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'lower': forecast_df['close'].tolist(),
                'upper': forecast_df['close'].tolist(),
            }

        data['currency'] = settings.PORTFOLIO_CURRENCY

        return data

    @staticmethod
    def prepare_moving_averages_data(full_df):
        """Prepare data for moving averages chart."""
        hist_df = full_df[full_df['data_type'] == 'actual']

        return {
            'historical': {
                'dates': hist_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'close': hist_df['close'].tolist(),
            },
            'ma_200': {
                'dates': full_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'values': full_df['ma_200'].tolist(),
            },
            'ma_50': {
                'dates': full_df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'values': full_df['ma_50'].tolist(),
            }
        }

    @staticmethod
    def prepare_technical_indicator_data(full_df, indicator_name):
        """Prepare data for a technical indicator chart."""
        return {
            'dates': full_df['date'].dt.strftime('%Y-%m-%d').tolist(),
            'values': full_df[indicator_name].fillna(0).tolist(),
        }

    @staticmethod
    def prepare_multi_line_data(full_df, indicators):
        """Prepare data for charts with multiple lines."""
        data = {
            'dates': full_df['date'].dt.strftime('%Y-%m-%d').tolist(),
        }
        for indicator in indicators:
            data[indicator] = full_df[indicator].fillna(0).tolist()
        return data