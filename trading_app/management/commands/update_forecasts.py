from django.core.management.base import BaseCommand
import time
import pandas as pd
from django.conf import settings
from data_ingestion.yfinance_loader import fetch_data
from preprocessing.features import compute_advanced_technical_indicators
from forecasting.forecast import run_forecasting_pipeline
from sentiment.multi_sentiment import run_sentiment_pipeline
from agent.decision_agent import get_top_n
from trading_app.utils import DataManager
from trading_app.models import DataUpdateLog


class Command(BaseCommand):
    help = 'Update stock data and run forecasting pipeline'

    def handle(self, *args, **options):
        start_time = time.time()

        self.stdout.write('[INFO] Starting data update...')

        log_entry = DataUpdateLog.objects.create(
            status='started',
            message='Data update initiated via management command'
        )

        try:
            # Load tickers
            tickers_df = pd.read_csv(settings.TICKERS_PATH)
            tickers = tickers_df['ticker'].tolist()
            self.stdout.write(f'[INFO] Loaded {len(tickers)} tickers')

            # Fetch data
            self.stdout.write('[INFO] Fetching market data...')
            df = fetch_data(tickers=tickers, raw_data_path=settings.RAW_DATA_PATH)

            # Run XGBoost forecasting
            # self.stdout.write('[INFO] Running XGBoost forecasting...')
            # summary_df_xgb, full_df_xgb = run_forecasting_pipeline(
            #     df=df,
            #     n_days=14,
            #     compute_indicators_fn=compute_advanced_technical_indicators,
            #     model_type='xgb_regressor'
            # )

            # Run TimesFM forecasting
            self.stdout.write('[INFO] Running TimesFM forecasting...')
            summary_df, full_df = run_forecasting_pipeline(
                df=df,
                n_days=14,
                compute_indicators_fn=compute_advanced_technical_indicators,
                model_type='timesfm'
            )

            # Run sentiment analysis
            self.stdout.write('[INFO] Running sentiment analysis...')
            sentiment_df = run_sentiment_pipeline(tickers)

            # Calculate gain per share
            summary_df['gain_per_share'] = (
                    summary_df['avg_n_days_forecast'] - summary_df['current_price']
            )

            # Merge sentiment data
            summary_df = summary_df.merge(sentiment_df, on='ticker', how='left')

            # Get top N
            self.stdout.write('[INFO] Calculating top gainers...')
            top_n = get_top_n(df=summary_df, n=25)

            # Save data
            self.stdout.write('[INFO] Saving data...')
            DataManager.save_data(summary_df, full_df, top_n)

            # Update log
            duration = time.time() - start_time
            log_entry.status = 'completed'
            log_entry.message = f'Update completed. Processed {len(tickers)} tickers.'
            log_entry.duration_seconds = duration
            log_entry.save()

            self.stdout.write(
                self.style.SUCCESS(
                    f'[SUCCESS] Data update completed in {duration:.2f} seconds'
                )
            )

        except Exception as e:
            duration = time.time() - start_time
            log_entry.status = 'failed'
            log_entry.message = f'Error: {str(e)}'
            log_entry.duration_seconds = duration
            log_entry.save()

            self.stdout.write(
                self.style.ERROR(f'[ERROR] Data update failed: {e}')
            )
            raise
