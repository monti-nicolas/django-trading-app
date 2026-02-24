import time
import json
from django.shortcuts import render
from django.views import View
from django.http import JsonResponse
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import logging
from data_ingestion.yfinance_loader import fetch_data
from preprocessing.features import compute_advanced_technical_indicators
from forecasting.forecast import run_forecasting_pipeline
from sentiment.multi_sentiment import run_sentiment_pipeline
from agent.decision_agent import get_top_n
import numpy as np
from decouple import config


from .utils import (
    DataManager,
    TickerDataProcessor,
    ChartDataProcessor
)
from .models import DataUpdateLog

logger = logging.getLogger('trading_app')


class DashboardView(View):
    """
    Main dashboard view - displays the stock forecast dashboard.
    """

    def get(self, request):
        """
        Handle GET request to display dashboard.
        """
        max_invest = config('MAX_INVEST', default=10500, cast=int)

        try:
            # Load data
            ticker_details = DataManager.load_ticker_details()
            summary_df = DataManager.load_summary_df()
            full_df = DataManager.load_full_df()
            top_n = DataManager.load_top_n()

            current_portfolio = settings.SOURCE_DIR_NAME
            stored_portfolio = request.session.get('current_portfolio')

            if stored_portfolio != current_portfolio:
                # Portfolio changed - clear old ticker selection
                logger.info(
                    f"Portfolio changed from {stored_portfolio} to {current_portfolio}. Resetting ticker selection.")
                request.session['selected_ticker'] = None
                request.session['current_portfolio'] = current_portfolio


            # Get selected ticker from session or default to first in top_n
            selected_ticker = request.session.get('selected_ticker')
            if selected_ticker is None and not top_n.empty:
                selected_ticker = top_n.iloc[0]['ticker']
                request.session['selected_ticker'] = selected_ticker

            # Validate that selected ticker exists in current portfolio
            if selected_ticker is not None:
                ticker_exists = selected_ticker in top_n['ticker'].values
                if not ticker_exists:
                    logger.warning(
                        f"Selected ticker {selected_ticker} not found in current portfolio. Resetting to first ticker.")
                    selected_ticker = None

            # If no valid ticker, use first from top_n
            if selected_ticker is None and not top_n.empty:
                selected_ticker = top_n.iloc[0]['ticker']
                request.session['selected_ticker'] = selected_ticker
                request.session['current_portfolio'] = current_portfolio

            # Get proposed investment from session or default to 1000
            proposed_investment = request.session.get('proposed_investment', 1000)

            # Prepare top N display data
            top_n_display = top_n[[
                'ticker', 'avg_n_days_forecast', 'mse', 'current_price', 'gain_per_share'
            ]].copy()
            top_n_display = top_n_display.reset_index(drop=True)

            # Get selected ticker data
            selected_summary = TickerDataProcessor.get_ticker_summary(
                summary_df, selected_ticker
            )
            selected_full = TickerDataProcessor.get_ticker_full_data(
                full_df, selected_ticker
            )

            # Get company name
            company_name = TickerDataProcessor.get_company_name(
                ticker_details, selected_ticker
            )

            # Get last updated date
            last_updated = TickerDataProcessor.get_last_updated_date(
                full_df, selected_ticker
            )

            # Calculate KPIs
            if not selected_summary.empty:
                current_price = selected_summary['current_price'].iloc[0]
                avg_forecast = selected_summary['avg_n_days_forecast'].iloc[0]
                mse = selected_summary['mse'].iloc[0]
                mape = selected_summary['mape'].iloc[0] if 'mape' in selected_summary.columns else 0.0
                gain_per_share = selected_summary['gain_per_share'].iloc[0]
                num_shares = TickerDataProcessor.calculate_shares_to_purchase(
                    current_price, proposed_investment
                )

                # Calculate estimated total gain after fees
                estimated_total_gain = gain_per_share * num_shares * 0.97

                # Prepare sentiment data
                # sentiment_data = TickerDataProcessor.prepare_sentiment_data(
                #     selected_summary
                # )
                sentiment_data = {}

                # Calculate 30-day averages for technical indicators
                recent = selected_full.tail(30)
                technical_averages = {
                    'rsi': recent['rsi'].mean(),
                    'macd': recent['macd'].mean(),
                    'atr': recent['atr'].mean(),
                    'stoch_k': recent['stoch_k'].mean(),
                    'adx': recent['adx'].mean(),
                    'vwap': recent['vwap'].mean(),
                }
            else:
                current_price = 0
                avg_forecast = 0
                mse = 0
                mape = 0
                gain_per_share = 0
                num_shares = 0
                estimated_total_gain = 0
                sentiment_data = {}
                technical_averages = {}

            # Investment options (500 to 10000, step 500)
            investment_options = list(range(500, max_invest, 500))

            context = {
                'ticker_details': ticker_details.to_dict('records'),
                'top_n': top_n_display.to_dict('records'),
                'selected_ticker': selected_ticker,
                'company_name': company_name,
                'last_updated': last_updated,
                'proposed_investment': proposed_investment,
                'investment_options': investment_options,

                # KPIs
                'avg_forecast': avg_forecast,
                'mse': mse,
                'mape': mape,
                'current_price': current_price,
                'gain_per_share': gain_per_share,
                'num_shares': num_shares,
                'estimated_total_gain': estimated_total_gain,

                # Sentiment data
                'sentiment_data': sentiment_data,

                # Technical indicators
                'technical_averages': technical_averages,

                # Chart data (will be loaded via JavaScript)
                'selected_full': selected_full.to_dict('records'),

                'portfolio_name': settings.PORTFOLIO_NAME,
                'portfolio_currency': settings.PORTFOLIO_CURRENCY,
                'portfolio_currency_symbol': settings.PORTFOLIO_CURRENCY_SYMBOL,
                'portfolio_config': settings.CURRENT_PORTFOLIO,
            }

            return render(request, 'trading_app/index.html', context)

        except Exception as e:
            logger.error(f"Error in DashboardView: {e}", exc_info=True)
            context = {
                'error': 'Failed to load dashboard data. Please check the logs.',
                'investment_options': list(range(500, max_invest, 500)),
                'portfolio_name': settings.PORTFOLIO_NAME,
                'portfolio_currency': settings.PORTFOLIO_CURRENCY,
                'portfolio_currency_symbol': settings.PORTFOLIO_CURRENCY_SYMBOL,
                'portfolio_config': settings.CURRENT_PORTFOLIO,
            }
            return render(request, 'trading_app/index.html', context)

    def post(self, request):
        """
        Handle POST request for ticker selection or investment change.
        """
        try:
            # Update selected ticker
            if 'selected_ticker' in request.POST:
                selected_ticker = request.POST.get('selected_ticker')
                request.session['selected_ticker'] = selected_ticker

            # Update proposed investment
            if 'proposed_investment' in request.POST:
                proposed_investment = int(request.POST.get('proposed_investment'))
                request.session['proposed_investment'] = proposed_investment

            # Redirect to GET to refresh page
            return self.get(request)

        except Exception as e:
            logger.error(f"Error in DashboardView POST: {e}", exc_info=True)
            return self.get(request)


class UpdateDataView(View):
    """
    View to handle data updates (triggered by Update button).
    """

    def post(self, request):
        """
        Update all data: fetch, forecast, sentiment analysis.
        """
        start_time = time.time()

        # Create log entry
        log_entry = DataUpdateLog.objects.create(
            status='started',
            message='Data update initiated'
        )

        try:
            logger.info("Starting data update...")

            # Step 1: Load tickers
            tickers_df = DataManager.load_tickers()
            tickers = tickers_df['ticker'].tolist()
            logger.info(f"Loaded {len(tickers)} tickers")

            # Step 2: Fetch data
            logger.info("Fetching market data...")
            df = fetch_data(
                tickers=tickers,
                raw_data_path=settings.RAW_DATA_PATH
            )

            # Step 3: Run XGBoost forecasting
            # logger.info("Running XGBoost forecasting...")
            # summary_df_xgb, full_df_xgb = run_forecasting_pipeline(
            #     df=df,
            #     n_days=14,
            #     compute_indicators_fn=compute_advanced_technical_indicators,
            #     model_type='xgb_regressor'
            # )

            # Step 4: Run TimesFM forecasting
            logger.info("Running TimesFM forecasting...")
            summary_df, full_df = run_forecasting_pipeline(
                df=df,
                n_days=config('N_DAYS', default=14, cast=int),
                compute_indicators_fn=compute_advanced_technical_indicators,
                model_type=config('FCAST_MODEL', default='timesfm', cast=str)
            )

            # Step 5: Run sentiment analysis
            # logger.info("Running sentiment analysis...")
            # sentiment_df = run_sentiment_pipeline(tickers)
            #
            # # Step 6: Merge sentiment data
            # summary_df = summary_df.merge(
            #     sentiment_df,
            #     on='ticker',
            #     how='left'
            # )

            # Step 7: Calculate gain per share and baseline total gain
            summary_df['gain_per_share'] = (
                    summary_df['avg_n_days_forecast'] - summary_df['current_price']
            )

            summary_df['baseline_shares_purchased'] = 1000 // summary_df['current_price']
            summary_df['baseline_total_gain'] = summary_df['gain_per_share'] * summary_df['baseline_shares_purchased'] * 0.97

            # Step 8: Get top N tickers
            logger.info("Calculating top gainers...")
            top_n = get_top_n(df=summary_df, n=config('TOP_N_TICKERS', default=25, cast=int))

            # Step 9: Save data
            logger.info("Saving data...")
            save_success = DataManager.save_data(summary_df, full_df, top_n)

            if not save_success:
                raise Exception("Failed to save data")

            # Update log entry
            duration = time.time() - start_time
            log_entry.status = 'completed'
            log_entry.message = f'Data update completed successfully. Processed {len(tickers)} tickers.'
            log_entry.duration_seconds = duration
            log_entry.save()

            logger.info(f"Data update completed in {duration:.2f} seconds")

            return JsonResponse({
                'success': True,
                'message': 'Data updated successfully',
                'duration': f"{duration:.2f}",
                'tickers_processed': len(tickers)
            })

        except Exception as e:
            # Update log entry with error
            duration = time.time() - start_time
            log_entry.status = 'failed'
            log_entry.message = f'Error: {str(e)}'
            log_entry.duration_seconds = duration
            log_entry.save()

            logger.error(f"Data update failed: {e}", exc_info=True)

            return JsonResponse({
                'success': False,
                'error': str(e),
                'message': 'Data update failed. Please check logs.'
            }, status=500)


class TickerDetailView(View):
    """
    View to get detailed data for a specific ticker (HTMX endpoint).
    """

    def get(self, request, ticker):
        """
        Get ticker details and update session.
        """
        try:
            # Update selected ticker in session
            request.session['selected_ticker'] = ticker

            # Load data
            ticker_details = DataManager.load_ticker_details()
            summary_df = DataManager.load_summary_df()
            full_df = DataManager.load_full_df()

            # Get proposed investment from session
            proposed_investment = request.session.get('proposed_investment', 1000)

            # Get selected ticker data
            selected_summary = TickerDataProcessor.get_ticker_summary(
                summary_df, ticker
            )
            selected_full = TickerDataProcessor.get_ticker_full_data(
                full_df, ticker
            )

            # Get company name
            company_name = TickerDataProcessor.get_company_name(
                ticker_details, ticker
            )

            # Get last updated date
            last_updated = TickerDataProcessor.get_last_updated_date(
                full_df, ticker
            )

            # Calculate KPIs
            if not selected_summary.empty:
                current_price = selected_summary['current_price'].iloc[0]
                avg_forecast = selected_summary['avg_n_days_forecast'].iloc[0]
                mse = selected_summary['mse'].iloc[0]
                mape = selected_summary['mape'].iloc[0] if 'mape' in selected_summary.columns else 0.0
                gain_per_share = selected_summary['gain_per_share'].iloc[0]
                num_shares = TickerDataProcessor.calculate_shares_to_purchase(
                    current_price, proposed_investment
                )

                estimated_total_gain = gain_per_share * num_shares * 0.97

                # Prepare sentiment data
                # sentiment_data = TickerDataProcessor.prepare_sentiment_data(
                #     selected_summary
                # )
                sentiment_data = {}

                # Calculate 30-day averages for technical indicators
                recent = selected_full.tail(30)
                technical_averages = {
                    'rsi': recent['rsi'].mean(),
                    'macd': recent['macd'].mean(),
                    'atr': recent['atr'].mean(),
                    'stoch_k': recent['stoch_k'].mean(),
                    'adx': recent['adx'].mean(),
                    'vwap': recent['vwap'].mean(),
                }
            else:
                current_price = 0
                avg_forecast = 0
                mse = 0
                mape = 0
                gain_per_share = 0
                num_shares = 0
                estimated_total_gain = 0
                sentiment_data = {}
                technical_averages = {}

            context = {
                'selected_ticker': ticker,
                'company_name': company_name,
                'last_updated': last_updated,
                'proposed_investment': proposed_investment,

                # KPIs
                'avg_forecast': avg_forecast,
                'mse': mse,
                'mape': mape,
                'current_price': current_price,
                'gain_per_share': gain_per_share,
                'num_shares': num_shares,
                'estimated_total_gain': estimated_total_gain,

                # Sentiment data
                'sentiment_data': sentiment_data,

                # Technical indicators
                'technical_averages': technical_averages,

                # Chart data
                'selected_full': selected_full.to_dict('records'),

                # Portfolio information
                'portfolio_currency': settings.PORTFOLIO_CURRENCY,
                'portfolio_currency_symbol': settings.PORTFOLIO_CURRENCY_SYMBOL,
            }

            return render(request, 'trading_app/partials/ticker_details.html', context)

        except Exception as e:
            logger.error(f"Error in TickerDetailView: {e}", exc_info=True)
            return JsonResponse({
                'error': str(e)
            }, status=500)


def get_chart_data(request, ticker):
    """
    API endpoint to get chart data for a specific ticker.
    Returns JSON data for Plotly charts.
    """
    try:
        # Load full dataframe
        full_df = DataManager.load_full_df()

        # Get ticker data
        selected_full = TickerDataProcessor.get_ticker_full_data(
            full_df, ticker
        )

        if selected_full.empty:
            return JsonResponse({
                'error': 'No data found for ticker'
            }, status=404)

        # Replace NaN with None (which becomes null in JSON)
        # selected_full = selected_full.fillna(value=None)
        # Replace NaN with None (which becomes null in JSON)
        # selected_full = selected_full.where(selected_full.notna(), None)
        selected_full = selected_full.replace({np.nan: None})

        historical_only = selected_full[selected_full['data_type'] == 'actual'].copy()

        # Prepare chart data
        chart_data = {
            'price_forecast': ChartDataProcessor.prepare_price_forecast_data(
                selected_full
            ),
            'moving_averages': ChartDataProcessor.prepare_moving_averages_data(
                selected_full
            ),
            'rsi': ChartDataProcessor.prepare_technical_indicator_data(
                historical_only, 'rsi'
            ),
            'macd': ChartDataProcessor.prepare_multi_line_data(
                historical_only, ['macd', 'macd_signal']
            ),
            'bollinger': ChartDataProcessor.prepare_multi_line_data(
                historical_only, ['bb_upper', 'bb_lower', 'close']
            ),
            'atr': ChartDataProcessor.prepare_technical_indicator_data(
                historical_only, 'atr'
            ),
            'obv': ChartDataProcessor.prepare_technical_indicator_data(
                historical_only, 'obv'
            ),
            'stochastic': ChartDataProcessor.prepare_multi_line_data(
                historical_only, ['stoch_k', 'stoch_d']
            ),
            'ichimoku': ChartDataProcessor.prepare_multi_line_data(
                historical_only, ['ichimoku_senkou_a', 'ichimoku_senkou_b', 'close']
            ),
            'adx': ChartDataProcessor.prepare_technical_indicator_data(
                historical_only, 'adx'
            ),
            'vwap': ChartDataProcessor.prepare_multi_line_data(
                historical_only, ['vwap', 'close']
            ),
        }

        return JsonResponse(chart_data)

    except Exception as e:
        logger.error(f"Error in get_chart_data: {e}", exc_info=True)
        return JsonResponse({
            'error': str(e)
        }, status=500)