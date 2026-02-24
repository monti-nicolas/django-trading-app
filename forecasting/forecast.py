import warnings
warnings.filterwarnings("ignore")
from datetime import timedelta
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy import stats
import os


# os.environ['HF_HOME'] = '~/.cache/huggingface/hub/'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # For Mac M1/M2
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ======================================================================================
# TimesFM Shared Model Loader (single checkpoint load, reused across tickers)
# ======================================================================================

_TIMESFM_MODEL_CACHE = {}

def load_shared_timesfm_model(
    checkpoint_repo: str = "google/timesfm-2.0-500m-pytorch",
    backend: str = "torch",
    hparams_overrides: dict | None = None,
    cache_key: str | None = None,
    force_reload: bool = False
):
    """
    Load (or reuse) a TimesFM model using the v1.2.0 style API:
      TimesFm(hparams=TimesFmHparams(...), checkpoint=TimesFmCheckpoint(huggingface_repo_id=...))

    - checkpoint_repo examples:
        google/timesfm-2.0-500m
        google/timesfm-2.0-1.7b
        google/timesfm-1.0-500m
      Avoid the '-jax' suffix if you want Torch weights present.
    - backend: 'torch' or 'jax'. If 'torch' load fails due to missing weights, a fallback to 'jax' is attempted.
    """
    try:
        import timesfm
        from timesfm import TimesFm
    except ImportError as e:
        raise ImportError("[ERROR] timesfm not installed. Run: pip install timesfm") from e

    if cache_key is None:
        cache_key = f"{checkpoint_repo}|{backend}|{sorted((hparams_overrides or {}).items())}"

    if not force_reload and cache_key in _TIMESFM_MODEL_CACHE:
        return _TIMESFM_MODEL_CACHE[cache_key]

    base_hparams = dict(
        backend=backend,            # 'torch' or 'jax'
        per_core_batch_size=32,
        horizon_len=512,
        num_layers=50,
        context_len=2048,
    )
    if hparams_overrides:
        base_hparams.update(hparams_overrides)

    hparams_obj = timesfm.TimesFmHparams(**base_hparams)
    checkpoint_obj = timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint_repo)

    try:
        model = TimesFm(hparams=hparams_obj, checkpoint=checkpoint_obj)
    except FileNotFoundError as fe:
        # Fallback: if torch weights missing in repo, retry with jax backend
        if backend == "torch":
            print(f"[WARNING] Torch weights not found for {checkpoint_repo}. Retrying with backend='jax'.")
            base_hparams['backend'] = 'jax'
            hparams_obj = timesfm.TimesFmHparams(**base_hparams)
            model = TimesFm(hparams=hparams_obj, checkpoint=checkpoint_obj)
        else:
            raise fe
    except TypeError as te:
        raise RuntimeError(
            "[ERROR] TimesFm constructor failed. This code targets TimesFM >=1.2.0 API "
            "which expects (hparams=..., checkpoint=...). Verify your installed version."
        ) from te

    _TIMESFM_MODEL_CACHE[cache_key] = model
    return model


# ======================================================================================
# Utility
# ======================================================================================

def get_next_trading_day(current_date):
    next_day = current_date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


# ======================================================================================
# Feature Preparation (XGB path only)
# ======================================================================================

def prepare_features(data, train=True):
    data = data.copy()
    data['return'] = data['close'].pct_change()
    data['close_lag_1'] = data['close'].shift(1)
    data['volume_lag_1'] = data['volume'].shift(1)
    data['return_lag_1'] = data['return'].shift(1)

    data['target_close'] = data['close'].shift(-1)
    data['target_high'] = data['high'].shift(-1)
    data['target_low'] = data['low'].shift(-1)
    data['target_volume'] = data['volume'].shift(-1)

    features = [
        'ma_5', 'ma_10', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr',
        'obv', 'stoch_k', 'stoch_d', 'ichimoku_senkou_a',
        'ichimoku_senkou_b', 'adx', 'vwap',
        'return', 'close_lag_1', 'volume_lag_1', 'return_lag_1'
    ]
    feature_df = data[features]
    target_df = data[['target_close', 'target_high', 'target_low', 'target_volume']]

    if train:
        mask = feature_df.notna().all(axis=1) & target_df.notna().all(axis=1)
        feature_df = feature_df[mask]
        target_df = target_df[mask]

    return feature_df, target_df


# ======================================================================================
# XGB Forecaster
# ======================================================================================

class XGBMultiTargetForecaster:
    def __init__(self, xgb_params=None):
        self.models = {}
        self.metrics = {}
        self.xgb_params = xgb_params or dict(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

    def fit(self, X, targets):
        self.models = {}
        self.metrics = {}
        for column in targets.columns:
            X_train, X_test, y_train, y_test = train_test_split(
                X, targets[column], test_size=0.2, shuffle=False
            )
            model = xgb.XGBRegressor(**self.xgb_params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            self.models[column] = model
            self.metrics[column] = mse
        return self

    def forecast_next_n_days(self, original_df, n_days, compute_indicators_fn):
        forecasts = []
        df_raw = original_df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'data_type']].copy()
        df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
        current_forecast_date = df_raw['date'].max()

        for _ in range(n_days):
            df_with_indicators = compute_indicators_fn(df_raw.copy())
            if df_with_indicators.empty:
                print('[WARNING] Skipping forecast iteration — insufficient data after indicators.')
                break
            features_df, _ = prepare_features(data=df_with_indicators, train=False)
            if features_df.empty:
                print('[WARNING] No feature row available for next step.')
                break
            latest = features_df.iloc[-1:]
            forecast_map = {t: m.predict(latest)[0] for t, m in self.models.items()}
            current_forecast_date = get_next_trading_day(current_forecast_date)
            new_row = {
                'date': current_forecast_date,
                'ticker': df_raw['ticker'].iloc[-1],
                'close': forecast_map['target_close'],
                'high': forecast_map['target_high'],
                'low': forecast_map['target_low'],
                'volume': forecast_map['target_volume'],
                'open': forecast_map['target_close'],
                'data_type': 'forecast',
                'forecast_lower': forecast_map['target_close'],
                'forecast_upper': forecast_map['target_close'],
            }
            df_raw = pd.concat([df_raw, pd.DataFrame([new_row])], ignore_index=True)
            forecasts.append(new_row)

        return pd.DataFrame(forecasts)


# ======================================================================================
# TimesFM Forecaster (shared model)
# ======================================================================================

class TimesFMForecaster:
    """
    Wraps a shared TimesFM model instance for multi-ticker forecasting.
    Each OHLCV field forecast independently as univariate.
    """

    def __init__(self, model, horizon, freq='D', confidence_level=0.90):
        self.model = model
        self.horizon = horizon
        self.freq = freq
        self.freq_code = self._get_freq_code(freq)
        self.confidence_level = confidence_level
        self.eval_mse_close = None
        self.eval_mape_close = None

    def _get_freq_code(self, freq):
        """
        Convert frequency string to TimesFM frequency code:
        - 0: high frequency (T, MIN, H, D, B, U) - daily and higher
        - 1: medium frequency (W, M) - weekly/monthly
        - 2: low frequency (Q, Y) - quarterly/yearly
        """
        freq_upper = freq.upper()
        if freq_upper in ['T', 'MIN', 'H', 'D', 'B', 'U']:
            return 0
        elif freq_upper in ['W', 'M']:
            return 1
        elif freq_upper in ['Q', 'Y']:
            return 2
        else:
            # Default to daily frequency
            return 0

    def fit(self, df):
        """
        Optional simple backtest on close series (last horizon vs forecast).
        Does not fine-tune TimesFM (foundation model is frozen).
        """
        close_series = df['close'].astype(float).values
        if len(close_series) > 2 * self.horizon:
            context = close_series[:-self.horizon]
            actual_future = close_series[-self.horizon:]
            try:
                # Call forecast method with correct API
                point_forecast, _ = self.model.forecast(
                    [context],  # Input as list of arrays
                    freq=[self.freq_code]  # Frequency as list
                )

                # Extract predictions - point_forecast should be a list of arrays
                if isinstance(point_forecast, list) and len(point_forecast) > 0:
                    preds = np.asarray(point_forecast[0]).flatten()
                else:
                    preds = np.asarray(point_forecast).flatten()

                # Ensure we don't exceed the actual future length
                preds = preds[:len(actual_future)]

                if len(preds) > 0 and len(actual_future) > 0:
                    # Adjust lengths to match
                    min_len = min(len(preds), len(actual_future))

                    self.eval_mse_close = float(mean_squared_error(
                        actual_future[:min_len],
                        preds[:min_len]
                    ))

                    # Avoid division by zero by filtering out very small actual values
                    mask = np.abs(actual_future[:min_len]) > 0.01
                    if mask.sum() > 0:
                        self.eval_mape_close = float(mean_absolute_percentage_error(
                            actual_future[:min_len][mask],
                            preds[:min_len][mask]
                        ) * 100)  # Convert to percentage
                    else:
                        self.eval_mape_close = None
                else:
                    self.eval_mse_close = None
                    self.eval_mape_close = None

            except Exception as e:
                print(f"[WARNING] Error in fit method: {e}")
                self.eval_mse_close = None
                self.eval_mape_close = None
        else:
            self.eval_mse_close = None
            self.eval_mape_close = None
        return self

    def _calculate_confidence_intervals(self, historical_series, point_forecast, confidence_level):
        """
        Calculate confidence intervals using historical volatility.

        Method: Use rolling standard deviation of returns to estimate uncertainty.
        """
        # Calculate historical returns
        returns = np.diff(historical_series) / historical_series[:-1]

        # Calculate rolling volatility (standard deviation of returns)
        # Use last 30 days or available data
        window = min(30, len(returns))
        recent_returns = returns[-window:]
        volatility = float(np.std(recent_returns))

        # Calculate z-score for confidence level
        # For 90% CI: z = 1.645
        # For 95% CI: z = 1.96
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        # Calculate expanding uncertainty (grows with forecast horizon)
        # Uncertainty increases with sqrt(time) - standard in financial forecasting
        lower_bound = np.zeros(self.horizon)
        upper_bound = np.zeros(self.horizon)

        for i in range(self.horizon):
            # Uncertainty grows with square root of time steps
            time_factor = np.sqrt(i + 1)
            margin = z_score * volatility * point_forecast[i] * time_factor

            lower_bound[i] = max(0, point_forecast[i] - margin)  # Price can't be negative
            upper_bound[i] = point_forecast[i] + margin

        return lower_bound, upper_bound

    def forecast_next_n_days(self, original_df, compute_indicators_fn):
        base = original_df[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'data_type']].copy()
        base = base.sort_values('date')
        last_date = base['date'].max()

        targets_map = {}
        confidence_intervals = {}

        # for col in ['close', 'high', 'low', 'volume']:
        for col in ['close']:
            try:
                series = base[col].astype(float).values

                # Call forecast method with correct API
                point_forecast, _ = self.model.forecast(
                    [series],  # Input as list of arrays
                    freq=[self.freq_code]  # Frequency as list
                )

                # Extract predictions - point_forecast should be a list of arrays
                if isinstance(point_forecast, list) and len(point_forecast) > 0:
                    preds = np.asarray(point_forecast[0]).flatten()
                else:
                    preds = np.asarray(point_forecast).flatten()

                # Ensure we have exactly horizon predictions
                if len(preds) < self.horizon:
                    # Pad with the last value if we have fewer predictions
                    last_val = preds[-1] if len(preds) > 0 else series[-1]
                    preds = np.concatenate([preds, np.full(self.horizon - len(preds), last_val)])
                elif len(preds) > self.horizon:
                    # Truncate if we have more predictions
                    preds = preds[:self.horizon]

                targets_map[col] = preds

                # Calculate confidence intervals using historical volatility
                if col == 'close':  # Only for close price
                    lower_bound, upper_bound = self._calculate_confidence_intervals(
                        series, preds, self.confidence_level
                    )
                    confidence_intervals['lower'] = lower_bound
                    confidence_intervals['upper'] = upper_bound

            except Exception as e:
                print(f"[WARNING] Error forecasting {col}: {e}")
                # Fallback: use the last known value
                last_val = base[col].iloc[-1]
                targets_map[col] = np.full(self.horizon, last_val)

                # Set dummy confidence intervals on error
                if col == 'close':
                    confidence_intervals['lower'] = np.full(self.horizon, last_val)
                    confidence_intervals['upper'] = np.full(self.horizon, last_val)

        forecast_rows = []
        current_date = last_date
        for i in range(self.horizon):
            current_date = get_next_trading_day(current_date)
            forecast_rows.append({
                'date': current_date,
                'ticker': base['ticker'].iloc[-1],
                'close': float(targets_map['close'][i]),
                # 'high': float(targets_map['high'][i]),
                'high': 0.0,
                # 'low': float(targets_map['low'][i]),
                'low': 0.0,
                # 'volume': float(targets_map['volume'][i]),
                'volume': 0.0,
                # 'open': float(targets_map['close'][i]),
                'open': 0.0,
                'data_type': 'forecast',
                'forecast_lower': float(confidence_intervals['lower'][i]),
                'forecast_upper': float(confidence_intervals['upper'][i]),
            })

        forecast_df = pd.DataFrame(forecast_rows)
        combined = pd.concat([base, forecast_df], ignore_index=True)
        combined_with_indicators = compute_indicators_fn(combined.copy())
        return forecast_df, combined_with_indicators


# ======================================================================================
# Unified Pipeline
# ======================================================================================

def run_forecasting_pipeline(
    df: pd.DataFrame,
    n_days: int,
    compute_indicators_fn,
    model_type: str = 'timesfm',
    xgb_params: dict | None = None,
    # TimesFM specific
    timesfm_checkpoint_repo: str = "google/timesfm-2.0-500m-pytorch",
    timesfm_backend: str = "torch",
    timesfm_hparams_overrides: dict | None = None,
    timesfm_cache_key: str | None = None,
    timesfm_force_reload: bool = False
):
    """
    model_type: 'xgb_regressor' or 'timesfm'
    """
    allowed = {'xgb_regressor', 'timesfm'}
    if model_type not in allowed:
        raise ValueError(f'[ERROR] model_type must be one of {allowed}')

    shared_timesfm_model = None
    if model_type == 'timesfm':
        shared_timesfm_model = load_shared_timesfm_model(
            checkpoint_repo=timesfm_checkpoint_repo,
            backend=timesfm_backend,
            hparams_overrides=timesfm_hparams_overrides,
            cache_key=timesfm_cache_key,
            force_reload=timesfm_force_reload
        )
        print(f"[INFO] Reusing TimesFM checkpoint={timesfm_checkpoint_repo} backend={timesfm_backend}")

    summary_results = []
    all_forecasted_dfs = []

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['data_type'] = 'actual'

    total_tickers = df['ticker'].nunique()

    for k, (ticker, group) in enumerate(df.groupby('ticker')):
        print(f'[INFO] Running forecast pipeline for {ticker} using {model_type}... {k+1} out of {total_tickers} tickers...')
        group = group.sort_values('date')
        group = compute_indicators_fn(group)

        if group.dropna().empty:
            print(f'[WARNING] Skipping {ticker} — all indicator rows NaN after computation.')
            continue

        # XGB path
        if model_type == 'xgb_regressor':
            X, y = prepare_features(group)
            if X.empty or y.empty:
                print(f'[WARNING] Skipping {ticker} — not enough data after feature prep.')
                continue
            try:
                forecaster = XGBMultiTargetForecaster(xgb_params=xgb_params)
                forecaster.fit(X, y)
            except Exception as e:
                print(f'[ERROR] Error training XGB models for {ticker}: {e}')
                continue

            next_n_df = forecaster.forecast_next_n_days(
                original_df=group,
                n_days=n_days,
                compute_indicators_fn=compute_indicators_fn
            )
            if next_n_df.empty:
                print(f'[WARNING] Skipping {ticker} — empty forecast output.')
                continue

            close_list = next_n_df['close'].tolist()
            raw = group[['date', 'open', 'high', 'low', 'close', 'volume', 'ticker', 'data_type']].copy()
            raw_plus_forecast = pd.concat([raw, next_n_df], ignore_index=True)
            raw_plus_forecast = compute_indicators_fn(raw_plus_forecast)

            summary_results.append({
                'ticker': ticker,
                'next_day_forecast': next_n_df.iloc[0]['close'],
                'next_n_days_forecast': close_list,
                'max_n_days_forecast': max(close_list),
                'min_n_days_forecast': min(close_list),
                'avg_n_days_forecast': float(np.mean(close_list)),
                'mse': forecaster.metrics.get('target_close'),
                'mape': 0.0,  # Dummy value for XGBoost, actual calculation done for Timesfm
                'current_price': group['close'].iloc[-1],
                'current_rsi': group['rsi'].iloc[-1] if 'rsi' in group.columns else None,
                'model_type': model_type
            })
            all_forecasted_dfs.append(raw_plus_forecast)

        # TimesFM path
        else:
            try:
                forecaster = TimesFMForecaster(
                    model=shared_timesfm_model,
                    horizon=n_days,
                    freq='D'
                ).fit(group)
                forecast_df, combined_with_indicators = forecaster.forecast_next_n_days(
                    original_df=group,
                    compute_indicators_fn=compute_indicators_fn
                )
            except Exception as e:
                print(f'[ERROR] TimesFM forecasting failed for {ticker}: {e}')
                continue

            if forecast_df.empty:
                print(f'[WARNING] Skipping {ticker} — TimesFM produced empty forecast.')
                continue

            close_list = forecast_df['close'].tolist()
            summary_results.append({
                'ticker': ticker,
                'next_day_forecast': forecast_df.iloc[0]['close'],
                'next_n_days_forecast': close_list,
                'max_n_days_forecast': max(close_list),
                'min_n_days_forecast': min(close_list),
                'avg_n_days_forecast': float(np.mean(close_list)),
                'mse': forecaster.eval_mse_close,
                'mape': forecaster.eval_mape_close,
                'current_price': group['close'].iloc[-1],
                'current_rsi': group['rsi'].iloc[-1] if 'rsi' in group.columns else None,
                'model_type': model_type
            })

            all_forecasted_dfs.append(combined_with_indicators)

    summary_df = pd.DataFrame(summary_results)
    full_forecast_df = pd.concat(all_forecasted_dfs, ignore_index=True) if all_forecasted_dfs else pd.DataFrame()
    print('[INFO] Forecasting process completed.')
    return summary_df, full_forecast_df
