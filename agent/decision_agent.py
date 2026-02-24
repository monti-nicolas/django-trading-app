

def get_top_n(df, n=25):
    gain_df = df.copy()
    # top_df = gain_df[(gain_df['gain_per_share'] > 0) & (gain_df['avg_sentiment'] > 0)]
    # top_df = gain_df[gain_df['gain_per_share'] > 0]
    top_n = gain_df.sort_values(by='gain_per_share', ascending=False).head(n)
    # top_n = gain_df.sort_values(by='baseline_total_gain', ascending=False).head(n)

    return top_n


def generate_trade_signal(forecasted_price, current_price, sentiment_scores, rsi):
    forecast_delta = (forecasted_price - current_price) / current_price
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    if forecast_delta > 0.01 and avg_sentiment > 0.6 and rsi < 70:
        return "BUY"
    elif forecast_delta < -0.01 or avg_sentiment < 0.4 or rsi > 80:
        return "SELL"
    else:
        return "HOLD"
