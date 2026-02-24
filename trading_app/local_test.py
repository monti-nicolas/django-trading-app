import pandas as pd
from agent.decision_agent import get_top_n
from decouple import config


summary_df = pd.read_csv('./source/etf/summary_df.csv')
top_n = get_top_n(df=summary_df, n=config('TOP_N_TICKERS', default=25, cast=int))
top_n.to_csv('./source/etf/top_n.csv', index=False)
