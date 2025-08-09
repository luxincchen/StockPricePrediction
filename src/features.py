import os
import pandas as pd

def beta_features(stocks_dir='data/train/stocks/', sp500_path='data/train/indices/SP500.csv', window=60):
    #Load stock data
    if os.path.isdir(stocks_dir):
        stocks_lst = []
        for stock in sorted(os.listdir(stocks_dir)): 
            file_path = os.path.join(stocks_dir, stock)
            df = pd.read_csv(file_path)
            stocks_lst.append(df)
        stocks_df = pd.concat(stocks_lst)
    else:
        stocks_df = pd.read_csv(stocks_dir)

        if "Ticker" not in stocks_df.columns:
            stocks_df["Ticker"] = os.path.splitext(os.path.basename(stocks_dir))[0]


    #Load S&P500
    sp500_df = pd.read_csv('data/train/indices/SP500.csv')
    sp500_df = sp500_df.rename(columns={'Returns': 'sp500_return'})
    sp500_df = sp500_df[['Date', 'sp500_return']]

    merged_df = pd.merge(stocks_df, sp500_df, on='Date')
    merged_df = merged_df.sort_values(['Ticker','Date'])

    g = merged_df.groupby('Ticker')

    cov = g['Returns'].rolling(window).cov(g['sp500_return']).reset_index(level=0, drop=True)
    var = g['sp500_return'].rolling(window).var().reset_index(level=0, drop=True)

    merged_df['rolling_beta'] = cov / var
    merged_df['beta_lag1']    = g['rolling_beta'].shift(1).reset_index(level=0, drop=True)

    return merged_df 