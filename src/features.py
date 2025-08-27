import os
import pandas as pd

def beta_features(stocks_dir='data/train/stocks/', sp500_path='data/train/indices/SP500.csv', window=60):
    # Load stock data (folder or single file)
    if os.path.isdir(stocks_dir):
        stocks_lst = []
        for stock in sorted(os.listdir(stocks_dir)):
            file_path = os.path.join(stocks_dir, stock)
            df = pd.read_csv(file_path)
            stocks_lst.append(df)
        stocks_df = pd.concat(stocks_lst, ignore_index=True)
    else:
        stocks_df = pd.read_csv(stocks_dir)

    # Ensure Ticker exists for single-file case
    if "Ticker" not in stocks_df.columns:
        stocks_df["Ticker"] = os.path.splitext(os.path.basename(stocks_dir))[0]

    # Load S&P500 (use the provided path)
    sp500_df = pd.read_csv(sp500_path).rename(columns={'Returns': 'sp500_return'})[['Date', 'sp500_return']]

    # Merge and sort
    merged_df = pd.merge(stocks_df, sp500_df, on='Date', how='inner')
    merged_df = merged_df.sort_values(['Ticker', 'Date'], kind='stable')

    # Group by ticker; group_keys=False keeps the original index (no MultiIndex to drop)
    g = merged_df.groupby('Ticker', group_keys=False, sort=False)

    # Rolling cov/var per ticker
    cov = g.apply(lambda d: d['Returns'].rolling(window).cov(d['sp500_return']))
    var = g.apply(lambda d: d['sp500_return'].rolling(window).var())

    # Compute beta and its lag
    merged_df['beta'] = cov / var
    merged_df['beta_lag1'] = g['beta'].shift(1)

    return merged_df

def mom_features(stocks_dir='data/train/stocks/', sp500_path='data/train/indices/SP500.csv'):
    # Load stock data (folder or single file)
    if os.path.isdir(stocks_dir):
        stocks_lst = []
        for stock in sorted(os.listdir(stocks_dir)):
            file_path = os.path.join(stocks_dir, stock)
            df = pd.read_csv(file_path)
            stocks_lst.append(df)
        stocks_df = pd.concat(stocks_lst, ignore_index=True)
    else:
        stocks_df = pd.read_csv(stocks_dir)

    # Ensure Ticker exists for single-file case
    if "Ticker" not in stocks_df.columns:
        stocks_df["Ticker"] = os.path.splitext(os.path.basename(stocks_dir))[0]

    # Load S&P500 (use the provided path)
    sp500_df = pd.read_csv(sp500_path).rename(columns={'Returns': 'sp500_return'})[['Date', 'sp500_return']]

    # Merge and sort
    merged_df = pd.merge(stocks_df, sp500_df, on='Date', how='inner')
    merged_df = merged_df.sort_values(['Ticker', 'Date'], kind='stable')

    # Group by ticker; group_keys=False keeps the original index (no MultiIndex to drop)
    g = merged_df.groupby('Ticker', group_keys=False, sort=False)

    #stock momentum
    price = 'Adjusted' if 'Adjusted' in merged_df.columns else 'Close'
    
    merged_df['mom_10'] = g[price].pct_change(10)
    merged_df['mom_20'] = g[price].pct_change(20)

    #lag to avoid leakage
    for c in ['mom_10', 'mom_20']:
        if c in merged_df.columns:
            merged_df[c + '_lag1'] = g[c].shift(1)

    return merged_df