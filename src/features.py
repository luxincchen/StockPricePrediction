import os
import pandas as pd




def beta_features(stocks_dir="data/train/stocks/", sp500_path="data/train/indices/SP500.csv", window=60):
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

    # Ensure stock returns exist
    if "Returns" not in stocks_df.columns:
        price_col = "Adjusted" if "Adjusted" in stocks_df.columns else "Close"
        stocks_df["Returns"] = stocks_df.groupby("Ticker")[price_col].pct_change()

    # Load S&P500 and make sure we have sp500_return
    sp500_df = pd.read_csv(sp500_path)
    if "sp500_return" not in sp500_df.columns:
        if "Returns" in sp500_df.columns:
            sp500_df = sp500_df.rename(columns={"Returns": "sp500_return"})
        else:
            price_col = "Adjusted" if "Adjusted" in sp500_df.columns else "Close"
            sp500_df["sp500_return"] = sp500_df[price_col].pct_change()
    sp500_df = sp500_df[["Date", "sp500_return"]]

    # Merge and sort
    merged_df = pd.merge(stocks_df, sp500_df, on="Date", how="inner")
    merged_df = merged_df.sort_values(["Ticker", "Date"], kind="stable")

    # Prepare columns
    merged_df["cov"] = pd.NA
    merged_df["var"] = pd.NA

    # Compute per-ticker rolling cov and var
    g = merged_df.groupby("Ticker", sort=False)
    for name, d in g:
        cov = d["Returns"].rolling(window).cov(d["sp500_return"])
        var = d["sp500_return"].rolling(window).var()
        merged_df.loc[d.index, "cov"] = cov
        merged_df.loc[d.index, "var"] = var

    merged_df["beta"] = merged_df["cov"] / merged_df["var"]
    merged_df["beta_lag1"] = merged_df.groupby("Ticker")["beta"].shift(1)

    merged_df = merged_df.drop(columns=["cov", "var"])

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
    
    merged_df['MA_7'] = g[price].transform(lambda x: x.rolling(window=7).mean())
    merged_df['MA_50'] = g[price].transform(lambda x: x.rolling(window=50).mean())
    merged_df['MA_7_lag1'] = g['MA_7'].shift(1)
    merged_df['MA_50_lag1'] = g['MA_50'].shift(1)


    return merged_df