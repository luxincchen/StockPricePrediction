import os
import pandas as pd
dir_path = 'data/train/stocks/'
stocks_lst = []

for stock in sorted(os.listdir(dir_path)): 
    file_path = os.path.join(dir_path, stock)
    df = pd.read_csv(file_path)
    stocks_lst.append(df)

stocks_df = pd.concat(stocks_lst)

sp500_df = pd.read_csv('data/train/indices/SP500.csv')
sp500_df = sp500_df.rename(columns={'Returns': 'sp500_return'})
sp500_df = sp500_df[['Date', 'sp500_return']]

merged_df = pd.merge(stocks_df, sp500_df, on='Date')
merged_df = merged_df.sort_values(by=['Date', 'Ticker'])

def rolling_beta(group, window=60):
    cov = group['Returns'].rolling(window).cov(group['sp500_return'])
    var = group['sp500_return'].rolling(window).var()
    return cov / var

merged_df['rolling_beta'] = merged_df.groupby('Ticker', group_keys=False).apply(rolling_beta)

