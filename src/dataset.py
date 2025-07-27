import pandas as pd
import os 
import numpy as np

class LinearDataset:
    def __init__(self, dir='data/train/stocks/'):
        self.dir = dir
    
    def _load_data(self): # "_"means we only use this function inside the class --> static function 
        all_stocks = os.listdir(self.dir) #listdir gives the list of what is inside the dir 
        all_dfs = []
        for stock in all_stocks:
            all_dfs.append(pd.read_csv(self.dir + stock))

        return pd.concat(all_dfs)

    def preprocess(self):
        df = self._load_data()
        filter_df = df[["Open", "High", "Low", "Close", "Adjusted", "Volume"]]

        X = filter_df.to_numpy()
        Y = df[["Returns"]].to_numpy()

        return X[:-10], Y[10:]

    def split(self, X, Y):
        X_test = X[int(len(X) * 0.8):]
        X_train = X[:int(len(X) * 0.8)]
        Y_test = Y[int(len(Y) * 0.8):]
        Y_train = Y[:int(len(Y) * 0.8)]

        return X_test, X_train, Y_test, Y_train

class ARIMADataset:
    def __init__(self, stock_path='data/train/stocks/AAPL.csv'):
        self.stock_path=stock_path
    
    def _load_data(self):
        return pd.read_csv(self.stock_path)
    
    def preprocess(self):
        df = self._load_data()
        filter_df = df['Returns']
        return filter_df
    
    def split(self, filter_df):
        train_df = filter_df[:int(len(filter_df) * 0.8)]
        test_df = filter_df[int(len(filter_df) * 0.8):]
        return train_df, test_df
    

         
        