import pandas as pd
import os 
import numpy as np
import torch
from torch.utils.data import Dataset
from src.features import beta_features
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
        df = beta_features(stocks_dir=self.dir, sp500_path='data/train/indices/SP500.csv', window=60)
        df = df.dropna(subset=["beta_lag1"])

        filter_df = df[["Open", "High", "Low", "Close", "Adjusted", "Volume", "beta_lag1"]]
        
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
    
    
class NNDataset(Dataset):
    def __init__(self, dir='data/train/stocks/', seq_len=10):
        self.dir = dir
        self.seq_len = seq_len
        self.X, self.Y = self.preprocess()

    def _load_data(self): # "_"means we only use this function inside the class --> static function 
        all_stocks = os.listdir(self.dir) #listdir gives the list of what is inside the dir 
        all_dfs = []
        for stock in all_stocks:
            all_dfs.append(pd.read_csv(self.dir + stock))

        return pd.concat(all_dfs)
    
    def preprocess(self):
        df = self._load_data()
        features = df[["Open", "High", "Low", "Close", "Adjusted", "Volume"]].values.tolist()
        returns = df[["Returns"]].values.tolist()
        features, returns = features[-self.seq_len:], returns[:self.seq_len]

        X, y = [], []
        model_input = []
        model_target = []
        for i, (feature_lst, target) in enumerate(zip(features, returns)):
            model_input += feature_lst   
            model_target += [target]
            if (i + 1) % self.seq_len == 0:
                X.append(model_input)
                y.append(model_target)
                model_input = []
                model_target = []
            
        return torch.tensor(X), torch.tensor(y)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32).flatten()  
        return x, y