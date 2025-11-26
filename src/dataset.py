import pandas as pd
import os 
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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
    
class NNDataset(Dataset):
    def __init__(self, dir='data/train/stocks/', seq_len=10):
        self.dir = dir
        self.seq_len = seq_len
        self.X, self.Y = self.preprocess()

    def _load_data(self):
        all_stocks = os.listdir(self.dir)
        dfs = []
        for stock in all_stocks:
            df = pd.read_csv(os.path.join(self.dir, stock))
            dfs.append(df)
        return pd.concat(dfs)

    def preprocess(self):
        df = self._load_data()
        features = df[["Open", "High", "Low", "Close", "Adjusted", "Volume"]].values
        returns = df["Returns"].values

        # normalize features
        features = (features - features.mean(axis=0)) / features.std(axis=0)

        X_windows, Y_windows = [], []
        horizon = 10

        for i in range(len(features) - self.seq_len - horizon):
            X = features[i : i + self.seq_len]
            y = returns[i + self.seq_len : i + self.seq_len + horizon]  # always 10
            X_windows.append(X)
            Y_windows.append(y)

        return torch.tensor(np.array(X_windows), dtype=torch.float32), torch.tensor(np.array(Y_windows), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LSTMDataset(Dataset):
    def __init__(self, dir='data/train/stocks/', seq_len=50, horizon=10):
        self.dir = dir
        self.seq_len = seq_len
        self.horizon = horizon
        self.X, self.Y = self.preprocess()

    def _load_data(self):
        all_stocks = os.listdir(self.dir)
        dfs = []
        for stock in all_stocks:
            df = pd.read_csv(os.path.join(self.dir, stock))
            dfs.append(df)
        return pd.concat(dfs)

    def preprocess(self):
        df = self._load_data()
        features = df[['Open','High','Low','Close','Adjusted','Volume','Returns']]

        # Scale features
        self.scaler = MinMaxScaler(feature_range=(0,1))
        scaled = self.scaler.fit_transform(features)

        # Create sequences
        X, Y = self.create_seq(scaled, self.seq_len, self.horizon)
        return X, Y

    def create_seq(self, data, window_size, horizon):
        x, y = [], []
        for i in range(len(data) - window_size - horizon):
            row = data[i:i+window_size, :-1]   # use first 6 cols as input (features)
            x.append(row)

            label = data[i+window_size : i+window_size+horizon, -1]  # returns column
            y.append(label)
        return np.array(x), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)
