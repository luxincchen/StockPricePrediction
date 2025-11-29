# src/main.py

import os
import json
import copy
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from .linear_model import train as linear_train, test as linear_test, predict as linear_predict
from .new_dataset import LinearDataset
from .features import beta_features, mom_features

from .arima_model import train as arima_train, predict as arima_predict

from .NN_model import StockNN, train_model, test_model
from .dataset import NNDataset


class LinearRunner:
    def __init__(self):
        self.train_stocks_dir = "data/train/stocks/"
        self.sp500_path = "data/train/indices/SP500.csv"
        self.test_template = "data/test/test_{}.csv"
        self.output_path = "results/submission.csv"

    def run(self):
        dataset = LinearDataset(self.train_stocks_dir)
        X, Y = dataset.preprocess()
        X_test, X_train, Y_test, Y_train = dataset.split(X, Y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = linear_train(X_train, Y_train)
        mse, _ = linear_test(model, X_test, Y_test)
        print(f"The MSE of the linear model is {mse}.")

        predictions = []

        feature_cols = [
            "Open", "High", "Low", "Close", "Adjusted", "Volume",
            "beta_lag1", "mom_10_lag1", "mom_20_lag1", "MA_7", "MA_50"
        ]

        for i in range(1, 6):
            test_path = self.test_template.format(i)
            df_beta = beta_features(stocks_dir=test_path, sp500_path=self.sp500_path)
            df_mom = mom_features(stocks_dir=test_path, sp500_path=self.sp500_path)

            df = pd.merge(
                df_beta,
                df_mom[["Ticker", "Date", "mom_10_lag1", "mom_20_lag1", "MA_7", "MA_50"]],
                on=["Ticker", "Date"],
                how="left"
            )

            df = df[feature_cols].dropna()
            X_pred = df.to_numpy()[-10:]
            X_pred = scaler.transform(X_pred)  # IMPORTANT FIX

            Y_pred = linear_predict(model, X_pred)
            predictions.append(Y_pred.flatten())

        pred_matrix = list(zip(*predictions))
        dates = [
            "2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-31",
            "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07"
        ]

        submission_df = pd.DataFrame(pred_matrix, columns=[f"Returns_{i}" for i in range(1, 6)])
        submission_df.insert(0, "Date", dates)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        submission_df.to_csv(self.output_path, index=False)
        print(f"Saved {self.output_path}!")


class ArimaRunner:
    def __init__(self):
        self.test_template = "data/test/test_{}.csv"
        self.output_path = "results/submission_arima.csv"

    def run(self):
        predictions = []

        for i in range(1, 6):
            df = pd.read_csv(self.test_template.format(i))
            model = arima_train(df["Returns"].dropna())
            forecast = arima_predict(model, 10)
            predictions.append(forecast.values)

        dates = [
            "2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-31",
            "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07"
        ]

        pred_matrix = list(zip(*predictions))
        submission_df = pd.DataFrame(pred_matrix, columns=[f"Returns_{i}" for i in range(1, 6)])
        submission_df.insert(0, "Date", dates)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        submission_df.to_csv(self.output_path, index=False)
        print(f"Saved {self.output_path}!")


class NNRunner:
    def __init__(self):
        self.SEED = 42
        self.n_features = 6
        self.horizon = 10
        self.seq_len_values = [10, 20, 50]
        self.hidden1_values = [64, 128]
        self.hidden2_values = [128, 256]
        self.learning_rates = [1e-4, 1e-3]
        self.epoch_values = [5, 10, 15]
        self.train_stocks_dir = "data/train/stocks"
        self.test_template = "data/test/test_{}.csv"
        self.output_path = "results/submission_NN_best.csv"
        self.best_params_path = "results/best_hyperparams.json"

    def set_seed(self):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)

    def run(self):
        self.set_seed()
        best = {"test_loss": float("inf")}

        for seq_len in self.seq_len_values:
            dataset = NNDataset(seq_len=seq_len)
            n = len(dataset)
            split = int(n * 0.8)

            train_loader = DataLoader(torch.utils.data.Subset(dataset, range(split)), batch_size=64, shuffle=True)
            test_loader = DataLoader(torch.utils.data.Subset(dataset, range(split, n)), batch_size=64, shuffle=False)

            for h1 in self.hidden1_values:
                for h2 in self.hidden2_values:
                    for lr in self.learning_rates:
                        for ep in self.epoch_values:
                            model = StockNN(seq_len, self.n_features, h1, h2, self.horizon)

                            train_loss = train_model(model, train_loader, epochs=ep, learning_rate=lr)
                            test_loss = test_model(model, test_loader)

                            print(f"[grid] seq={seq_len}, h1={h1}, h2={h2}, ep={ep}, lr={lr} -> test={test_loss:.6f}")

                            if test_loss < best.get("test_loss", float("inf")):
                                best.update({
                                    "test_loss": test_loss, "epochs": ep, "lr": lr,
                                    "seq_len": seq_len, "hidden1": h1, "hidden2": h2,
                                    "state_dict": copy.deepcopy(model.state_dict())
                                })

        with open(self.best_params_path, "w") as f:
            json.dump(best, f, indent=2)

        print("Best model:", best)

        best_model = StockNN(best["seq_len"], self.n_features, best["hidden1"], best["hidden2"], self.horizon)
        best_model.load_state_dict(best["state_dict"])
        best_model.eval()

        # Scale inference
        train_df = pd.concat(
            [pd.read_csv(os.path.join(self.train_stocks_dir, f)) for f in os.listdir(self.train_stocks_dir) if f.endswith(".csv")],
            ignore_index=True,
        )
        scaler = StandardScaler().fit(train_df[["Open", "High", "Low", "Close", "Adjusted", "Volume"]])

        predictions = []
        for i in range(1, 6):
            df = pd.read_csv(self.test_template.format(i))
            feats_scaled = scaler.transform(df[["Open", "High", "Low", "Close", "Adjusted", "Volume"]])

            window = feats_scaled[-best["seq_len"]:] if len(feats_scaled) >= best["seq_len"] \
                else np.vstack([np.repeat(feats_scaled[0:1], best["seq_len"] - len(feats_scaled), axis=0), feats_scaled])

            X = torch.tensor(window.reshape(1, best["seq_len"], self.n_features), dtype=torch.float32)
            with torch.no_grad():
                predictions.append(best_model(X).numpy().flatten())

        pred_matrix = list(zip(*predictions))
        dates = [
            "2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-31",
            "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07"
        ]

        submission_df = pd.DataFrame(pred_matrix, columns=[f"Returns_{i}" for i in range(1, 6)])
        submission_df.insert(0, "Date", dates)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        submission_df.to_csv(self.output_path, index=False)
        print(f"Saved {self.output_path}!")


def parse_args():
    parser = argparse.ArgumentParser(description="Run stock prediction models.")
    parser.add_argument("--model", choices=["linear", "arima", "nn"], required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    {"linear": LinearRunner, "arima": ArimaRunner, "nn": NNRunner}[args.model]().run()


if __name__ == "__main__":
    main()
