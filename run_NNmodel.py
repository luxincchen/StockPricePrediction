import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.NN_model import StockNN, train_model
from src.dataset import NNDataset

def main():
    dataset = NNDataset("data/train/stocks/")
    model = StockNN()
    predictions = []

    for i in range(1, 6):
        path = f"data/test/test_{i}.csv"
        df = pd.read_csv(path)

        # 🧠 Preprocess the test data
        df = df[["Open", "High", "Low", "Close", "Adjusted", "Volume", "Returns"]]
        features = df[["Open", "High", "Low", "Close", "Adjusted", "Volume"]].to_numpy()

        # Use last 10 days as input
        X_input = features[-10:].reshape(1, 10, 6)  # batch of one
        X_tensor = torch.tensor(X_input, dtype=torch.float32)

        with torch.no_grad():
            Y_pred = model(X_tensor).numpy().flatten()

        predictions.append(Y_pred[:10])  # only take 10-day forecast

    # 🔽 Combine into matrix: each column is one stock
    pred_matrix = list(zip(*predictions))

    # 📅 Dates for submission
    dates = ["2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-31",
             "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07"]
    
    submission_df = pd.DataFrame(pred_matrix, columns=[
        "Returns_1", "Returns_2", "Returns_3", "Returns_4", "Returns_5"
    ])
    submission_df.insert(0, "Date", dates)

    submission_df.to_csv("submission_NN.csv", index=False)
    print("📁 Saved submission_NN.csv!")

if __name__ == "__main__":
    main()