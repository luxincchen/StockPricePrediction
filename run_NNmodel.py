import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.NN_model import StockNN, train_model
from src.dataset import NNDataset
from sklearn.preprocessing import StandardScaler

def main():
    # 1️⃣ Load training dataset
    dataset = NNDataset("data/train/stocks/")
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2️⃣ Build and train model
    model = StockNN()
    train_model(model, train_loader, epochs=10, learning_rate=0.001)

    # 3️⃣ Predict returns for test sets
    predictions = []

    for i in range(1, 6):
        path = f"data/test/test_{i}.csv"
        # Load full training dataset to fit the scaler
        train_df = pd.concat([pd.read_csv(f"data/train/stocks/{f}") for f in os.listdir("data/train/stocks/")])
        train_features = train_df[["Open", "High", "Low", "Close", "Adjusted", "Volume"]].to_numpy()

        # Fit the same scaler
        scaler = StandardScaler()
        scaler.fit(train_features)

        # Apply to test input
        features = train_df[["Open", "High", "Low", "Close", "Adjusted", "Volume"]].to_numpy()
        features_scaled = scaler.transform(features)
        X_input = features_scaled[-10:].reshape(1, 10, 6)

        X_tensor = torch.tensor(X_input, dtype=torch.float32)

        # 🔮 Inference
        with torch.no_grad():
            Y_pred = model(X_tensor).numpy().flatten()

        predictions.append(Y_pred[:10])  # 10-day forecast

    # 🧾 Format submission
    pred_matrix = list(zip(*predictions))
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