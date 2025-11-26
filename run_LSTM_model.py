# run_LSTMModel.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.dataset import LSTMDataset
from src.LSTM_model import LSTMModel


def main():
    #load dataset
    dataset = LSTMDataset(dir='data/train/stocks/', seq_len=50, horizon=10)

    #train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 3. Training loop
    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

     
    # Forecast for each test file
    predictions = []

    for i in range(1, 6):  # 5 test files
        test_path = f"data/test/test_{i}.csv"
        df = pd.read_csv(test_path)

        # Use only features your LSTM expects
        features = df[['Open','High','Low','Close','Adjusted','Volume','Returns']].dropna()
        scaled = dataset.scaler.transform(features)  # reuse scaler from training dataset

        # Take the last `seq_len` window as input
        last_window = scaled[-dataset.seq_len:, :-1]   # exclude Returns col
        last_window = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)  
        # shape = (1, seq_len, 6)

        model.eval()
        with torch.no_grad():
            forecast = model(last_window).cpu().numpy().flatten()  # (10,)

        predictions.append(forecast)

 
    # Create submission DataFrame
  
    dates = ["2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-31",
        "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07"]

    # Transpose: rows = dates, cols = test files
    pred_matrix = list(zip(*predictions))

    submission_df = pd.DataFrame(pred_matrix, columns=["Returns_1", "Returns_2", "Returns_3", "Returns_4", "Returns_5"])
    submission_df.insert(0, "Date", dates)

    submission_df.to_csv("submission_LSTM.csv", index=False)
    print("Saved submission_LSTM.csv")
    
if __name__ == "__main__":
    main()
