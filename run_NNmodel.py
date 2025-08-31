import os, json, copy, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.NN_model import StockNN, train_model, test_model
from src.dataset import NNDataset  # your Dataset class

def main():
    # Reproducibility
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    n_features = 6   # Open, High, Low, Close, Adjusted, Volume
    horizon = 10

    # =========================
    # Hyperparameter search space
    # =========================
    seq_len_values   = [10, 20, 50]        # lookback window
    hidden1_values   = [64, 128]           # first hidden layer size
    hidden2_values   = [128, 256]          # second hidden layer size
    learning_rates   = [1e-4, 1e-3]
    epoch_values     = [5, 10, 15]

    results = []
    best = {
        "test_loss": float("inf"),
        "epochs": None,
        "lr": None,
        "seq_len": None,
        "hidden1": None,
        "hidden2": None,
        "state_dict": None,
    }

    # =========================
    # Grid search
    # =========================
    for seq_len in seq_len_values:
        train_dataset = NNDataset(seq_len=seq_len)
        n = len(train_dataset)
        split = int(n * 0.8)
        train_loader = DataLoader(torch.utils.data.Subset(train_dataset, range(split)), batch_size=64, shuffle=True)
        test_loader  = DataLoader(torch.utils.data.Subset(train_dataset, range(split, n)), batch_size=64, shuffle=False)

        for h1 in hidden1_values:
            for h2 in hidden2_values:
                for lr in learning_rates:
                    for ep in epoch_values:
                        model = StockNN(seq_len=seq_len, n_features=n_features,
                                        hidden1=h1, hidden2=h2, horizon=horizon)

                        train_loss = train_model(model, train_loader, epochs=ep, learning_rate=lr)
                        test_loss  = test_model(model,  test_loader)

                        results.append({
                            "seq_len": seq_len,
                            "hidden1": h1,
                            "hidden2": h2,
                            "epochs": ep,
                            "lr": lr,
                            "train_loss": float(train_loss),
                            "test_loss": float(test_loss),
                        })

                        print(f"[grid] seq={seq_len}, h1={h1}, h2={h2}, "
                              f"ep={ep}, lr={lr} -> train={train_loss:.6f}, test={test_loss:.6f}")

                        if test_loss < best["test_loss"]:
                            best.update({
                                "test_loss": float(test_loss),
                                "epochs": ep,
                                "lr": lr,
                                "seq_len": seq_len,
                                "hidden1": h1,
                                "hidden2": h2,
                                "state_dict": copy.deepcopy(model.state_dict())
                            })

    # Save best hyperparams
    with open("best_hyperparams.json", "w") as f:
        json.dump({
            "epochs": best["epochs"],
            "learning_rate": best["lr"],
            "seq_len": best["seq_len"],
            "hidden1": best["hidden1"],
            "hidden2": best["hidden2"],
            "test_loss": best["test_loss"]
        }, f, indent=2)

    print(f"✅ Best combo: seq_len={best['seq_len']}, h1={best['hidden1']}, h2={best['hidden2']}, "
          f"epochs={best['epochs']}, lr={best['lr']}, "
          f"test_loss={best['test_loss']:.6f} (saved to best_hyperparams.json)")

    # Restore best model
    best_model = StockNN(seq_len=best["seq_len"], n_features=n_features,
                         hidden1=best["hidden1"], hidden2=best["hidden2"], horizon=horizon)
    best_model.load_state_dict(best["state_dict"])
    best_model.eval()

    # =========================
    # Scaler fit ONLY on training features (no leakage)
    # =========================
    train_df = pd.concat([
        pd.read_csv(os.path.join("data/train/stocks", f))
        for f in os.listdir("data/train/stocks/")
        if f.endswith(".csv")
    ], ignore_index=True)

    train_features = train_df[["Open", "High", "Low", "Close", "Adjusted", "Volume"]].to_numpy()
    scaler = StandardScaler().fit(train_features)

    # =========================
    # Inference: use last seq_len rows of each test file
    # =========================
    predictions = []
    seq_len = best["seq_len"]

    for i in range(1, 6):   # test_1..test_5
        test_path = f"data/test/test_{i}.csv"
        test_df = pd.read_csv(test_path)

        feats = test_df[["Open", "High", "Low", "Close", "Adjusted", "Volume"]].to_numpy()
        feats_scaled = scaler.transform(feats)

        if len(feats_scaled) < seq_len:
            pad = np.repeat(feats_scaled[0:1, :], seq_len - len(feats_scaled), axis=0)
            window = np.vstack([pad, feats_scaled])
        else:
            window = feats_scaled[-seq_len:]

        X_tensor = torch.tensor(window.reshape(1, seq_len, n_features), dtype=torch.float32)

        with torch.no_grad():
            y_pred = best_model(X_tensor).numpy().flatten()

        predictions.append(y_pred[:horizon])

    # transpose: rows = days, cols = test_i
    pred_matrix = list(zip(*predictions))
    dates = [
        "2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-31",
        "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07"
    ]
    submission_df = pd.DataFrame(
        pred_matrix,
        columns=["Returns_1", "Returns_2", "Returns_3", "Returns_4", "Returns_5"]
    )
    submission_df.insert(0, "Date", dates)
    submission_df.to_csv("submission_NN_best.csv", index=False)
    print("📁 Saved submission_NN_best.csv")

if __name__ == "__main__":
    main()
