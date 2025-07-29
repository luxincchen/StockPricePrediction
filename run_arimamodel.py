import pandas as pd
from src.arima_model import train, predict
from src.metrics import MSE

def main():
    predictions = []

    for i in range(1, 6):
        test_path = f"data/test/test_{i}.csv"
        df = pd.read_csv(test_path)
        returns = df["Returns"].dropna()

        # Use 80% for training, 20% for internal testing
        split_index = int(len(returns) * 0.8)
        train_series = returns[:split_index]
        test_series = returns[split_index:]

        # Train model on 80% of the data
        model = train(train_series)

        # Forecast the length of the held-out 20%
        forecast_for_eval = predict(model, len(test_series))

        # Compute MSE
        mse = MSE(forecast_for_eval, test_series)
        print(f"Model for test_{i}: MSE on internal validation set = {mse:.8f}")

        # Now forecast the next 10 values using all data
        full_model = train(returns)
        forecast = predict(full_model, 10)

        predictions.append(forecast.values)

    # Create submission DataFrame
    dates = [
        "2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-31",
        "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07"
    ]

    pred_matrix = list(zip(*predictions))
    submission_df = pd.DataFrame(pred_matrix, columns=[
        "Returns_1", "Returns_2", "Returns_3", "Returns_4", "Returns_5"
    ])
    submission_df.insert(0, "Date", dates)

    submission_df.to_csv("submission_arima.csv", index=False)
    print("Saved submission_arima.csv!")

if __name__ == "__main__":
    main()


