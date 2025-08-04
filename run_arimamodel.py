import pandas as pd
from src.arima_model import train, predict
from src.dataset import ARIMADataset

def main():
    predictions = []

    for i in range(1, 6):
        test_path = f"data/test/test_{i}.csv"
        df = pd.read_csv(test_path)
        returns = df["Returns"].dropna()
        
        model = train(returns)
        forecast = predict(model, 10)

        predictions.append(forecast.values)

        #accuracy, forecast = model.get_forecast(steps=10)
        #print(f"The MSE of the model is {accuracy}.")

     # Create submission DataFrame
    dates = ["2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07"]
    
    pred_matrix = list(zip(*predictions))

    submission_df = pd.DataFrame(pred_matrix, columns=["Returns_1", "Returns_2", "Returns_3", "Returns_4", "Returns_5"])
    submission_df.insert(0, "Date", dates)

    # Save to CSV
    submission_df.to_csv("submission_arima.csv", index=False)
    print("Saved submission_arima.csv!")

if __name__ == "__main__":
    main()

