from src.arima_model import train, test
from src.dataset import ARIMADataset

def main():
    dataset = ARIMADataset("data/train/stocks/AAPL.csv")

    df = dataset.preprocess()
    train_df, test_df = dataset.split(df)

    model = train(train_df)
    
    accuracy, forecast = test(model, test_df)
    print(f"The MSE of the model is {accuracy}.")
    return forecast

if __name__ == "__main__":
    main()