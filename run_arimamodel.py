from src.arima_model import train, test, predict
from src.dataset import ARIMADataset

def main():
    dataset = ARIMADataset("data/train/stocks/AAPL.csv")

    df = dataset.preprocess()
    train_df, test_df = dataset.split(df)

    model = train(train_df)
    
    accuracy, forecast = test(model, test_df)
    print(f"The MSE of the model is {accuracy}.")
    return forecast

def create_submission():
    dataset = ARIMADataset("data/train/stocks/AAPL.csv")

    df = dataset.preprocess()
    train_df, test_df = dataset.split(df)

    model = train(train_df)
    
    


if __name__ == "__main__":
    main()

