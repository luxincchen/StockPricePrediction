import pandas as pd 
from src.linear_model import train, test, predict
from src.new_dataset import LinearDataset
from src.features import beta_features

#Train the model 
def main():
    
    dataset = LinearDataset("data/train/stocks/")
    X, Y = dataset.preprocess()
    X_test, X_train, Y_test, Y_train = dataset.split(X, Y)

    model = train(X_train, Y_train)
    
    accuracy, pred = test(model, X_test, Y_test)
    print(f"The MSE of the model is {accuracy}.")

    # predictions = []
    # for i in range(1, 6):
    #     test_path = f"data/test/test_{i}.csv"
    #     df = pd.read_csv(test_path)
    #     df = beta_features(stocks_dir=test_path, sp500_path="data/train/indices/SP500.csv")

    #     X_pred = df[["Open", "High", "Low", "Close", "Adjusted", "Volume", "beta_lag1"]].values
    #     X_pred = X_pred[:-10] 

    #     Y_pred = predict(model, X_pred)
    #     predictions.append(Y_pred[:10].flatten())

    # pred_matrix = list(zip(*predictions))

    #  # Create submission DataFrame
    # dates = ["2025-03-25", "2025-03-26", "2025-03-27", "2025-03-28", "2025-03-31", "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07"]
    # submission_df = pd.DataFrame(pred_matrix, columns=["Returns_1", "Returns_2", "Returns_3", "Returns_4", "Returns_5"])
    # submission_df.insert(0, "Date", dates)

    # # Save to CSV
    # submission_df.to_csv("submission.csv", index=False)
    print("Saved submission.csv!")

if __name__ == "__main__":
    main()