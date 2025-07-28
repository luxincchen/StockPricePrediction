import pandas as pd 
from src.linear_model import train, test, predict
from src.dataset import LinearDataset

#Train the model 
def main():
    dataset = LinearDataset("data/train/stocks/")

    X, Y = dataset.preprocess()
    X_test, X_train, Y_test, Y_train = dataset.split(X, Y)

    model = train(X_train, Y_train)
    
    accuracy, pred = test(model, Y_test, X_test)
    print(f"The MSE of the model is {accuracy}.")

    predictions = []
    for i in range(1, 6):
        test_path = f"data/test/test_{i}.csv"
        df = pd.read_csv(test_path)
        
        X_pred = df[["Open", "High", "Low", "Close", "Volume", "Adjusted"]].values

        Y_pred = predict(model, X_pred)
        predictions.append(Y_pred[:10].flatten())


    pred_matrix = list(zip(*predictions))

     # Create submission DataFrame
    dates = ["3/24/2025", "3/25/2025", "3/26/2025", "3/27/2025", "3/28/2025", "3/31/2025", "4/1/2025", "4/2/2025", "4/3/2025", "4/4/2025"]
    submission_df = pd.DataFrame(pred_matrix, columns=["Returns_1", "Returns_2", "Returns_3", "Returns_4", "Returns_5"])
    submission_df.insert(0, "Date", dates)

    # Save to CSV
    submission_df.to_csv("submission.csv", index=False)
    print("Saved submission.csv!")

if __name__ == "__main__":
    main()