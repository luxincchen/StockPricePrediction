from src.linear_model import train, test
from src.dataset import LinearDataset

def main():
    dataset = LinearDataset("data/train/stocks/")

    X, Y = dataset.preprocess()
    X_test, X_train, Y_test, Y_train = dataset.split(X, Y)

    model = train(X_train, Y_train)
    
    accuracy, pred = test(model, Y_test, X_test)
    print(f"The MSE of the model is {accuracy}.")
    return pred

if __name__ == "__main__":
    main()