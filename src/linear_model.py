from dataset import LinearDataset
from sklearn.linear_model import LinearRegression
from metrics import MSE

def train(X_train, Y_train):

    model = LinearRegression()
    return model.fit(X_train, Y_train)

def test(model, Y_test, X_test):
    pred = model.predict(X_test)
    return MSE(pred, Y_test)

def main():
    dataset = LinearDataset("data/train/stocks/")

    X, Y = dataset.preprocess()
    X_test, X_train, Y_test, Y_train = dataset.split(X, Y)

    model = train(X_train, Y_train)
    
    accuracy = test(model, Y_test, X_test)
    print(f"The MSE of the model is {accuracy}.")

if __name__ == "__main__":
    main()