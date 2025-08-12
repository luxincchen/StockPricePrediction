from sklearn.linear_model import LinearRegression
from src.metrics import MSE

def train(X_train, Y_train):

    model = LinearRegression()
    return model.fit(X_train, Y_train)

def test(model, X_test, Y_test):
    pred = model.predict(X_test)
    return MSE(pred, Y_test), pred

def predict(model, X):
    return model.predict(X)