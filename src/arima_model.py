from statsmodels.tsa.arima.model import ARIMA
from src.metrics import MSE

def train(train_df):
    model = ARIMA(train_df, order=(1, 0, 1))
    return model.fit()

def predict(model, X):
    return model.get_forecast(X).predicted_mean

