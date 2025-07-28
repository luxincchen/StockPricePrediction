from statsmodels.tsa.arima.model import ARIMA
from src.metrics import MSE

def train(train_df):
    model = ARIMA(train_df, order=(1, 0, 1))
    return model.fit()

def test(model, test_df):
    forecast = model.get_forecast(steps = len(test_df))
    mean_forecast = forecast.predicted_mean
    return MSE(mean_forecast, test_df), mean_forecast

def predict(model, X):
    return model.get_forecast(X).predicted_mean

