import src.linear_model as linear 
import src.arima_model as arima

def create_submission_csv(model_class):
    model_class.train