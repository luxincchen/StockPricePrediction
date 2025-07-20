import numpy as np

def MSE(prediction, target):
    return np.mean((target - prediction)**2)