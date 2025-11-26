# Stock Price Prediction Project
Predicting future stock prices using historical market data 

---
## Overview
This project explores different modeling approaches to **predict stock returns** for the next 10 trading days as part of the Kaggle **Stock Price Prediction Challenge** (hosted by Nikita Manaenkov).
- Data preprocessing & feature engineering
- Feature engineering experiments 
- Multiple predictive models

---
## Methods Used
1. Linear Regression
   - Serve as a baseline model using lagged features 
2. ARIMA Model
  - ARIMA achieved the lowest MSE in this project.
3. Neural Network
  - A simple feedforward network was implemented. Performance was weaker than ARIMA. 

---
## Feature Engineering 
Beta and momentum were derived from the datasets and added as additional predictors. 
Both features yielded worse model performance in this project, increasing MSE across all models.
After reviewing some literature, this makes sense:

**Beta**
- Beta measures a stock's sensitivity to market movements.
- Although useful in asset pricing, beta is not strongly predictive of short-term returns.
- Empirical finance shows that the CAPM beta explains cross-sectional expected returns, not short-horizon predictions.
- Beta is relatively stable, while short-term returns could be dominated by noise.

**Momentum** 
- Momentum is one of the strongest documented return anomalies for 3-12 month horizons (Jegadeesh & Titman, 1993).
- However, short-term momentum is much weaker, or can even reverse (short-term reversal effect)
  
Takeaway:
Widely used financial features do not necessarily translate to short-term forecasting. More features do not guarantee better performance, especially in noisy financial time series. 

## Datasets
The dataset comes from the Kaggle competition: Stock Price Prediction Challenge.
It contains daily price and volume data for S&P 500 stocks from 2015-01-05 to 2025-03-21, including:
- Ticker
- Open, High, Low, Close
- Adjusted Close
- Returns
- Trading volume

## How to Run

## Evaluation
Model performance is evaluated using mean squared error (MSE), consistent with the Kaggle competition's evaluation metric. 
