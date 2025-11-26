# Stock Price Prediction Project
Predicting future stock prices using historical market data.


## Overview
This project explores different modeling approaches to **predict stock returns** for the next 10 trading days as part of the Kaggle **Stock Price Prediction Challenge** (hosted by Nikita Manaenkov). This project is done in collaboration with Maarten Drinhuyzen. 

The workflow includes:
- Data preprocessing
- Feature engineering
- Training multiple predictive models
- Evaluating performance using Mean Squared Error (MSE)


## Methods Used
1. Linear Regression
- Serve as a baseline model using lagged features 
2. ARIMA Model
- ARIMA achieved the lowest MSE in this project, capturing autocorrelated patterns in returns. 
3. Neural Network
- A simple feedforward network was implemented. Performance was weaker than ARIMA. 


## Feature Engineering 
Two additional financial features were tested: beta and momentum. 
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

## Repository Structure
Main directory
```
├── README.md
├── requirements.txt            # Project dependencies
├── run_arimamodel.py           # Script to train ARIMA model + generate predictions
├── run_linearmodel.py          # Script to train linear regression model + generate predictions
├── run_NNmodel.py              # Script to train neural network model + generate predictions
├── loss_curve.png              # Neural network training loss plot
├── loss_comparison.png         # MSE comparison plot
├── best_hyperparams.json       # Saved hyperparameters for NN
├── submission.csv              # Linear regression model Kaggle submission
├── submission_NN.csv           # NN Kaggle submission
├── submission_NN_best.csv      # NN with tuned parameters Kaggle submission
├── submission_arima.csv        # ARIMA Kaggle submission
├── test.ipynb                  # Notebook for testing
```
src/folder
```
├── __init__.py
├── __pycache__/
├── .DS_Store
├── NN_model.py            # Neural network architecture & training utilities
├── arima_model.py         # ARIMA model implementation
├── linear_model.py        # Linear regression implementation
├── dataset.py             # Data loading & preparation
├── new_dataset.py         # Dataset with features 
├── features.py            # Feature engineering (beta, momentum, etc.)
├── metrics.py             # MSE function 
└── create_submission.py   # Outputs Kaggle-formatted submission file
```

## How to Run
1. Clone the repository
   ```
   git clone https://github.com/luxincchen/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Prepare the dataset
- Download the S&P 500 datasets
   ```
   import yfinance as yf
   sp500 = yf.download("^GSPC", start="2015-01-05", end="2025-03-21")
   ```
- Place the CSV files in your working directory
- Paths may be adjusted in src/dataset.py and/pr src/new_dataset.py
4. Run models
    ```
    python run_linearmode.py
    python run_arimamodel.py
    python run_NNmodel.py
    ```
- Each script trains the model and generates a submission file
## Evaluation
Model performance is evaluated using mean squared error (MSE), consistent with the Kaggle competition's evaluation metric. 
Across the tested models, ARIMA achieved the best performance (MSE = 0.000777535246), which placed us 9th out of 17 participants in this Kaggle challenge. 
