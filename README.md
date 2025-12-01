
# Stock Price Prediction Project

Predicting future stock prices using historical market data.

## Overview

This project explores different modeling approaches to predict stock returns for the next 10 trading days as part of the Kaggle Stock Price Prediction Challenge (hosted by Nikita Manaenkov). This project is done in collaboration with Maarten Drinhuyzen.

The workflow includes:

* Data preprocessing
* Feature engineering
* Training multiple predictive models
* Evaluating performance using Mean Squared Error (MSE)

## Methods Used

1. Linear Regression
   - Served as a baseline model using lagged features.

2. ARIMA Model
   - Achieved the lowest MSE in this project, capturing autocorrelated patterns in returns.

3. Neural Network
   - A simple feed-forward network was implemented. Performance was weaker than ARIMA.

## Feature Engineering

Two additional financial features were tested: beta and momentum. Both features yielded worse model performance in this project, increasing MSE across all models.

This aligns with financial literature:

Beta

* Measures a stock's sensitivity to market movements.
* Useful in CAPM and risk modeling, but weak for short-term return prediction.
* Explains cross-sectional expected returns, not short-term price movement.

Momentum

* Documented anomaly on 3–12 month horizons (Jegadeesh & Titman, 1993).
* Short-term effects are weak and often reverse due to mean reversion.

Conclusion:
Widely used financial features do not necessarily translate to short-term prediction. More features do not guarantee better performance in noisy financial signals.

## Dataset

The dataset comes from the Kaggle Stock Price Prediction Challenge.
It contains daily S&P 500 historical price data including:

* Ticker
* OHLC prices
* Adjusted close
* Returns
* Trading volume

## Repository Structure

```
├── README.md
├── requirements.txt
├── scripts/
│   └── run_all_experiments.sh
├── results/
│   ├── submission.csv
│   ├── submission_arima.csv
│   ├── submission_NN_best.csv
│   └── best_hyperparams.json
└── src/
    ├── run.py
    ├── linear_model.py
    ├── arima_model.py
    ├── NN_model.py
    ├── dataset.py
    ├── new_dataset.py
    ├── features.py
    └── metrics.py
```

## How to Run

1. Clone the repository

```
git clone https://github.com/luxincchen/StockPricePrediction.git
cd StockPricePrediction
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Prepare the dataset

Download S&P500 index data if needed:

```python
import yfinance as yf
sp500 = yf.download("^GSPC", start="2015-01-05", end="2025-03-21")
```

Place the CSVs inside the `data/` directory. Paths may be adjusted in `src/dataset.py` and `src/new_dataset.py`.

4. Run models

```
python -m src.run --model linear
python -m src.run --model arima
python -m src.run --model nn
```

Or run all sequentially:

```
./scripts/run_all_experiments.sh
```

## Evaluation

Model performance is evaluated using Mean Squared Error (MSE), consistent with the Kaggle challenge metric.

Across all experiments, the ARIMA model achieved the best performance with an MSE of:

```
0.000777535246
```

This placed the submission 9th out of 17 participants in the challenge.

## Conclusion

Short-term stock return prediction remains difficult due to noise and low signal-to-noise ratio. ARIMA performed best in this context, likely due to its ability to model autocorrelation patterns in returns.

Possible improvements include:

* LSTM or Transformer sequence models
* GARCH or other volatility models
* Cross-sectional market or sector features

