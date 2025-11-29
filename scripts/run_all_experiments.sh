#!/usr/bin/env bash

set -e

python -m src.run --model linear
python -m src.run --model arima
python -m src.run --model nn
