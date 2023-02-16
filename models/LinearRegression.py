import pandas as pd
import numpy as np
import pandas_ta as ta 
import yfinance as yf

import sklearn.linear_model as LinearRgression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def LinearRegressionModel(ticker): 
    data = yf.Ticker(ticker).history(period="max", interval="1d")
    data = data[['Close']]
    data.index = data.index.date
    data.set_index(pd.DatetimeIndex(data.index), inplace=True)
    data.ta.ema(close='Close', length=20, append=True)
    data = data.iloc[10:]

    

