from sqlite3 import Date
import pandas as pd
import yfinance as yf
import datetime as datetime
from prophet import Prophet
from time import strftime


def pred(ticker):
    today = datetime.date.today()
    end_date = today.strftime("%Y-%m-%d")
    data = yf.Ticker(ticker).history(period='max', interval='1d')
    
    df_forecast = data.copy()
    df_forecast.reset_index(inplace=True)
    df_forecast["ds"] = df_forecast["Date"].dt.date
    df_forecast["y"] = df_forecast["Close"]
    df_forecast = df_forecast[["ds","y"]]
    
    model = Prophet()
    model.fit(df_forecast)
    
    future = pd.to_datetime(end_date) + pd.DateOffset(days=10)
    future_date = future.strftime("%Y-%m-%d")
    dates = pd.date_range(start= end_date, end= future_date)
    df_pred = pd.DataFrame({"ds" : dates})
    forecast = model.predict(df_pred)
    prediction_list = forecast.tail(10)

    forecast = prediction_list[['ds', 'trend']].to_json(orient='index')

    return (forecast)