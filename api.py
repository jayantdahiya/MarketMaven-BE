from sqlite3 import Date
import pandas as pd
import yfinance as yf
import datetime as datetime
from prophet import Prophet
from time import strftime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


def predict(ticker, start_date):
    today = datetime.date.today()
    end_date = today.strftime("%Y-%m-%d")
    data = yf.Ticker(ticker).history(period='max', interval='1d')
    
    df_forecast = data.copy()
    df_forecast.reset_index(inplace=True)
    df_forecast["ds"] = df_forecast["Date"]
    df_forecast["y"] = df_forecast["Close"]
    df_forecast = df_forecast[["ds","y"]]
    df_forecast
    
    model = Prophet()
    model.fit(df_forecast)
    
    future = pd.to_datetime(end_date) + pd.DateOffset(days=7)
    future_date = future.strftime("%Y-%m-%d")
    dates = pd.date_range(start= end_date, end= future_date)
    df_pred = pd.DataFrame({"ds" : dates})
    
    forecast = model.predict(df_pred)
    prediction_list = forecast.tail(7).to_dict("records")
    
    # output = {}

    # i=0

    # for data in prediction_list:

    #     date = data["ds"].strftime("%Y-%m-%d")
    #     output [i]= {
    #         "date" : date,
    #         "price" : data["trend"]
    #     } 
    #     i = i+1
        
    
    
    return (prediction_list)

app = FastAPI()

origins = [
    "http://localhost:8000",
    "localhost:8000",
    "localhost:3000",
    "http://localhost:3000",
    "https://stock-predictor-react-fe.web.app/",
    "https://stock-predictor-react-fe.web.app/Result",
    "stock-predictor-react-fe.web.app/",
    "stock-predictor-react-fe.web.app/Result",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class StockIn(BaseModel):
    ticker: str
    # start_date: datetime.date


class StockOut(BaseModel):
    response_object : dict




@app.post('/predict')
def prediction(payload: StockIn):
    start_date = '2019-01-01'
    ticker = payload.ticker
    # start_date = payload.start_date
    pred = predict(ticker, start_date)

    if not pred:
        raise HTTPException(status_code=400, detail="Model not found")
    
    response_object = {
        # "ticker" : ticker, 
        "forecast": pred
        }
    return (response_object)