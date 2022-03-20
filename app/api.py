from email.policy import default
from pickle import GLOBAL
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.model_selection as model_selection
from datetime import date
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel




def stock_prediction(stock_name):
    date_today = date.today()

    data = yf.download(stock_name, start = "2013-10-10", end=date_today)
    df = pd.DataFrame(data)
    x = df.loc[:,'High':'Volume']
    y = df.loc[:,'Open']

    x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size= 0.2,random_state = 0)
    LR = LinearRegression()
    LR.fit(x_train,y_train)
    LR.score(x_test,y_test)

    x_test = pd.DataFrame(x)
    ticker = yf.Ticker(stock_name)
    stock_today = ticker.history(period="1d")
    today = pd.DataFrame(stock_today["Open"])

    test_data = x_test.iloc[-1:]
    prediction = LR.predict(test_data).round(2)

    return (int(prediction))




# print(stock_prediction(stock_name))


app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)




# @app.get("/", tags=["root"])
# async def read_root() -> dict:
#     return {"message":"Hello world"}

class StockIn(BaseModel):
    stock_name: str

class StockOut(BaseModel):
    a : dict




@app.post("/predict")
def prediction(payload: StockIn):
    stock_name = payload.stock_name
    pred = stock_prediction(stock_name)
    a = int(pred)
    response_object = {"data" : a }
    return response_object

# @app.get('/')
# def prediction(input_name):
#     pred = stock_prediction(input_name)
#     a = int(pred)
#     return (a)

