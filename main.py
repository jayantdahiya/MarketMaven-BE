from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Models.Prophet import pred as Prophet_Forecast

app = FastAPI()

origins = [
    "http://localhost:8000",
    "localhost:8000",
    "http://127.0.0.1:5173/result",
    "http://127.0.0.1:5173/",
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


@app.get('/')
def index():
    return {'Stock Predictor API'}

# This is the route that is called when the user wants the Prophet model
@app.get('/Prophet')
def Prophet(ticker: str):
    return Prophet_Forecast(ticker)