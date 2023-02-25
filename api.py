from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from Models.Prophet import pred as Prophet_Forecast

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000",
    "https://localhost:8000",
    "https://0.0.0.0:8000",
    "https://5173-jayantdahiy-stockpredic-fmaagmgmu8d.ws-us87.gitpod.io"
]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
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