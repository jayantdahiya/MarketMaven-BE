from fastapi import FastAPI
from pydantic import BaseModel

from Models.Prophet import pred as Prophet_Forecast

import logging
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)


class StockIn(BaseModel):
    ticker: str


@app.get('/')
def index():
    return {'Stock Predictor API'}

# This is the route that is called when the user wants the Prophet model
@app.get('/prophet')
def Prophet(ticker: str):
    return Prophet_Forecast(ticker)

@app.get('/tickers')
def get_tickers():
    tickers = supabase.table('tickers').select('*').execute().data
    return tickers
