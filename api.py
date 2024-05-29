from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Models.Prophet import pred as Prophet_Forecast

import logging
import os
from supabase import create_client, Client
from dotenv import load_dotenv

import json
import redis

load_dotenv()

app = FastAPI()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

redisDB = redis.Redis(
  host= os.environ.get("REDIS_HOST"),
  port=6379,
  password= os.environ.get("REDIS_PASSWORD"),
  ssl=True
)


class StockIn(BaseModel):
    ticker: str
    
class LoginRequest(BaseModel):
    email: str
    password: str
    
class SignUpRequest(BaseModel):
    email: str
    password: str



@app.get('/')
def index():
    return {'Stock Predictor API'}

# This is the route that is called when the user wants the Prophet model
@app.get('/prophet')
def Prophet(ticker: str):
    return Prophet_Forecast(ticker)

@app.get('/tickers')
async def get_tickers():
    
    try:
        cached_tickers = redisDB.get('tickers')
        
        if cached_tickers:
            logging.error("Using cached tickers")
            return cached_tickers
        
        logging.error("Fetching tickers from supabase")
        tickers = supabase.table('tickers').select('*').execute().data
        
        logging.error("Caching tickers")
        redisDB.set('tickers', json.dumps(tickers))
        
        return tickers
    except Exception as e:
        logging.error(f"Failed to fetch tickers: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tickers") from e


@app.post('/signup')
async def signup(signup_request: SignUpRequest):
    credentials = {
    "email": signup_request.email,
    "password": signup_request.password
    }
    user = supabase.auth.sign_up(credentials)
    
    return user


@app.post('/login')
async def login(login_request: LoginRequest):
    
    credentials = {
        "email" : login_request.email,
        "password" : login_request.password
    }
    
    try:
        user = supabase.auth.sign_in_with_password(credentials)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials") from e
        return
    
    return user

@app.get('/logout')
async def logout():
    res = supabase.auth.sign_out()
    return res