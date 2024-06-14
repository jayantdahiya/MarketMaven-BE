from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from Models.Prophet import pred as Prophet_Forecast

import logging
import os
from supabase import create_client, Client
from dotenv import load_dotenv

import json
import redis

load_dotenv()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

# Redis configuration
redisDB = redis.Redis(
    host=os.environ.get("REDIS_HOST"),
    port=6379,
    password=os.environ.get("REDIS_PASSWORD"),
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
    return {'message': 'Welcome to the Market Maven API'}


@app.get('/prophet')
def Prophet(ticker: str):
    return Prophet_Forecast(ticker)


@app.get('/tickers')
async def get_tickers():
    try:
        cached_tickers = redisDB.get('tickers')
        if cached_tickers:
            return json.loads(cached_tickers)

        tickers = supabase.table('tickers').select('*').execute().data
        redisDB.set('tickers', json.dumps(tickers))

        return tickers
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch tickers")


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
        "email": login_request.email,
        "password": login_request.password
    }
    try:
        user = supabase.auth.sign_in_with_password(credentials)
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials") from e


@app.get('/logout')
async def logout():
    res = supabase.auth.sign_out()
    return res
