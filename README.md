# MarketMaven-BE

<p>
This repository contains the backend code for the forecasting application. The backend is responsible for receiving requests from the frontend and returning the forecasted results.
</p>

## Architecture

<p>The backend is designed to work with the following architecture:</p>

1. The frontend sends requests to the backend through Cloudflare Tunnel using HTTPS.
2. The NGINX Proxy (on AWS instance) receives the requests and triggers the Gunicorn services (FastAPI).
3. FastAPI routes the request to the main.py file.
4. The main.py file receives the ticker name and model name as input.
5. The code uses yFinance to fetch the historical data of the ticker to train the model.
6. The model forecasts the result for 10 days from the current date.
7. The result is returned back in the JSON format to the frontend.

## Installation and Setup

1. Clone this repository.
2. Install the required packages using `pip install -r requirements.txt`
3. Start the Gunicorn server using `gunicorn main:app`
4. Open `localhost:3000/docs` on your browser to try it yourself.

## Usage

Send a POST request to the endpoint with the following JSON payload:

```json
{ 
   "ticker": "<ticker_name>",
   "model": "<model_name>"
}
```

Where `<ticker_name>` is the name of the stock ticker and `<model_name>` is the name of the forecasting model.

The backend will respond with a JSON object containing the forecasted results for the next 10 days. The format of the response is as follows:

```json
{
    "ticker": "<ticker_name>",
    "model": "<model_name>",
    "forecast": [
        {
            "date": "<date>",
            "value": "<value>"
        },
        ...
    ]
}
```
Where `<date>` is the date of the forecasted value and `<value>` is the forecasted value for that date.
