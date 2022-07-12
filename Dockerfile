FROM tiangolo/uvicorn-gunicorn-fastapi

WORKDIR /

COPY . .

RUN pip3 install numpy
RUN pip3 install prophet
RUN pip3 install yfinance
RUN pip3 install fastapi

# COPY . . 

EXPOSE $PORT

CMD gunicorn api:app --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker