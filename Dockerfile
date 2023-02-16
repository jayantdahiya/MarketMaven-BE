FROM tiangolo/uvicorn-gunicorn-fastapi

WORKDIR /

COPY . .

RUN pip3 install numpy
RUN pip3 install prophet
RUN pip3 install yfinance
RUN pip3 install fastapi

# COPY . . 

EXPOSE 8000

CMD gunicorn api:app --bind 0.0.0.0:8000 --worker-class uvicorn.workers.UvicornWorker