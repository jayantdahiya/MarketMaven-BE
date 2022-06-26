FROM continuumio/miniconda3

WORKDIR /

RUN pip install numpy
RUN pip install fastapi
RUN pip install gunicorn
RUN pip install uvicorn
RUN pip install yfinance
RUN conda install -c conda-forge prophet

COPY . . 

EXPOSE $PORT

CMD gunicorn api:app --bind 0.0.0.0:$PORT --worker-class uvicorn.workers.UvicornWorker