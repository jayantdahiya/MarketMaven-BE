FROM --platform=arm64 continuumio/miniconda3

WORKDIR /

RUN pip install numpy
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install yfinance
RUN conda install -c conda-forge prophet

COPY . . 


CMD uvicorn app.api:app --reload --host 0.0.0.0 --port $PORT

















# FROM --platform=linux/arm64 continuumio/miniconda3
# ARG port

# USER root
# COPY . .
# WORKDIR /

# ENV PORT=$port

# RUN chgrp -R 0 /stock-predictor-api1 \
#     && chmod -R g=u /stock-predictor-api1

# EXPOSE $PORT

# COPY environment.yml ./
# RUN conda env create -f environment.yml
# SHELL [ "conda", "run", "-n", "api", "/bin/bash", "-c" ]
# RUN echo "conda activate api" >> ~/.bashrc
# RUN conda install -c conda-forge \
#                          prophet \
#                          uvicorn \
#                          fastapi \
#                          yfinance \
#                          pandas


    

# CMD uvicorn api:app --host 0.0.0.0 --port $PORT
# CMD gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT api:app
# CMD gunicorn api:app -bind 0.0.0.0:$PORT --preload





# -----------------------------------------------------------------------------------------------






# FROM python:3.8-slim-buster

# WORKDIR /app

# RUN apt-get -y update  && apt-get install -y \
#   python3-dev \
#   apt-utils \
#   python-dev \
#   build-essential \
#   gcc mono-mcs \
# && rm -rf /var/lib/apt/lists/*

# RUN pip install --upgrade setuptools
# RUN pip install \
#     cython==0.29.24 \
#     numpy==1.21.1 \
#     pandas==1.3.1 

# COPY requirements.txt .
# RUN pip install -r requirements.txt

# COPY . .

# CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT


