FROM continuumio/miniconda3

WORKDIR /

COPY environment.yml ./
RUN conda env create -f environment.yml
SHELL [ "conda", "run", "-n", "api", "/bin/bash", "-c" ]
RUN echo "conda activate api" >> ~/.bashrc
RUN conda install -c conda-forge prophet
RUN pip install \
        uvicorn \
        fastapi \
        yfinance \
        pandas 
    

# COPY main.py boot.sh /
# RUN chmod +x boot.sh
# ENTRYPOINT ["python3", "main.py"]
EXPOSE 8000
COPY api.py main.py /
CMD python main.py


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


