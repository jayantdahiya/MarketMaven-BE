FROM continuumio/miniconda3
WORKDIR ./
COPY . .
RUN pip install fastapi uvicorn supabase pandas yfinance redis
RUN conda install -c conda-forge -y prophet