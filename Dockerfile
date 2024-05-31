FROM continuumio/miniconda3
WORKDIR ./
COPY environment.yml environment.yml
RUN pip install fastapi uvicorn supabase pandas yfinance redis
RUN conda install -c conda-forge -y prophet