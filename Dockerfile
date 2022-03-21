FROM continuumio/miniconda3

WORKDIR /

COPY environment.yml ./
RUN conda env create -f environment.yml

SHELL [ "conda", "run", "-n", "env_11", "/bin/bash", "-c" ]


RUN echo "Running prophet......."
RUN python -c "import prophet"

COPY main.py .
ENTRYPOINT [ "conda", "run", "-n", "env_11", "python", "main.py" ]

EXPOSE 5000


