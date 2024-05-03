FROM continuumio/miniconda3
WORKDIR ./
COPY environment.yml environment.yml
RUN conda env create --name marketMaven --file environment.yml