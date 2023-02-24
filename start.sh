#! /bin/bash

conda activate api
python main.py

ngrok http 8000