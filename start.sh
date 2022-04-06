
python port = process.env.PORT || 8000

uvicorn app.api:app --host 0.0.0.0 --port port