# Use the official Miniconda image
FROM continuumio/miniconda3

WORKDIR /app

# Copy project files
COPY . .

# Install pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy config for runtime
COPY config/ config/

# Local Docker defaults to 8000 via ${PORT:-8000}. Render sets PORT (often 10000).
EXPOSE 8000

# Run the application — canonical entrypoint api.app:app
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
