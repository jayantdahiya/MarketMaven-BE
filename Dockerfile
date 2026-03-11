# Use the official Miniconda image
FROM continuumio/miniconda3

WORKDIR /app

# Copy project files
COPY . .

# Install pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy config for runtime
COPY config/ config/

# Expose the port used by your application
EXPOSE 8000

# Run the application — canonical entrypoint api.app:app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
