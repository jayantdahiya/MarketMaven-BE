# Use the official Miniconda image
FROM continuumio/miniconda3

# Set the working directory
# WORKDIR /app

# Copy only the environment files first
COPY . .

# Install pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Install conda packages
RUN conda env update --file environment.yml

# Expose the port used by your application
EXPOSE 8000

CMD ["cd", "api"]

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
