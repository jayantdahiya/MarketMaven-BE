# Use the official Miniconda image
FROM continuumio/miniconda3

# Set the working directory
# WORKDIR /api

# Copy only the environment files first
COPY environment.yml requirements.txt ./

# Install pip packages
RUN pip install --no-cache-dir -r requirements.txt

# Install conda packages
RUN conda env update --file environment.yml

# Copy the rest of the application files
COPY . .

# Expose the port used by your application
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--port", "8000", "--reload"]
