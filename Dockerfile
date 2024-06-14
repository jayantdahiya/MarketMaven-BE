# Dockerfile

# Use the official Miniconda image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the current directory to /app in the container
COPY . .

# Install prophet using conda from conda-forge channel
RUN conda install -c conda-forge prophet -y

# Install pip packages based on requirements.txt
RUN pip install -r requirements.txt

# Expose the port used by your application
EXPOSE 8000

# Specify the command to run your application
CMD cd api && python main.py
