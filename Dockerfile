# Use an official Python runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Run the training script
CMD ["python", "example.py"]