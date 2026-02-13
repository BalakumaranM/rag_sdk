# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip and install the package in editable mode with a longer timeout
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 -e ".[dev]"

# Default command
CMD ["python", "main.py"]
