# Ingestion Service Dockerfile - Heavy-duty container for repository processing
FROM python:3.11-slim

WORKDIR /app

# Install git for GitPython
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app/src

# Run the Ingestion Service
ENTRYPOINT ["python", "-m", "ingestion_service.main"]