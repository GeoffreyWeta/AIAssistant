# syntax=docker/dockerfile:1.6

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Faster pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Cache deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Ensure runtime dirs exist
RUN mkdir -p vector_db temp_uploads reports

EXPOSE 8000

# Default to prod style start, compose will override for dev
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "75"]
