# Dockerfile for Recommender Service

# Use official slim Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install build dependencies for compiling LightFM and implicit
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential python3-dev libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port for FastAPI
EXPOSE 8080

# Start FastAPI with Uvicorn
# по умолчанию слушаем на 8080 (Cloud Run передаёт в PORT)
ENV PORT 8080

# запускаем uvicorn, подхватывая порт из переменной
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]

