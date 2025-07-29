# Dockerfile for Recommender Service

# 1. Базовый образ Python 3.11 slim
FROM python:3.11-slim

# 2. Рабочая директория
WORKDIR /app

# 3. Сборка нативных зависимостей
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential python3-dev libopenblas-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Установка Python-зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Копирование кода приложения
COPY . .

# 6. Документируем, что контейнер слушает 8080
EXPOSE 8080

# 7. Запускаем uvicorn на порту 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
