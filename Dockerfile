FROM python:3.14-slim

WORKDIR /app

# Копируем только нужные файлы благодаря .dockerignore
COPY . /app

# Устанавливаем зависимости, если есть
RUN pip install --upgrade pip \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Команда по умолчанию
CMD ["python3"]
