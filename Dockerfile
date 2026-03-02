FROM python:3.14-slim

# Системные зависимости для numpy/scipy/pybaselines/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем pyproject.toml + uv.lock (если используешь uv sync)
COPY pyproject.toml uv.lock* ./

# uv — очень быстрый, но если хочешь классику — pip
# Вариант 1: uv (рекомендую, если uv уже в проекте)
RUN pip install uv && \
    uv sync --frozen --no-install-project --no-dev && \
    uv pip install --system .

# Вариант 2: классический pip (если uv не хочешь)
# RUN pip install --no-cache-dir .[standard-no-fastapi-cloud-cli]

COPY . .

# Важно: static должна быть в правильном месте
EXPOSE 8001

# Запуск без reload
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
