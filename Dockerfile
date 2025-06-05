FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m pip install --upgrade pip
RUN curl -sSL https://install.python-poetry.org | python3.10 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry
WORKDIR /app
COPY pyproject.toml poetry.lock* /app/
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi
COPY my_app/ /app/my_app/
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONPATH="${PYTHONPATH}:/app"
VOLUME /app/model_cache
VOLUME /app/qdrant_storage
EXPOSE 58084
CMD ["python3.10", "-m", "my_app.app"]
