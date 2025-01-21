# Base image
FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc git && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY configs configs/
COPY tasks.py tasks.py

RUN mkdir models
RUN mkdir data
RUN mkdir data/raw
RUN mkdir data/processed
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

RUN pip install fastapi uvicorn

# setup the data
ENV DATA_BUCKET="catdog-data"
ENV CATS_PATH="data/raw/cats"
ENV DOGS_PATH="data/raw/dogs"
ENV MODEL_BUCKET="catdog-models"
ENV MODEL_PATH="models"

RUN python3 src/catdogdetection/download_bucket.py $DATA_BUCKET $CATS_PATH $CATS_PATH
RUN python3 src/catdogdetection/download_bucket.py $DATA_BUCKET $DOGS_PATH $DOGS_PATH
RUN python3 src/catdogdetection/download_bucket.py $MODEL_BUCKET $MODEL_PATH $MODEL_PATH
RUN python3 src/catdogdetection/data.py 1000

# Expose port 8080
ENV PORT=8080

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn src.catdogdetection.api:app --host 0.0.0.0 --port $PORT"]
