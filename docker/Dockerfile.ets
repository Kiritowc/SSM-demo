FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY ets ./ets
COPY configs ./configs
COPY datasets ./datasets
COPY scripts ./scripts

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

VOLUME ["/app/runs", "/app/outputs", "/app/exports"]

CMD ["python", "scripts/train.py"]
