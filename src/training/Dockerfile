FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (for build cache efficiency)
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src /app/src

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.training.trainer"]
