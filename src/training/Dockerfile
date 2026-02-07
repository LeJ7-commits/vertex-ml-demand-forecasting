FROM python:3.11-slim
WORKDIR /app

COPY src /app/src

RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir \
      numpy pandas scikit-learn joblib \
      google-cloud-bigquery google-cloud-storage \
      db-dtypes pyarrow \
      fsspec gcsfs

ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "src.training.trainer"]
