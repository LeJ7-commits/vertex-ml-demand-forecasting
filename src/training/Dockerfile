FROM python:3.11-slim

WORKDIR /app

# Copy source
COPY src /app/src

# Install runtime deps (include fsspec + gcsfs for gs:// IO)
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    pyarrow \
    db-dtypes \
    google-cloud-bigquery \
    google-cloud-storage \
    fsspec \
    gcsfs

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "-m", "src.training.trainer"]
