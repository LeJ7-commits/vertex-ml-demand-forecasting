FROM python:3.11-slim
WORKDIR /app

# copy full src tree so "src.training..." exists
COPY src /app/src

# install deps your trainer uses
RUN pip install --no-cache-dir \
  numpy pandas scikit-learn joblib \
  google-cloud-bigquery db-dtypes pyarrow

# make /app importable so "import src...." works
ENV PYTHONPATH=/app

# run as module (best practice)
ENTRYPOINT ["python", "-m", "src.training.trainer"]
