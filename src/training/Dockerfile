FROM python:3.11-slim

WORKDIR /app

# Copy only training code
COPY . /app

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    joblib \
    google-cloud-bigquery \
    db-dtypes \
    pyarrow

ENTRYPOINT ["python", "trainer.py"]
