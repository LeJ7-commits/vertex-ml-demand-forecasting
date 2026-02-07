FROM python:3.11-slim

WORKDIR /app

# Copy requirements from its actual location (relative to build context = repo root)
COPY src/training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY src /app/src
ENV PYTHONPATH=/app

# fail-fast sanity check
RUN python -c "import pandas, fsspec, gcsfs; print('deps-ok')"

ENTRYPOINT ["python", "-m", "src.training.trainer"]
