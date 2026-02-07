FROM python:3.11-slim
WORKDIR /app

COPY src/training/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
ENV PYTHONPATH=/app

RUN python -c "import pandas, fsspec, gcsfs; print('deps-ok')"

ENTRYPOINT ["python", "-m", "src.training.trainer"]
