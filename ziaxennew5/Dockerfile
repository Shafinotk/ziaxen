FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
# COPY artifacts/ ./artifacts/      # or mount at runtime
# COPY .env ./.env                  # optional

ENV ARTIFACT_DIR=/app/artifacts
ENV DB_PATH=/app/db/events.db
ENV CSV_MIRROR=true
ENV CSV_PATH=/app/data/events.csv

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
