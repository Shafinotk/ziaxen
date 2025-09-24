# ---- Base image ----
FROM python:3.12-slim

# ---- Set workdir ----
WORKDIR /app

# ---- Install system dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy requirements ----
COPY requirements.txt .

# ---- Install Python deps ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy project ----
COPY . .

# ---- Environment vars ----
ENV PYTHONUNBUFFERED=1 \
    ARTIFACT_DIR=/app/artifacts \
    DB_PATH=/app/db/events.db \
    CSV_PATH=/app/data/events.csv \
    EGO_LOW=0.5 \
    EGO_HIGH=0.9 \
    LR_LOW=0.5 \
    LR_HIGH=0.9 \
    COMBINED_POLICY=max_severity

# ---- Expose port ----
EXPOSE 8000

# ---- Run server ----
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
