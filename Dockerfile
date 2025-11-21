# Dockerfile — ASL Detector (Python 3.10)
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=5000
ENV TF_CPP_MIN_LOG_LEVEL=2

# Install system deps required for opencv/mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates git curl \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose port and start
EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4"]
