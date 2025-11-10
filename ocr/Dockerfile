# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Tesseract + tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir pillow==10.4.0 pytesseract==0.3.10

WORKDIR /app
COPY hello_ocr.py .

# Default I/O
ENV INPUT_IMAGE=/inputs/sample.jpg OUTPUT_FILE=/outputs/ocr.txt
CMD ["python", "hello_ocr.py"]
