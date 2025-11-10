FROM python:3.11-slim

# Install torch (CPU) + transformers
RUN apt-get update && apt-get install -y --no-install-recommends git curl && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir diffusers && \
    pip install --no-cache-dir transformers && \
    pip install --no-cache-dir accelerate && \
    pip install --no-cache-dir safetensors && \
    pip install --no-cache-dir pillow && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV OUTPUT_DIR=/outputs
COPY hello_image_gen.py .

CMD ["python", "hello_image_gen.py"]
