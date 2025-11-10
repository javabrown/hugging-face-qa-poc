FROM python:3.11-slim

# Install torch (CPU) + transformers
RUN apt-get update && apt-get install -y --no-install-recommends git curl && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers && \
    rm -rf /var/lib/apt/lists/*

# Copy your hello script
WORKDIR /app
COPY hello_hf_pipeline.py .

# Run the script when container starts
CMD ["python", "hello_hf_pipeline.py"]