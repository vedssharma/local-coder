# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt requirements-embedded.txt ./

# Install Python dependencies
# For CPU-only version (default)
RUN CMAKE_BUILD_PARALLEL_LEVEL=2 pip install --no-cache-dir -r requirements-embedded.txt

# For GPU support, uncomment the following and comment the line above:
# RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir -r requirements-embedded.txt

# Copy application files
COPY *.py ./
COPY skills ./skills
COPY entrypoint.sh .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Create a directory for models
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.gguf

# The model file should be mounted as a volume at /app/models
# Example: docker run -v /path/to/model.gguf:/app/models/model.gguf ...

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (interactive chat mode)
CMD ["chat"]
