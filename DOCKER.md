# Docker Usage Guide

This guide explains how to run Local Coder using Docker.

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t local-coder .
```

### 2. Run with Docker

**Interactive Chat Mode:**
```bash
docker run -it --rm \
  -v /path/to/your/model.gguf:/app/models/model.gguf:ro \
  local-coder
```

**Ask a Question:**
```bash
docker run -it --rm \
  -v /path/to/your/model.gguf:/app/models/model.gguf:ro \
  local-coder ask "How do I read a file in Python?"
```

**With File References:**
```bash
docker run -it --rm \
  -v /path/to/your/model.gguf:/app/models/model.gguf:ro \
  -v $(pwd):/workspace:ro \
  -w /workspace \
  local-coder ask "Explain what @main.py does"
```

## Using Docker Compose

### 1. Update docker-compose.yml

Edit `docker-compose.yml` and update the model path to point to your model file:

```yaml
volumes:
  - ./your-model-name.gguf:/app/models/model.gguf:ro
```

### 2. Run with Docker Compose

**Start interactive chat:**
```bash
docker-compose run --rm local-coder
```

**Ask a question:**
```bash
docker-compose run --rm local-coder ask "your question here"
```

**Edit files:**
```bash
docker-compose run --rm local-coder edit "your edit request" --dry-run
```

## GPU Support

### NVIDIA GPU (CUDA)

**1. Build with GPU support:**

Edit the `Dockerfile` and uncomment the GPU installation lines:

```dockerfile
# Uncomment this:
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python --force-reinstall && \
    pip install --no-cache-dir typer rich

# Comment out the CPU version:
# RUN pip install --no-cache-dir -r requirements.txt
```

**2. Build the image:**
```bash
docker build -t local-coder:gpu .
```

**3. Run with GPU:**
```bash
docker run -it --rm --gpus all \
  -v /path/to/your/model.gguf:/app/models/model.gguf:ro \
  local-coder:gpu
```

**4. Or use Docker Compose:**

Uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then run:
```bash
docker-compose run --rm local-coder
```

### Apple Silicon (Metal)

Docker on Mac doesn't support Metal GPU acceleration. It's recommended to run the application natively on macOS for GPU support.

## Environment Variables

- `MODEL_PATH`: Path to the model file inside the container (default: `/app/models/model.gguf`)

Example:
```bash
docker run -it --rm \
  -e MODEL_PATH=/app/models/custom.gguf \
  -v /path/to/your/model.gguf:/app/models/custom.gguf:ro \
  local-coder
```

## Volume Mounts

### Model File (Required)
Mount your GGUF model file to `/app/models/model.gguf`:

```bash
-v /path/to/your/model.gguf:/app/models/model.gguf:ro
```

### Workspace Directory (Optional)
Mount your code directory to use `@filename` references:

```bash
-v $(pwd):/workspace:ro -w /workspace
```

The `:ro` flag mounts the volume as read-only for safety.

## Common Use Cases

### 1. Development Assistant

```bash
# Chat about your code
docker run -it --rm \
  -v ./model.gguf:/app/models/model.gguf:ro \
  -v $(pwd):/workspace:ro \
  -w /workspace \
  local-coder
```

Then use `@filename` to reference files in your prompts.

### 2. One-off Questions

```bash
# Quick answers without entering interactive mode
docker run -it --rm \
  -v ./model.gguf:/app/models/model.gguf:ro \
  local-coder ask "What is a Python decorator?"
```

### 3. Code Analysis

```bash
# Analyze specific files
docker run -it --rm \
  -v ./model.gguf:/app/models/model.gguf:ro \
  -v $(pwd):/workspace:ro \
  -w /workspace \
  local-coder ask "Review @main.py for potential bugs"
```

## Troubleshooting

### Model Not Found

**Error:** `Model file not found at /app/models/model.gguf`

**Solution:** Ensure you're mounting the model file correctly:
```bash
docker run -it --rm \
  -v /absolute/path/to/model.gguf:/app/models/model.gguf:ro \
  local-coder
```

### Permission Denied

**Error:** Permission issues when mounting files

**Solution:** Ensure the files have read permissions:
```bash
chmod +r /path/to/model.gguf
```

### Container Exits Immediately

**Solution:** Make sure to use the `-it` flags for interactive mode:
```bash
docker run -it --rm local-coder
```

### Out of Memory

**Solution:** Limit container memory or use a smaller model:
```bash
docker run -it --rm --memory=8g \
  -v ./model.gguf:/app/models/model.gguf:ro \
  local-coder
```

## Building for Different Platforms

### Multi-platform Build

```bash
# Build for both AMD64 and ARM64
docker buildx build --platform linux/amd64,linux/arm64 -t local-coder .
```

### CPU-only Lightweight Image

The default Dockerfile builds a CPU-only image. This is smaller and works on any platform but is slower.

### GPU-optimized Image

Follow the GPU Support section above to build an image with CUDA support for faster inference.

## Image Size Optimization

The current image excludes the model file to keep it lightweight. The model should always be mounted as a volume rather than included in the image.

**Image size breakdown:**
- Base Python image: ~150MB
- Dependencies: ~200MB
- Application code: <1MB
- **Total: ~350MB (without model)**

Models are typically 4-8GB and should be stored separately.
