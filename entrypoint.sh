#!/bin/bash
# Entrypoint script for Docker container

# Default model path
MODEL_PATH="${MODEL_PATH:-/app/models/model.gguf}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please mount your model file using -v flag:"
    echo "  docker run -v /path/to/your/model.gguf:/app/models/model.gguf local-coder"
    exit 1
fi

# Update the model path in main.py dynamically
sed -i "s|model_path=\".*\"|model_path=\"$MODEL_PATH\"|g" /app/main.py

# Execute the command
exec python /app/main.py "$@"
