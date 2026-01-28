#!/bin/bash
# Cloumask LLM Container Entrypoint
# Starts Ollama and ensures the required model is available

set -e

echo "Starting Cloumask AI service..."

# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for AI service to initialize..."
max_attempts=30
attempt=0
while ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    attempt=$((attempt + 1))
    if [ $attempt -ge $max_attempts ]; then
        echo "Error: AI service failed to start"
        exit 1
    fi
    sleep 1
done

echo "AI service is running"

# Check if model exists
model_exists=$(curl -s http://localhost:11434/api/tags | grep -c "$CLOUMASK_MODEL" || true)

if [ "$model_exists" -eq 0 ]; then
    echo "Downloading AI model: $CLOUMASK_MODEL (this may take several minutes)..."
    ollama pull "$CLOUMASK_MODEL"
    echo "Model download complete"
else
    echo "AI model already available: $CLOUMASK_MODEL"
fi

echo "Cloumask AI service ready"

# Keep the container running
wait $OLLAMA_PID
