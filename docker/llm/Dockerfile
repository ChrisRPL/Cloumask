# Cloumask LLM Container
# Bundles Ollama with the required model for a self-contained AI environment
FROM ollama/ollama:latest

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Default model - can be overridden via environment variable
ENV CLOUMASK_MODEL=qwen3:14b

# Expose Ollama API port
EXPOSE 11434

# Health check - verify Ollama is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

ENTRYPOINT ["/entrypoint.sh"]
