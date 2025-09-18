FROM ubuntu:22.04

WORKDIR /app

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default (if links don't already exist)
RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi && \
    if [ ! -e /usr/bin/pip ]; then ln -s /usr/bin/pip3 /usr/bin/pip; fi

# Create a non-root user to run the application
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy dependency files first (better layer caching)
COPY pyproject.toml setup.py README.md ./

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    AGENT_CONFIG_PATH=/app/config/agents.yaml \
    LOG_LEVEL=INFO \
    SERVER_MODE=sync

# Create necessary directories
RUN mkdir -p /app/config /app/logs /app/prompts

# Copy the application code
COPY agent_builder/ /app/agent_builder/
COPY prompts/ /app/prompts/
COPY config/ /app/config/ 
COPY startup.sh /app/

# Make startup script executable
RUN chmod +x /app/startup.sh

# Install the package in development mode (after code is copied)
# Use --no-build-isolation to ensure we use the package directly
# Retry installation up to 3 times in case of network issues
RUN pip install --no-cache-dir --no-build-isolation -e . || \
    (sleep 2 && pip install --no-cache-dir --no-build-isolation -e .) || \
    (sleep 5 && pip install --no-cache-dir --no-build-isolation -e .)

# Ensure the default config file exists
RUN touch /app/config/agents.yaml

# Set proper permissions (after all files are copied)
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Add healthcheck (HTTP based for reliability)
HEALTHCHECK --interval=30s --timeout=5s --start-period=8s --retries=3 CMD curl -fsS http://localhost:5000/health || exit 1

# Command selector: if SERVER_MODE=async use async_app, else legacy startup script
ENTRYPOINT ["/bin/bash", "-lc", "if [ \"$SERVER_MODE\" = \"async\" ]; then echo 'Starting Async Quart server'; python agent_builder/async_app.py --config $AGENT_CONFIG_PATH --host 0.0.0.0 --port 5000; else echo 'Starting Sync Flask server'; /app/startup.sh; fi"]
