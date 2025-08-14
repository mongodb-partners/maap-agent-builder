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
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AGENT_CONFIG_PATH=/app/config/agents.yaml
ENV LOG_LEVEL=INFO

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

# Add healthcheck to verify python process is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD ps aux | grep "[p]ython -m agent_builder.cli" || exit 1

# Command to run the application with proper error handling
CMD ["/app/startup.sh"]
