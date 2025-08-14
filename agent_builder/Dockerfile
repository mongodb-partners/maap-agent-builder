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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default (if links don't already exist)
RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi && \
    if [ ! -e /usr/bin/pip ]; then ln -s /usr/bin/pip3 /usr/bin/pip; fi

# Create a non-root user to run the application
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy requirements and install dependencies
COPY requirements.txt .
COPY setup.py .
COPY __init__.py .

# Install the package in development mode
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install -e .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AGENT_CONFIG_PATH=/app/config/agents.yaml
ENV LOG_LEVEL=INFO

# Create directories for configuration and logs and set proper permissions
RUN mkdir -p /app/config /app/logs && \
    chown -R appuser:appuser /app

# Copy the rest of the application
COPY . .
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--pythonpath", "/app", "agent_builder.wsgi:application"]
