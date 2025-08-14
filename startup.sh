#!/bin/bash
# startup.sh - Handles application startup with proper initialization and error handling

# Ensure configuration directory exists
mkdir -p /app/config /app/logs

# Check if agents.yaml exists, create a basic one if not
if [ ! -s /app/config/agents.yaml ]; then
    echo "Warning: No agents.yaml found or file is empty. Creating minimal configuration."
    cat > /app/config/agents.yaml << EOF
# Default minimal configuration
agent:
  type: react
  llm:
    type: openai
    model: gpt-4-turbo
EOF
fi

# Start the application
echo "Starting MAAP Agent Builder..."
exec python -m agent_builder.cli serve --config /app/config/agents.yaml --host 0.0.0.0 --port 5000
