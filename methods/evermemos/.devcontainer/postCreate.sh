#!/bin/bash
set -e

echo "=== EverMemOS Dev Container Setup ==="

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y libgl1 libgomp1 libglib2.0-0 ffmpeg vim wget curl zip unzip g++ build-essential

# Install Python dependencies with uv
echo "Installing Python dependencies..."
cd /workspace
uv sync --dev

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp env.template .env

    # Update hostnames for Docker network
    sed -i 's/REDIS_HOST=localhost/REDIS_HOST=redis/' .env
    sed -i 's/MONGODB_HOST=localhost/MONGODB_HOST=mongodb/' .env
    sed -i 's|ES_HOSTS=http://localhost:19200|ES_HOSTS=http://elasticsearch:9200|' .env
    sed -i 's/MILVUS_HOST=localhost/MILVUS_HOST=milvus-standalone/' .env

    echo ""
    echo "NOTE: .env file created with Docker service hostnames."
    echo "Please update API keys (LLM_API_KEY, VECTORIZE_API_KEY, etc.) before running."
fi

# Install pre-commit hooks
echo "Setting up pre-commit hooks..."
uv run pre-commit install || true

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the application:"
echo "  make run"
echo ""
echo "Or manually:"
echo "  uv run python src/run.py"
echo ""
