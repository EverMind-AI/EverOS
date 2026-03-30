#!/bin/bash

echo "=== Waiting for infrastructure services ==="

# Function to wait for a service using timeout and bash tcp check
wait_for_service() {
    local name=$1
    local host=$2
    local port=$3
    local max_attempts=${4:-30}
    local attempt=1

    echo -n "Waiting for $name ($host:$port)..."
    while ! timeout 1 bash -c "echo > /dev/tcp/$host/$port" 2>/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            echo " timeout (service may still be starting)"
            return 1
        fi
        sleep 2
        attempt=$((attempt + 1))
    done
    echo " ready"
}

# Wait for core services
wait_for_service "MongoDB" "mongodb" 27017
wait_for_service "Redis" "redis" 6379
wait_for_service "Elasticsearch" "elasticsearch" 9200 60
wait_for_service "Milvus" "milvus-standalone" 19530 90

echo ""
echo "=== Infrastructure Ready ==="
echo ""
echo "Available services:"
echo "  - MongoDB:       mongodb:27017"
echo "  - Redis:         redis:6379"
echo "  - Elasticsearch: elasticsearch:9200"
echo "  - Milvus:        milvus-standalone:19530"
echo "  - MinIO Console: milvus-minio:9001"
echo ""
echo "Run 'make run' to start EverMemOS"
echo ""
