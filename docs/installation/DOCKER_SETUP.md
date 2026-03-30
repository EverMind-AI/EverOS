# Docker Setup Guide

[Home](../../README.md) > [Docs](../README.md) > [Installation](.) > Docker Setup

This guide provides detailed information about Docker setup and configuration for EverMemOS.

---

## Table of Contents

- [Overview](#overview)
- [Services Overview](#services-overview)
- [Quick Start](#quick-start)
- [Service Configuration](#service-configuration)
- [Port Mapping](#port-mapping)
- [Volume Management](#volume-management)
- [Service Management](#service-management)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## Overview

EverMemOS uses Docker Compose to manage four essential services:
- **MongoDB** - Primary database
- **Elasticsearch** - Keyword search engine
- **Milvus** - Vector database
- **Redis** - Cache service

All services are defined in `docker-compose.yml` and can be started with a single command.

---

## Services Overview

| Service | Host Port | Container Port | Purpose | Memory Usage |
|---------|-----------|----------------|---------|--------------|
| **MongoDB** | 27017 | 27017 | Primary database for storing memory cells and profiles | ~500MB |
| **Elasticsearch** | 19200 | 9200 | Keyword search engine (BM25) | ~2GB |
| **Milvus** | 19530 | 19530 | Vector database for semantic retrieval | ~1GB |
| **Redis** | 6379 | 6379 | Cache service | ~100MB |

**Total Memory Requirements**: Approximately 4GB minimum

---

## Quick Start

### Start All Services

```bash
docker-compose up -d
```

The `-d` flag runs containers in detached mode (background).

### Check Service Status

```bash
docker-compose ps
```

Expected output:
```
NAME                STATUS              PORTS
mongodb             Up 2 minutes        0.0.0.0:27017->27017/tcp
elasticsearch       Up 2 minutes        0.0.0.0:19200->9200/tcp
milvus-standalone   Up 2 minutes        0.0.0.0:19530->19530/tcp
redis               Up 2 minutes        0.0.0.0:6379->6379/tcp
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mongodb
docker-compose logs -f elasticsearch
```

### Stop Services

```bash
docker-compose down
```

To also remove volumes (⚠️ **deletes all data**):

```bash
docker-compose down -v
```

---

## Service Configuration

### MongoDB

**Configuration:**
- Version: 7.0
- Default credentials: `admin` / `memsys123`
- Port: 27017

**Connection String:**
```
mongodb://admin:memsys123@localhost:27017
```

**Collections:**
- `memcells` - Atomic memory units
- `episodes` - Episodic memories
- `profiles` - User profiles
- `preferences` - User preferences
- And more...

See [MongoDB Guide](../usage/MONGODB_GUIDE.md) for detailed database information.

### Elasticsearch

**Configuration:**
- Version: 8.x
- Security: Disabled (for local development)
- Port: 19200 (mapped from container port 9200)

**Connection URL:**
```
http://localhost:19200
```

**Test Connection:**
```bash
curl http://localhost:19200
```

**Indices:**
- Memory cells indexed for keyword search
- BM25 algorithm for relevance ranking

### Milvus

**Configuration:**
- Version: 2.4+
- Standalone deployment
- Port: 19530

**Connection:**
```python
from pymilvus import connections
connections.connect(host="localhost", port="19530")
```

**Collections:**
- Vector embeddings for semantic search
- Multiple index types supported

### Redis

**Configuration:**
- Version: 7.x
- Port: 6379
- No password (for local development)

**Connection:**
```bash
redis-cli -h localhost -p 6379 ping
```

**Usage:**
- Caching frequently accessed data
- Session management
- Temporary data storage

---

## Port Mapping

### Understanding Port Mapping

Format: `HOST_PORT:CONTAINER_PORT`

When connecting from your local machine, always use the **HOST PORT**.

### Default Port Mapping

| Service | Host Port | Container Port | URL from Host |
|---------|-----------|----------------|---------------|
| MongoDB | 27017 | 27017 | `localhost:27017` |
| Elasticsearch | 19200 | 9200 | `http://localhost:19200` |
| Milvus | 19530 | 19530 | `localhost:19530` |
| Redis | 6379 | 6379 | `localhost:6379` |

### Custom Port Configuration

To use different host ports, edit `docker-compose.yml`:

```yaml
services:
  mongodb:
    ports:
      - "27018:27017"  # Use port 27018 on host

  elasticsearch:
    ports:
      - "9200:9200"  # Use standard port 9200 on host
```

After changing ports, update your `.env` file accordingly:

```bash
MONGODB_URI=mongodb://admin:memsys123@localhost:27018
ELASTICSEARCH_URL=http://localhost:19200
```

---

## Volume Management

### Data Persistence

All services use Docker volumes to persist data:

```bash
# List volumes
docker volume ls

# Inspect a volume
docker volume inspect evermemos_mongodb_data
docker volume inspect evermemos_elasticsearch_data
docker volume inspect evermemos_milvus_data
docker volume inspect evermemos_redis_data
```

### Backup Data

**MongoDB:**
```bash
# Backup
docker exec mongodb mongodump --username admin --password memsys123 --authenticationDatabase admin --out /tmp/backup

# Copy backup out of container
docker cp mongodb:/tmp/backup ./mongodb_backup
```

**Elasticsearch:**
```bash
# Take snapshot (requires snapshot repository setup)
curl -X PUT "http://localhost:19200/_snapshot/my_backup/snapshot_1?wait_for_completion=true"
```

### Restore Data

**MongoDB:**
```bash
# Copy backup into container
docker cp ./mongodb_backup mongodb:/tmp/backup

# Restore
docker exec mongodb mongorestore --username admin --password memsys123 --authenticationDatabase admin /tmp/backup
```

### Clear All Data

⚠️ **Warning**: This will delete all memory data!

```bash
docker-compose down -v
docker-compose up -d
```

---

## Service Management

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart mongodb
```

### Update Services

```bash
# Pull latest images
docker-compose pull

# Recreate containers with new images
docker-compose up -d
```

### Scale Services (Advanced)

Some services can be scaled:

```bash
# Not typically needed for local development
docker-compose up -d --scale redis=2
```

### Resource Limits

Edit `docker-compose.yml` to set memory/CPU limits:

```yaml
services:
  elasticsearch:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'
```

---

## Troubleshooting

### Services Won't Start

**Check Docker Status:**
```bash
docker info
```

**Check Logs:**
```bash
docker-compose logs
```

**Common Issues:**
- Port conflicts: Another service using the same port
- Insufficient memory: Docker doesn't have enough RAM allocated
- Permission issues: Docker doesn't have permission to create volumes

### Port Conflicts

**Find what's using a port:**
```bash
# macOS/Linux
lsof -i :27017
lsof -i :19200

# Or use netstat
netstat -an | grep 27017
```

**Solutions:**
- Stop conflicting service
- Change host port in docker-compose.yml

### Out of Memory

**Symptoms:**
- Elasticsearch crashes
- Milvus becomes unresponsive
- Services restart repeatedly

**Solutions:**

1. **Increase Docker Memory** (Docker Desktop):
   - Docker Desktop > Preferences > Resources
   - Increase memory to 8GB or more

2. **Reduce Elasticsearch Heap:**
   Edit docker-compose.yml:
   ```yaml
   elasticsearch:
     environment:
       - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
   ```

3. **Close other applications** to free up memory

### Connection Refused

**Check service is running:**
```bash
docker-compose ps
```

**Verify port mapping:**
```bash
docker-compose port mongodb 27017
```

**Test connectivity:**
```bash
# MongoDB
telnet localhost 27017

# Elasticsearch
curl http://localhost:19200

# Redis
redis-cli -h localhost -p 6379 ping
```

### Elasticsearch Yellow/Red Status

**Check cluster health:**
```bash
curl http://localhost:19200/_cluster/health?pretty
```

**Common cause**: Single-node cluster (expected for local development)

**Set to single-node mode:**
```yaml
elasticsearch:
  environment:
    - discovery.type=single-node
```

---

## Advanced Configuration

### Custom docker-compose.yml

Create `docker-compose.override.yml` for custom configuration:

```yaml
version: '3.8'
services:
  mongodb:
    environment:
      - MONGO_INITDB_ROOT_USERNAME=myuser
      - MONGO_INITDB_ROOT_PASSWORD=mypassword
```

This file is automatically merged with `docker-compose.yml`.

### Network Configuration

Services communicate on a shared network:

```yaml
networks:
  evermemos-network:
    driver: bridge
```

### Environment Variables

Pass environment variables to services:

```yaml
services:
  mongodb:
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGO_USER:-admin}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGO_PASSWORD:-memsys123}
```

Then create `.env.docker`:
```bash
MONGO_USER=admin
MONGO_PASSWORD=secretpassword
```

### Health Checks

Add health checks to ensure services are ready:

```yaml
services:
  mongodb:
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh localhost:27017/test --quiet
      interval: 10s
      timeout: 5s
      retries: 5
```

---

## Production Considerations

⚠️ **Note**: The default docker-compose.yml is for **local development only**.

For production deployment:

1. **Change default passwords**
2. **Enable authentication** for all services
3. **Use external volumes** for backups
4. **Configure SSL/TLS**
5. **Set resource limits**
6. **Use managed services** (AWS, Azure, GCP) when possible
7. **Implement monitoring** and alerting

---

## See Also

- [Setup Guide](SETUP.md) - Complete installation guide
- [Configuration Guide](../usage/CONFIGURATION_GUIDE.md) - Environment variable configuration
- [MongoDB Guide](../usage/MONGODB_GUIDE.md) - MongoDB-specific documentation
- [Troubleshooting](SETUP.md#troubleshooting) - Common installation issues
