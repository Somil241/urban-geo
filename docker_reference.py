"""
Docker setup for Urban-Geo (optional).
"""
# docker-compose.yml would be created separately
# This is a reference for Docker setup

DOCKERFILE_CONTENT = """
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python", "run_server.py"]
"""

# Note: For production, use docker-compose.yml with:
# - PostgreSQL with PostGIS
# - Redis (for Celery)
# - API service
# - Celery worker
# - Celery beat scheduler

