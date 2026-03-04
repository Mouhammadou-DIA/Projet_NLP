# Deployment Guide

This guide covers various deployment options for the Reddit RAG Chatbot.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Production Deployment](#production-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)

---

## Local Development

### Prerequisites

- Python 3.10+
- pip or uv package manager
- Git
- 4GB+ RAM
- (Optional) Ollama for local LLM

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/reddit-rag-chatbot.git
cd reddit-rag-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your settings
```

### Prepare Data

```bash
# Process raw data
python scripts/prepare_data.py

# Index into vector store (takes 10-15 minutes)
python scripts/index_conversations.py
```

### Run Services

```bash
# Terminal 1: Start API
python run_api.py

# Terminal 2: Start UI
python run_ui.py
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/api/v1/health/

# Test chat
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

---

## Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM (for building)

### Quick Start

```bash
# Build and run
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| api | 8000 | FastAPI REST API |
| ui | 7861 | Gradio Web Interface |
| nginx | 80, 443 | Reverse Proxy |
| redis | 6379 | Cache (optional) |
| prometheus | 9090 | Metrics (optional) |

### Build Images Separately

```bash
# Build API image
docker build -f docker/Dockerfile -t reddit-rag-api .

# Run API container
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  reddit-rag-api
```

### Docker Compose Configuration

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../data:/app/data
      - ../logs:/app/logs
    environment:
      - ENVIRONMENT=production
      - LOG_FORMAT=json
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Production Deployment

### Environment Configuration

```bash
# .env (production)
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
SECRET_KEY=<generate-strong-key>
ENABLE_AUTH=true
CORS_ORIGINS=https://yourdomain.com

# Performance
API_WORKERS=4
EMBEDDING_DEVICE=cuda  # if GPU available

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

### Generate Secret Key

```python
import secrets
print(secrets.token_urlsafe(32))
```

### SSL/TLS Configuration

Using Nginx as reverse proxy with Let's Encrypt:

```nginx
# /etc/nginx/sites-available/rag-chatbot
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # UI
    location / {
        proxy_pass http://localhost:7861;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Systemd Service

```ini
# /etc/systemd/system/rag-api.service
[Unit]
Description=Reddit RAG Chatbot API
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/reddit-rag-chatbot
Environment="PATH=/opt/reddit-rag-chatbot/venv/bin"
ExecStart=/opt/reddit-rag-chatbot/venv/bin/python run_api.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable rag-api
sudo systemctl start rag-api
sudo systemctl status rag-api
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.0+ (optional)

### Kubernetes Manifests

#### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-chatbot
```

#### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-chatbot
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  EMBEDDING_MODEL: "paraphrase-multilingual-MiniLM-L12-v2"
  LLM_PROVIDER: "ollama"
```

#### Secret

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-chatbot
type: Opaque
stringData:
  SECRET_KEY: "your-secret-key"
  OPENAI_API_KEY: "sk-..."
```

#### Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-chatbot
spec:
  replicas: 2
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
        - name: api
          image: your-registry/reddit-rag-api:latest
          ports:
            - containerPort: 8000
          envFrom:
            - configMapRef:
                name: rag-config
            - secretRef:
                name: rag-secrets
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
          livenessProbe:
            httpGet:
              path: /api/v1/health/live
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /api/v1/health/ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
          volumeMounts:
            - name: data
              mountPath: /app/data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: rag-data-pvc
```

#### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-chatbot
spec:
  selector:
    app: rag-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP
```

#### Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: rag-chatbot
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - rag.yourdomain.com
      secretName: rag-tls
  rules:
    - host: rag.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: rag-api-service
                port:
                  number: 80
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply configurations
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Check status
kubectl get pods -n rag-chatbot
kubectl get services -n rag-chatbot
```

---

## Cloud Deployment

### AWS (ECS/Fargate)

1. Push image to ECR
2. Create ECS cluster
3. Define task definition
4. Create service with ALB

### Google Cloud (Cloud Run)

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/reddit-rag-api

# Deploy
gcloud run deploy reddit-rag-api \
  --image gcr.io/PROJECT_ID/reddit-rag-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Azure (Container Apps)

```bash
# Create container app
az containerapp create \
  --name reddit-rag-api \
  --resource-group myResourceGroup \
  --image your-registry/reddit-rag-api:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 1 \
  --memory 2Gi
```

---

## Monitoring

### Prometheus Metrics

Enable with `METRICS_ENABLED=true`:

```bash
# Scrape endpoint
curl http://localhost:8000/metrics
```

### Grafana Dashboard

Import the provided dashboard from `docker/grafana/dashboards/`.

### Log Aggregation

Configure log shipping to your preferred platform:

```yaml
# For ELK Stack
LOG_FORMAT=json
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory**: Increase container memory limits
2. **Slow Startup**: Vector store loading takes time on first start
3. **Connection Refused**: Check service ports and firewall rules
4. **GPU Not Detected**: Verify CUDA installation and driver compatibility

### Health Checks

```bash
# Full health check
curl http://localhost:8000/api/v1/health/

# Component-specific
curl http://localhost:8000/api/v1/health/ready
curl http://localhost:8000/api/v1/health/live
```

### Logs

```bash
# Docker logs
docker-compose logs -f api

# Kubernetes logs
kubectl logs -f deployment/rag-api -n rag-chatbot

# Local logs
tail -f logs/app.log
```
