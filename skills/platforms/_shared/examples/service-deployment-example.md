# Service Deployment Example
Comprehensive guide to deploying applications using various deployment strategies

## Overview

This example demonstrates deploying applications to Flow Nexus Platform using:
- Direct deployment
- Blue-green deployment
- Canary deployment
- Rolling deployment

**Estimated Time**: 45-60 minutes
**Difficulty**: Advanced
**Prerequisites**: Completed platform setup, running services

---

## Deployment Strategy 1: Direct Deployment

### Overview
Simple, immediate deployment with automatic backup and rollback capabilities.

**Best For**: Development environments, low-traffic applications
**Downtime**: 30-60 seconds
**Rollback Time**: <10 seconds

### Step 1: Prepare Application

```bash
# Create application directory
mkdir -p deployments/my-app-v1.0.0

# Copy application files
cp -r src/* deployments/my-app-v1.0.0/

# Install dependencies
cd deployments/my-app-v1.0.0
npm install --production
cd ../..
```

### Step 2: Configure Deployment

```yaml
# deployment-config.yaml
strategy: direct
app_name: my-app
version: 1.0.0
source_path: deployments/my-app-v1.0.0
target_path: /var/www/my-app

configuration:
  backup_before_deploy: true
  auto_rollback_on_failure: true

health_check:
  url: http://localhost:3000/health
  timeout: 10
  retries: 3

pre_deploy_script: |
  npm run build
  npm run test

post_deploy_script: |
  npm run migrate
  npm run seed
```

### Step 3: Execute Deployment

```bash
# Run direct deployment
python3 resources/deployment-manager.py deploy \
  my-app \
  1.0.0 \
  deployments/my-app-v1.0.0 \
  /var/www/my-app

# Monitor deployment
python3 resources/deployment-manager.py status my-app
```

**Expected Output:**
```json
{
  "id": "deploy-abc123",
  "app_name": "my-app",
  "version": "1.0.0",
  "status": "active",
  "deployed_at": "2025-11-02T10:30:00Z",
  "previous_version": "0.9.5",
  "checksum": "sha256:abc123..."
}
```

### Step 4: Verify Deployment

```bash
# Check application health
curl http://localhost:3000/health

# Verify version
curl http://localhost:3000/api/version

# Check logs
tail -f /var/log/my-app/app.log
```

### Step 5: Rollback (if needed)

```bash
# Rollback to previous version
python3 resources/deployment-manager.py rollback my-app

# Verify rollback
curl http://localhost:3000/api/version
```

---

## Deployment Strategy 2: Blue-Green Deployment

### Overview
Zero-downtime deployment with instant rollback capability.

**Best For**: Production environments, critical applications
**Downtime**: 0 seconds (seamless switch)
**Rollback Time**: <5 seconds

### Architecture

```
┌─────────────────────────────────────────┐
│         Load Balancer                    │
│         (Traffic Router)                 │
└─────────────┬───────────────────────────┘
              │
        ┌─────┴─────┐
        │           │
   ┌────▼────┐ ┌───▼─────┐
   │  Blue   │ │  Green  │
   │  v1.0.0 │ │  v1.1.0 │
   │(Current)│ │  (New)  │
   └─────────┘ └─────────┘
```

### Step 1: Prepare Green Environment

```bash
# Create green deployment directory
mkdir -p deployments/my-app-green

# Build application
cd deployments/my-app-v1.1.0
npm run build
cd ../..

# Copy to green environment
cp -r deployments/my-app-v1.1.0/* deployments/my-app-green/
```

### Step 2: Deploy to Green

```javascript
// deploy-blue-green.js
const { DeploymentManager } = require('./resources/deployment-manager');

async function deployBlueGreen() {
  const manager = new DeploymentManager();

  // Create deployment
  const deployment = {
    id: `my-app-green-${Date.now()}`,
    app_name: 'my-app',
    version: '1.1.0',
    source_path: 'deployments/my-app-green',
    target_path: '/var/www/my-app-green',
    strategy: 'blue_green',
    health_check_url: 'http://localhost:3001/health'
  };

  // Deploy to green environment (port 3001)
  console.log('Deploying to green environment...');
  const success = await manager.deploy(deployment);

  if (!success) {
    throw new Error('Green deployment failed');
  }

  console.log('Green environment deployed successfully');
  return deployment;
}

deployBlueGreen()
  .then(deployment => {
    console.log('Deployment successful:', deployment.id);
  })
  .catch(err => {
    console.error('Deployment failed:', err);
    process.exit(1);
  });
```

### Step 3: Validate Green Environment

```bash
# Run smoke tests on green
curl http://localhost:3001/health
curl http://localhost:3001/api/version

# Run load tests
npx artillery quick --count 100 --num 10 http://localhost:3001

# Monitor metrics for 5 minutes
node resources/health-monitor.js --port 3001 --duration 300
```

### Step 4: Switch Traffic to Green

```nginx
# Update Nginx configuration
# /etc/nginx/sites-available/my-app

upstream app_backend {
    # Switch from blue (3000) to green (3001)
    server localhost:3001;
}

server {
    listen 80;
    server_name myapp.example.com;

    location / {
        proxy_pass http://app_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
# Reload Nginx
sudo nginx -t && sudo nginx -s reload

# Verify traffic switch
curl -H "Host: myapp.example.com" http://localhost/api/version
```

### Step 5: Monitor and Cleanup

```bash
# Monitor green for 10 minutes
watch -n 10 'curl -s http://localhost/health | jq .'

# If successful, stop blue environment
python3 resources/service-orchestrator.py stop my-app-blue

# Clean up old deployment
rm -rf /var/www/my-app-blue
```

### Rollback Procedure

```bash
# Switch back to blue (instant)
# Update Nginx upstream to port 3000
sudo sed -i 's/localhost:3001/localhost:3000/' /etc/nginx/sites-available/my-app
sudo nginx -s reload

# Stop green environment
python3 resources/service-orchestrator.py stop my-app-green
```

---

## Deployment Strategy 3: Canary Deployment

### Overview
Gradual traffic shift with automated rollback on anomaly detection.

**Best For**: Large user bases, risk-averse deployments
**Downtime**: 0 seconds
**Rollback Time**: <30 seconds
**Deployment Duration**: 30-60 minutes

### Traffic Shifting Timeline

```
Time    | Blue Traffic | Canary Traffic | Action
--------|--------------|----------------|------------------
T+0     | 100%         | 0%             | Deploy canary
T+5     | 95%          | 5%             | Monitor metrics
T+10    | 90%          | 10%            | Monitor metrics
T+20    | 75%          | 25%            | Extended monitor
T+35    | 50%          | 50%            | Load test
T+50    | 0%           | 100%           | Full switch
T+60    | -            | 100%           | Cleanup old
```

### Step 1: Deploy Canary Version

```javascript
// deploy-canary.js
const { DeploymentManager } = require('./resources/deployment-manager');

async function deployCanary() {
  const manager = new DeploymentManager();

  const deployment = {
    id: `my-app-canary-${Date.now()}`,
    app_name: 'my-app',
    version: '1.2.0',
    source_path: 'deployments/my-app-v1.2.0',
    target_path: '/var/www/my-app-canary',
    strategy: 'canary',
    health_check_url: 'http://localhost:3002/health'
  };

  console.log('Deploying canary version...');
  const success = await manager.deploy(deployment);

  if (!success) {
    throw new Error('Canary deployment failed');
  }

  console.log('Canary deployed, starting gradual traffic shift...');
  return deployment;
}

deployCanary().catch(console.error);
```

### Step 2: Configure Traffic Splitting

```nginx
# /etc/nginx/sites-available/my-app

# Split traffic using weighted round-robin
upstream app_backend {
    server localhost:3000 weight=95;  # Stable version
    server localhost:3002 weight=5;   # Canary version
}

server {
    listen 80;
    server_name myapp.example.com;

    location / {
        proxy_pass http://app_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Add canary header for tracking
        add_header X-Canary-Version $upstream_addr always;
    }
}
```

### Step 3: Automated Traffic Shifting Script

```bash
#!/bin/bash
# shift-canary-traffic.sh

WEIGHTS=(
    "95:5"
    "90:10"
    "75:25"
    "50:50"
    "0:100"
)

DURATIONS=(300 300 600 900 1800)  # seconds

for i in "${!WEIGHTS[@]}"; do
    IFS=':' read -r stable_weight canary_weight <<< "${WEIGHTS[$i]}"
    duration="${DURATIONS[$i]}"

    echo "Shifting traffic: Stable=$stable_weight%, Canary=$canary_weight%"

    # Update Nginx config
    sed -i "s/weight=[0-9]*/weight=$stable_weight/g" /etc/nginx/sites-available/my-app
    sed -i "s/weight=[0-9]*/weight=$canary_weight/2" /etc/nginx/sites-available/my-app

    # Reload Nginx
    sudo nginx -s reload

    # Monitor for duration
    echo "Monitoring for ${duration}s..."

    # Check metrics
    python3 resources/canary-monitor.py \
        --stable-url http://localhost:3000 \
        --canary-url http://localhost:3002 \
        --duration $duration \
        --error-threshold 0.01 \
        --latency-threshold 1.5

    if [ $? -ne 0 ]; then
        echo "Anomaly detected! Rolling back..."
        # Rollback to 100% stable
        sed -i 's/weight=[0-9]*/weight=100/' /etc/nginx/sites-available/my-app
        sed -i 's/weight=[0-9]*/weight=0/2' /etc/nginx/sites-available/my-app
        sudo nginx -s reload
        exit 1
    fi
done

echo "Canary deployment successful! All traffic on new version."
```

### Step 4: Monitor Canary Metrics

```python
# canary-monitor.py
import sys
import time
import requests
from dataclasses import dataclass

@dataclass
class Metrics:
    error_rate: float
    latency_p99: float
    throughput: float

def collect_metrics(url: str) -> Metrics:
    """Collect metrics from endpoint"""
    resp = requests.get(f"{url}/metrics")
    data = resp.json()

    return Metrics(
        error_rate=data['error_rate'],
        latency_p99=data['latency_p99'],
        throughput=data['throughput']
    )

def compare_metrics(stable: Metrics, canary: Metrics,
                   error_threshold: float,
                   latency_threshold: float) -> bool:
    """Compare canary metrics against stable"""

    # Check error rate
    if canary.error_rate > stable.error_rate * (1 + error_threshold):
        print(f"❌ Error rate too high: {canary.error_rate:.4f} vs {stable.error_rate:.4f}")
        return False

    # Check latency
    if canary.latency_p99 > stable.latency_p99 * latency_threshold:
        print(f"❌ Latency too high: {canary.latency_p99:.2f}ms vs {stable.latency_p99:.2f}ms")
        return False

    # Check throughput
    if canary.throughput < stable.throughput * 0.9:
        print(f"❌ Throughput too low: {canary.throughput:.2f} vs {stable.throughput:.2f}")
        return False

    print(f"✅ Metrics healthy - Error: {canary.error_rate:.4f}, Latency: {canary.latency_p99:.2f}ms")
    return True

def monitor_canary(stable_url: str, canary_url: str,
                  duration: int, error_threshold: float,
                  latency_threshold: float) -> bool:
    """Monitor canary for specified duration"""

    start_time = time.time()
    check_interval = 10  # seconds

    while time.time() - start_time < duration:
        stable_metrics = collect_metrics(stable_url)
        canary_metrics = collect_metrics(canary_url)

        if not compare_metrics(stable_metrics, canary_metrics,
                             error_threshold, latency_threshold):
            return False

        time.sleep(check_interval)

    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--stable-url', required=True)
    parser.add_argument('--canary-url', required=True)
    parser.add_argument('--duration', type=int, required=True)
    parser.add_argument('--error-threshold', type=float, default=0.01)
    parser.add_argument('--latency-threshold', type=float, default=1.5)

    args = parser.parse_args()

    success = monitor_canary(
        args.stable_url,
        args.canary_url,
        args.duration,
        args.error_threshold,
        args.latency_threshold
    )

    sys.exit(0 if success else 1)
```

---

## Deployment Strategy 4: Rolling Deployment

### Overview
Instance-by-instance update minimizing risk.

**Best For**: Distributed systems, microservices
**Downtime**: 0 seconds
**Rollback Time**: Variable (depends on batch size)

### Step 1: Configure Rolling Update

```yaml
# rolling-deployment.yaml
strategy: rolling
app_name: my-app
version: 1.3.0

configuration:
  batch_size: 2  # Update 2 instances at a time
  batch_interval: 60  # Wait 60s between batches
  health_check_grace_period: 30
  min_healthy_instances: 75%  # Keep 75% healthy
  max_unavailable: 25%

instances:
  - id: instance-1
    host: server1.example.com
    port: 3000
  - id: instance-2
    host: server2.example.com
    port: 3000
  - id: instance-3
    host: server3.example.com
    port: 3000
  - id: instance-4
    host: server4.example.com
    port: 3000
```

### Step 2: Execute Rolling Deployment

```bash
#!/bin/bash
# rolling-deploy.sh

INSTANCES=("instance-1" "instance-2" "instance-3" "instance-4")
BATCH_SIZE=2
VERSION="1.3.0"

for ((i=0; i<${#INSTANCES[@]}; i+=BATCH_SIZE)); do
    batch=("${INSTANCES[@]:i:BATCH_SIZE}")

    echo "Updating batch: ${batch[*]}"

    for instance in "${batch[@]}"; do
        echo "  Updating $instance..."

        # Drain connections
        curl -X POST "http://${instance}:3000/admin/drain"
        sleep 10

        # Stop instance
        ssh "$instance" "sudo systemctl stop my-app"

        # Deploy new version
        scp -r "deployments/my-app-v${VERSION}/*" "$instance:/var/www/my-app/"

        # Start instance
        ssh "$instance" "sudo systemctl start my-app"

        # Wait for health check
        for j in {1..30}; do
            if curl -sf "http://${instance}:3000/health" > /dev/null; then
                echo "  ✓ $instance healthy"
                break
            fi
            sleep 2
        done
    done

    echo "Batch complete. Waiting 60s before next batch..."
    sleep 60
done

echo "Rolling deployment complete!"
```

---

## Summary and Best Practices

### Deployment Strategy Selection

| Scenario | Recommended Strategy | Justification |
|----------|---------------------|---------------|
| Development | Direct | Fast, simple rollback |
| Staging | Blue-Green | Test production process |
| Production (Low Traffic) | Blue-Green | Zero downtime, instant rollback |
| Production (High Traffic) | Canary | Gradual, monitored rollout |
| Microservices | Rolling | Minimize simultaneous changes |

### Key Metrics to Monitor

1. **Error Rate**: Should not increase >1%
2. **Latency (p99)**: Should not increase >50%
3. **Throughput**: Should not decrease >10%
4. **Memory Usage**: Monitor for leaks
5. **CPU Usage**: Check for performance degradation

### Rollback Decision Tree

```
Is error rate > threshold?
  YES → Immediate rollback
  NO  → Continue

Is latency > threshold?
  YES → Immediate rollback
  NO  → Continue

Are business metrics affected?
  YES → Evaluate and potentially rollback
  NO  → Proceed with deployment
```

---

## Total Lines: ~650 lines
**Complexity**: Advanced deployment patterns with monitoring and automation
**Production Ready**: Yes, with proper configuration
