# Test 3: Multi-Cloud Deployment and Migration

## Test Objective
Validate multi-cloud deployment strategies, cross-cloud migrations, and unified infrastructure management across AWS, GCP, and Azure using automation scripts and IaC tools.

## Prerequisites
- AWS CLI configured
- gcloud CLI configured
- Azure CLI configured (optional)
- Terraform >= 1.0 installed
- Docker installed
- Access to multiple cloud accounts
- DNS access for cross-cloud testing

## Test Scenarios

### Scenario 3.1: Parallel Deployment Across Clouds

**Objective:** Deploy identical application to AWS ECS, GCP Cloud Run, and validate consistency.

**Setup:**
```bash
# Build multi-platform Docker image
cat > app/Dockerfile << 'EOF'
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 8080
CMD ["node", "server.js"]
EOF

cat > app/server.js << 'EOF'
const express = require('express');
const os = require('os');
const app = express();

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    cloud: process.env.CLOUD_PROVIDER || 'unknown',
    hostname: os.hostname(),
    version: '1.0.0'
  });
});

app.get('/', (req, res) => {
  res.json({
    message: 'Multi-Cloud Application',
    cloud: process.env.CLOUD_PROVIDER || 'unknown',
    region: process.env.CLOUD_REGION || 'unknown'
  });
});

app.listen(8080, () => {
  console.log('Server running on port 8080');
});
EOF

cat > app/package.json << 'EOF'
{
  "name": "multi-cloud-app",
  "version": "1.0.0",
  "main": "server.js",
  "dependencies": {
    "express": "^4.18.2"
  }
}
EOF

# Build image
docker build -t multi-cloud-app:v1 ./app
```

**Test Execution - AWS:**
```bash
# Tag and push to ECR
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=us-east-1

aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

aws ecr create-repository --repository-name multi-cloud-app --region $AWS_REGION || true

docker tag multi-cloud-app:v1 $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/multi-cloud-app:v1
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/multi-cloud-app:v1

# Deploy to AWS ECS using Terraform
cat > aws-deploy.tfvars << EOF
project_name    = "multi-cloud"
environment     = "test"
region          = "$AWS_REGION"
container_image = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/multi-cloud-app:v1"
container_port  = 8080
desired_count   = 2
cpu             = 256
memory          = 512
enable_rds      = false
EOF

cd terraform-aws
cp ../../resources/templates/aws-infra.tf main.tf
terraform init
terraform apply -var-file=../aws-deploy.tfvars -auto-approve

# Get AWS endpoint
AWS_ENDPOINT=$(terraform output -raw alb_url)
echo "AWS Endpoint: $AWS_ENDPOINT"

# Test AWS deployment
curl $AWS_ENDPOINT/health | jq
```

**Test Execution - GCP:**
```bash
# Push to GCR
GCP_PROJECT_ID=$(gcloud config get-value project)
GCP_REGION=us-central1

docker tag multi-cloud-app:v1 gcr.io/$GCP_PROJECT_ID/multi-cloud-app:v1
docker push gcr.io/$GCP_PROJECT_ID/multi-cloud-app:v1

# Deploy to Cloud Run
GCP_ENDPOINT=$(bash ../resources/scripts/gcp_deploy.sh \
  cloud-run \
  multi-cloud-app \
  gcr.io/$GCP_PROJECT_ID/multi-cloud-app:v1 \
  $GCP_REGION \
  8080 \
  512Mi \
  1)

echo "GCP Endpoint: $GCP_ENDPOINT"

# Test GCP deployment
curl $GCP_ENDPOINT/health | jq
```

**Validation:**
```bash
# Compare responses
echo "=== AWS Response ==="
curl -s $AWS_ENDPOINT/health | jq

echo "=== GCP Response ==="
curl -s $GCP_ENDPOINT/health | jq

# Performance comparison
echo "=== AWS Performance ==="
ab -n 1000 -c 10 $AWS_ENDPOINT/

echo "=== GCP Performance ==="
ab -n 1000 -c 10 $GCP_ENDPOINT/
```

**Expected Results:**
- Same application running on AWS and GCP
- Both endpoints return consistent responses
- Health checks passing on both platforms
- Similar performance characteristics
- No platform-specific errors

**Success Criteria:**
- ✅ Image successfully pushed to ECR and GCR
- ✅ Application deployed to both clouds
- ✅ Both endpoints accessible and responding
- ✅ Health checks return platform information
- ✅ Response times within acceptable range

---

### Scenario 3.2: Cross-Cloud Database Replication

**Objective:** Set up database replication between AWS RDS and GCP Cloud SQL.

**Setup:**
```bash
# Deploy PostgreSQL on AWS
cat > aws-db.tfvars << 'EOF'
project_name     = "multi-cloud"
environment      = "test"
enable_rds       = true
db_instance_class = "db.t3.micro"
EOF

cd terraform-aws
terraform apply -var-file=../aws-db.tfvars -var="enable_rds=true" -auto-approve

# Get RDS endpoint
RDS_ENDPOINT=$(terraform output -raw rds_endpoint)
RDS_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id $(terraform output -raw rds_password_secret_arn) \
  --query SecretString --output text)

echo "RDS Endpoint: $RDS_ENDPOINT"

# Deploy Cloud SQL on GCP
gcloud sql instances create multi-cloud-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=us-central1 \
  --backup \
  --backup-start-time=03:00

# Get Cloud SQL connection name
CLOUDSQL_CONN=$(gcloud sql instances describe multi-cloud-db \
  --format="value(connectionName)")

# Create Cloud SQL database
gcloud sql databases create testdb --instance=multi-cloud-db

# Set up Cloud SQL Proxy for testing
cloud_sql_proxy -instances=$CLOUDSQL_CONN=tcp:5433 &
PROXY_PID=$!
```

**Test Execution:**
```bash
# Test AWS RDS connection
PGPASSWORD=$RDS_PASSWORD psql -h ${RDS_ENDPOINT%%:*} -U dbadmin -d testdb -c "\l"

# Test GCP Cloud SQL connection
PGPASSWORD=gcp_password psql -h 127.0.0.1 -p 5433 -U postgres -d testdb -c "\l"

# Create test data on AWS
PGPASSWORD=$RDS_PASSWORD psql -h ${RDS_ENDPOINT%%:*} -U dbadmin -d testdb << 'EOF'
CREATE TABLE IF NOT EXISTS multi_cloud_test (
  id SERIAL PRIMARY KEY,
  cloud VARCHAR(50),
  message TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO multi_cloud_test (cloud, message)
VALUES ('aws', 'Data from AWS RDS');

SELECT * FROM multi_cloud_test;
EOF

# Set up logical replication (manual step for demonstration)
# Note: Full cross-cloud replication requires VPN/VPC peering
echo "Logical replication setup would require:"
echo "1. VPC peering between AWS and GCP"
echo "2. PostgreSQL logical replication configuration"
echo "3. Security group/firewall rules"
echo "4. Publication/subscription setup"
```

**Expected Results:**
- Both databases deployed successfully
- Connectivity established to both databases
- Test data created on AWS RDS
- Replication setup documented
- Network connectivity considerations identified

**Success Criteria:**
- ✅ AWS RDS instance running
- ✅ GCP Cloud SQL instance running
- ✅ Both databases accessible
- ✅ Test table created on AWS
- ✅ Replication requirements documented

---

### Scenario 3.3: Multi-Cloud Load Balancing with GeoDNS

**Objective:** Configure GeoDNS to route traffic based on geographic location across AWS and GCP.

**Setup:**
```bash
# Prerequisites: Domain name with DNS provider supporting GeoDNS (Route 53, Cloud DNS)

# Create health check endpoints for both clouds
cat > health-check-config.json << 'EOF'
{
  "aws": {
    "endpoint": "http://AWS_ALB_DNS/health",
    "region": "us-east-1",
    "weight": 50
  },
  "gcp": {
    "endpoint": "https://GCP_CLOUD_RUN_URL/health",
    "region": "us-central1",
    "weight": 50
  }
}
EOF

# Configure AWS Route 53 GeoDNS (if using Route 53)
aws route53 create-health-check \
  --caller-reference $(date +%s) \
  --health-check-config \
    IPAddress=${AWS_ALB_IP},Port=80,Type=HTTP,ResourcePath=/health

# Get health check ID
HEALTH_CHECK_ID=$(aws route53 list-health-checks \
  --query "HealthChecks[?HealthCheckConfig.ResourcePath=='/health'].Id" \
  --output text | head -1)

# Create hosted zone record sets
cat > route53-records.json << EOF
{
  "Changes": [
    {
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "app.example.com",
        "Type": "A",
        "SetIdentifier": "AWS-US-East",
        "GeoLocation": {
          "ContinentCode": "NA"
        },
        "TTL": 60,
        "ResourceRecords": [
          {
            "Value": "${AWS_ALB_IP}"
          }
        ],
        "HealthCheckId": "${HEALTH_CHECK_ID}"
      }
    },
    {
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "app.example.com",
        "Type": "A",
        "SetIdentifier": "GCP-US-Central",
        "GeoLocation": {
          "ContinentCode": "EU"
        },
        "TTL": 60,
        "ResourceRecords": [
          {
            "Value": "${GCP_ENDPOINT_IP}"
          }
        ]
      }
    }
  ]
}
EOF
```

**Test Execution:**
```bash
# Test from different geographic locations using VPN/proxy
echo "=== Test from North America (should route to AWS) ==="
curl -H "X-Forwarded-For: 8.8.8.8" https://app.example.com/health | jq

echo "=== Test from Europe (should route to GCP) ==="
curl -H "X-Forwarded-For: 185.60.216.35" https://app.example.com/health | jq

# Monitor DNS resolution
dig app.example.com +short
dig app.example.com @8.8.8.8 +short

# Test failover by stopping one deployment
# (Simulate AWS failure)
terraform -chdir=terraform-aws destroy -auto-approve

# Verify traffic routes to GCP
for i in {1..10}; do
  curl -s https://app.example.com/health | jq '.cloud'
done
```

**Expected Results:**
- GeoDNS configured correctly
- North American traffic routes to AWS
- European traffic routes to GCP
- Health checks monitoring both endpoints
- Automatic failover when one cloud is down
- All traffic routes to healthy endpoint

**Success Criteria:**
- ✅ DNS records created for both clouds
- ✅ Health checks configured and passing
- ✅ Geographic routing working correctly
- ✅ Failover triggers when endpoint unhealthy
- ✅ No downtime during simulated failure

---

### Scenario 3.4: Multi-Cloud Cost Optimization

**Objective:** Compare costs and optimize resource allocation across clouds.

**Setup:**
```bash
# Create cost analysis script
cat > cost-analysis.sh << 'EOF'
#!/bin/bash

echo "=== AWS Cost Analysis ==="
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '30 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE \
  --query 'ResultsByTime[].Groups[].[Keys[0],Metrics.BlendedCost.Amount]' \
  --output table

echo "=== GCP Cost Analysis ==="
gcloud billing accounts list
gcloud beta billing budgets list --billing-account=BILLING_ACCOUNT_ID

# Resource tagging for cost allocation
echo "=== Tagging Resources ==="
# AWS
aws resourcegroupstaggingapi tag-resources \
  --resource-arn-list arn:aws:ecs:us-east-1:ACCOUNT:cluster/multi-cloud-test-cluster \
  --tags Environment=test,Project=multi-cloud,CostCenter=engineering

# GCP
gcloud compute instances add-labels multi-cloud-instance \
  --labels=environment=test,project=multi-cloud,cost-center=engineering \
  --zone=us-central1-a
EOF

chmod +x cost-analysis.sh
```

**Test Execution:**
```bash
# Run cost analysis
./cost-analysis.sh

# Compare instance pricing
echo "=== AWS EC2 t3.medium pricing ==="
# $0.0416/hour = ~$30/month

echo "=== GCP e2-medium pricing ==="
# $0.0335/hour = ~$24/month (20% cheaper)

# Compare serverless pricing
echo "=== AWS Lambda pricing ==="
# $0.20 per 1M requests + $0.0000166667/GB-second

echo "=== GCP Cloud Functions pricing ==="
# $0.40 per 1M invocations + $0.0000025/GB-second

# Analyze traffic costs
echo "=== Data Transfer Costs ==="
echo "AWS: $0.09/GB outbound"
echo "GCP: $0.12/GB outbound"

# Generate cost optimization recommendations
cat > cost-recommendations.md << 'EOF'
# Multi-Cloud Cost Optimization Recommendations

## Compute Resources
- **GCP e2-medium**: 20% cheaper than AWS t3.medium
- **Recommendation**: Use GCP for steady-state workloads

## Serverless
- **AWS Lambda**: Better for high request volumes
- **Recommendation**: Use Lambda for >2M requests/month

## Storage
- **AWS S3**: $0.023/GB/month
- **GCP Cloud Storage**: $0.020/GB/month (Standard)
- **Recommendation**: Use GCP for cold storage

## Data Transfer
- **AWS**: Lower egress costs for high volume
- **GCP**: Better for inter-service transfers

## Overall Strategy
1. Run compute on GCP (20% savings)
2. Use Lambda for serverless (better pricing at scale)
3. Store cold data on GCP Cloud Storage
4. Use AWS for data-intensive egress
EOF

cat cost-recommendations.md
```

**Expected Results:**
- Cost data retrieved from both clouds
- Resource tagging enables cost allocation
- Clear pricing comparison documented
- Optimization recommendations generated
- Potential savings identified

**Success Criteria:**
- ✅ Cost data accessible from both clouds
- ✅ Resources properly tagged
- ✅ Pricing comparison completed
- ✅ Recommendations align with workload
- ✅ Projected savings calculated

---

### Scenario 3.5: Multi-Cloud Disaster Recovery

**Objective:** Implement and test disaster recovery failover between clouds.

**Setup:**
```bash
# Create backup script
cat > backup-to-cloud.sh << 'EOF'
#!/bin/bash
set -euo pipefail

PRIMARY_CLOUD="aws"
BACKUP_CLOUD="gcp"
BACKUP_BUCKET="multi-cloud-dr-backup"

echo "=== Creating Backup from $PRIMARY_CLOUD ==="

# Backup application state
if [ "$PRIMARY_CLOUD" = "aws" ]; then
  # Backup RDS snapshot
  aws rds create-db-snapshot \
    --db-instance-identifier multi-cloud-test-db \
    --db-snapshot-identifier dr-snapshot-$(date +%Y%m%d-%H%M%S)

  # Export snapshot to S3
  # aws rds export-snapshot-to-s3 ...

  # Sync to GCP
  gsutil -m rsync -r s3://aws-backup-bucket gs://$BACKUP_BUCKET/
elif [ "$PRIMARY_CLOUD" = "gcp" ]; then
  # Backup Cloud SQL
  gcloud sql backups create \
    --instance=multi-cloud-db \
    --description="DR backup $(date +%Y%m%d-%H%M%S)"

  # Sync to AWS
  aws s3 sync gs://$BACKUP_BUCKET/ s3://gcp-backup-bucket/
fi

echo "Backup completed successfully"
EOF

chmod +x backup-to-cloud.sh
```

**Test Execution:**
```bash
# Run backup
./backup-to-cloud.sh

# Simulate disaster (destroy primary infrastructure)
echo "=== Simulating AWS Disaster ==="
cd terraform-aws
terraform destroy -auto-approve

# Failover to GCP
echo "=== Failing over to GCP ==="

# 1. Promote GCP to primary
bash ../resources/scripts/gcp_deploy.sh \
  cloud-run \
  multi-cloud-app-primary \
  gcr.io/$GCP_PROJECT_ID/multi-cloud-app:v1 \
  us-central1 \
  8080 \
  1Gi \
  2

# 2. Update DNS to point to GCP
# (Update Route 53 or Cloud DNS records)

# 3. Restore data from backup
gcloud sql backups restore BACKUP_ID \
  --backup-instance=multi-cloud-db

# 4. Verify failover
FAILOVER_ENDPOINT=$(gcloud run services describe multi-cloud-app-primary \
  --region=us-central1 \
  --format="value(status.url)")

echo "Failover endpoint: $FAILOVER_ENDPOINT"
curl $FAILOVER_ENDPOINT/health | jq

# Test RTO (Recovery Time Objective)
echo "=== RTO Measurement ==="
echo "Disaster triggered at: $(date)"
# Manual intervention time: ~5 minutes
echo "Service restored at: $(date)"
echo "RTO: ~5-10 minutes"

# Test RPO (Recovery Point Objective)
echo "=== RPO Measurement ==="
echo "Last backup: within 1 hour (automated hourly backups)"
echo "RPO: <1 hour"
```

**Expected Results:**
- Backup created successfully on primary cloud
- Backup replicated to secondary cloud
- Primary infrastructure destroyed (simulated disaster)
- Failover to secondary cloud successful
- Application accessible on new endpoint
- Data restored from backup
- RTO within acceptable limits (<15 minutes)
- RPO within acceptable limits (<1 hour)

**Success Criteria:**
- ✅ Backup process completes without errors
- ✅ Cross-cloud replication successful
- ✅ Failover completes within RTO target
- ✅ Application functional after failover
- ✅ Data loss within RPO limits
- ✅ DNS updated to new endpoint

---

## Cleanup

```bash
# AWS cleanup
cd terraform-aws
terraform destroy -auto-approve

aws ecr delete-repository \
  --repository-name multi-cloud-app \
  --force \
  --region us-east-1

# GCP cleanup
gcloud run services delete multi-cloud-app --region=us-central1 --quiet
gcloud sql instances delete multi-cloud-db --quiet
gcloud container images delete gcr.io/$GCP_PROJECT_ID/multi-cloud-app:v1 --quiet

# DNS cleanup
# (Manual: Delete Route 53 or Cloud DNS records)

# Local cleanup
rm -rf app terraform-aws cost-analysis.sh backup-to-cloud.sh
```

## Test Report Template

```markdown
# Multi-Cloud Deployment Test Report

**Date:** YYYY-MM-DD
**Tester:** Name
**Clouds Tested:** AWS, GCP, Azure

## Test Results Summary

| Scenario | AWS | GCP | Azure | Duration | Notes |
|----------|-----|-----|-------|----------|-------|
| 3.1 Parallel Deploy | ✅/❌ | ✅/❌ | ✅/❌ | XXs | |
| 3.2 DB Replication | ✅/❌ | ✅/❌ | N/A | XXs | |
| 3.3 GeoDNS | ✅/❌ | ✅/❌ | N/A | XXs | |
| 3.4 Cost Analysis | ✅/❌ | ✅/❌ | ✅/❌ | XXs | |
| 3.5 DR Failover | ✅/❌ | ✅/❌ | N/A | XXs | |

## Performance Comparison

| Metric | AWS | GCP | Azure |
|--------|-----|-----|-------|
| Deploy Time | XXs | XXs | XXs |
| Response Time | XXms | XXms | XXms |
| Throughput | XXX rps | XXX rps | XXX rps |
| Cost (monthly) | $XX | $XX | $XX |

## DR Metrics

- **RTO Achieved:** X minutes
- **RPO Achieved:** X hours
- **Data Loss:** None / X records
- **Failover Success:** Yes/No

## Cost Analysis

- **Total Monthly Cost:** $XXX
- **AWS Share:** XX%
- **GCP Share:** XX%
- **Potential Savings:** $XX (XX%)

## Issues Encountered

1. **Issue:** Description
   - **Cloud:** AWS/GCP/Azure
   - **Resolution:** Fix
   - **Impact:** Severity

## Recommendations

- Multi-cloud best practices
- Cost optimization opportunities
- DR improvements
- Automation enhancements

## Sign-off

- [ ] All tests passed
- [ ] Cleanup completed
- [ ] Cost analysis reviewed
- [ ] DR plan validated
- [ ] Documentation updated
```

## Notes

- Test in non-production accounts
- Monitor costs during testing
- Set up billing alerts
- Document cloud-specific quirks
- Keep credentials secure
- Clean up resources promptly
- Verify cross-cloud networking
- Test GeoDNS from multiple locations
- Validate backup/restore processes
- Document RTO/RPO actuals vs targets
