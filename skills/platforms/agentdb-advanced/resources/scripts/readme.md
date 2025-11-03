# AgentDB Advanced Scripts

Production-ready scripts for distributed AgentDB operations.

## Scripts Overview

### quic_sync.py
**QUIC Synchronization Manager for AgentDB**

Sub-millisecond latency synchronization between distributed AgentDB nodes.

**Features**:
- QUIC protocol with TLS 1.3
- <1ms synchronization latency
- Automatic peer discovery
- Event-based broadcasting
- Batch operations (100 patterns/batch)
- Exponential backoff retry
- Compression support

**Usage**:
```bash
# Start QUIC server
python quic_sync.py --config ../templates/quic-config.yaml

# With environment variables
AGENTDB_QUIC_SYNC=true \
AGENTDB_QUIC_PORT=4433 \
AGENTDB_QUIC_PEERS=192.168.1.10:4433,192.168.1.11:4433 \
python quic_sync.py
```

**Dependencies**:
```bash
pip install aioquic asyncio
```

---

### multi_db_manage.sh
**Multi-Database Management Script**

Manage multiple AgentDB instances with sharding, backup, and optimization.

**Commands**:
```bash
# Initialize shards
./multi_db_manage.sh init <num_shards> [prefix]

# List databases
./multi_db_manage.sh list

# Backup all
./multi_db_manage.sh backup

# Restore backup
./multi_db_manage.sh restore <backup_path>

# Merge databases
./multi_db_manage.sh merge <output> <db1> <db2> ...

# Optimize all
./multi_db_manage.sh optimize

# Show statistics
./multi_db_manage.sh stats

# Reshard
./multi_db_manage.sh reshard <old_count> <new_count>
```

**Environment Variables**:
```bash
export AGENTDB_BASE_DIR=".agentdb"
export BACKUP_DIR="$AGENTDB_BASE_DIR/backups"
```

---

### custom_metrics.py
**Custom Distance Metrics Library**

15+ distance metrics for domain-specific vector similarity.

**Available Metrics**:
- **Standard**: cosine, euclidean, manhattan, chebyshev, minkowski
- **Advanced**: mahalanobis, canberra, angular, bhattacharyya, hellinger
- **Statistical**: pearson, spearman
- **Set-based**: hamming, jaccard
- **Domain-specific**: time_weighted, hierarchical, semantic_drift

**Usage**:
```python
from custom_metrics import CustomMetrics, MetricFactory

# Basic usage
vec1 = np.random.rand(384)
vec2 = np.random.rand(384)

# Cosine similarity
similarity = CustomMetrics.cosine_similarity(vec1, vec2)

# Weighted Euclidean
weights = np.array([1.0, 2.0, 1.5, ...])
distance = CustomMetrics.weighted_euclidean(vec1, vec2, weights)

# Metric factory
factory = MetricFactory()
print("Available:", factory.list_metrics())
distance = factory.compute('euclidean', vec1, vec2)

# Register custom metric
def my_metric(v1, v2):
    return np.sum(np.abs(v1 - v2))
factory.register('my_metric', my_metric)
```

**Custom Metric Creation**:
```python
from custom_metrics import create_weighted_metric, create_domain_metric

# Weighted metric
weights = np.array([1.0, 2.0, 1.5] + [1.0] * 381)
weighted_dist = create_weighted_metric(weights)
distance = weighted_dist(vec1, vec2)

# Domain-specific metric
config = {
    "type": "hierarchical",
    "hierarchy_levels": [[0, 128], [128, 256], [256, 384]],
    "weights": [1.0, 0.5, 0.25]
}
hierarchical_metric = create_domain_metric(config)
distance = hierarchical_metric(vec1, vec2)
```

---

## Installation

### Prerequisites
```bash
# Python 3.8+
python --version

# Required packages
pip install numpy aioquic asyncio

# Optional (for advanced features)
pip install scipy scikit-learn
```

### Make Scripts Executable
```bash
chmod +x quic_sync.py
chmod +x multi_db_manage.sh
chmod +x custom_metrics.py
```

---

## Configuration

### QUIC Sync Configuration
Copy and edit the QUIC config template:
```bash
cp ../templates/quic-config.yaml my-quic-config.yaml
vim my-quic-config.yaml
```

### Multi-DB Environment
Set environment variables:
```bash
export AGENTDB_BASE_DIR="/var/lib/agentdb"
export BACKUP_DIR="/var/backups/agentdb"
```

---

## Production Deployment

### QUIC Sync Deployment
```bash
# Generate TLS certificates
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Start on each node
python quic_sync.py --config production-quic-config.yaml &

# Monitor logs
tail -f .agentdb/logs/quic-sync.log
```

### Multi-DB Automated Backups
```bash
# Add to crontab
echo "0 2 * * * /path/to/multi_db_manage.sh backup" | crontab -

# Verify cron job
crontab -l
```

---

## Monitoring

### QUIC Sync Metrics
```bash
# Check sync status
tail -f .agentdb/logs/quic-sync.log | grep "Broadcasted pattern"

# Monitor latency
tail -f .agentdb/logs/quic-sync.log | grep "latency"
```

### Database Statistics
```bash
# Show stats
./multi_db_manage.sh stats

# Watch stats in real-time
watch -n 5 './multi_db_manage.sh stats'
```

---

## Troubleshooting

### QUIC Sync Issues
```bash
# Check firewall
sudo ufw status
sudo ufw allow 4433/udp

# Verify peer connectivity
ping <peer-host>
telnet <peer-host> 4433

# Enable debug logging
export DEBUG=agentdb:quic
python quic_sync.py --config config.yaml
```

### Multi-DB Issues
```bash
# Check disk space
df -h

# Verify database integrity
sqlite3 .agentdb/shard-0.db "PRAGMA integrity_check"

# Fix permissions
chmod -R 755 .agentdb/
chown -R $(whoami) .agentdb/
```

### Custom Metrics Issues
```python
# Verify vector dimensions
assert vec1.shape == vec2.shape, "Dimension mismatch"

# Check for NaN/Inf
assert not np.isnan(vec1).any(), "NaN in vector 1"
assert not np.isinf(vec1).any(), "Inf in vector 1"

# Normalize if needed
vec1_norm = vec1 / np.linalg.norm(vec1)
vec2_norm = vec2 / np.linalg.norm(vec2)
```

---

## Performance Benchmarks

### QUIC Sync
- **Latency**: P50 < 10ms, P95 < 50ms, P99 < 100ms
- **Throughput**: > 100 patterns/sec
- **CPU**: < 50% per node
- **Memory**: < 200MB per node

### Multi-DB Management
- **Backup**: < 10s for 5 shards
- **Restore**: < 15s for 5 shards
- **Merge**: < 5s for 5 shards
- **Optimize**: ~30% space reclaimed

### Custom Metrics
- **Cosine**: 10-20 µs per comparison
- **Euclidean**: 15-25 µs per comparison
- **Weighted**: 20-30 µs per comparison
- **Mahalanobis**: 50-100 µs per comparison (with covariance matrix)

---

## Examples

See parent directory `examples/` for comprehensive usage examples:
- `example-1-quic-sync.md` - QUIC synchronization patterns
- `example-2-multi-database.md` - Multi-database management
- `example-3-sharding.md` - Sharding strategies

---

## Testing

See `../../tests/` for comprehensive test suites:
- `test-1-quic-sync.md` - QUIC sync validation (6 test cases)
- `test-2-multi-db.md` - Multi-DB operations (10 test cases)
- `test-3-hybrid-search.md` - Hybrid search features (10 test cases)

---

## Additional Resources

- **Main Documentation**: `../../skill.md`
- **Templates**: `../templates/`
- **AgentDB Docs**: https://agentdb.ruv.io
- **QUIC Protocol**: https://www.rfc-editor.org/rfc/rfc9000.html

---

**Last Updated**: 2025-01-01
**Version**: 1.0.0
