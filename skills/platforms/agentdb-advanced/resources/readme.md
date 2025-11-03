# AgentDB Advanced Resources

This directory contains production-ready scripts, templates, and configurations for advanced AgentDB features including QUIC synchronization, multi-database management, and custom distance metrics.

## Directory Structure

```
resources/
├── scripts/                    # Executable scripts
│   ├── quic_sync.py           # QUIC synchronization manager
│   ├── multi_db_manage.sh     # Multi-database management
│   └── custom_metrics.py      # Custom distance metrics
└── templates/                  # Configuration templates
    ├── quic-config.yaml       # QUIC sync configuration
    ├── distributed-db.json    # Distributed database setup
    └── hybrid-search.yaml     # Hybrid search configuration
```

## Scripts

### quic_sync.py
**Purpose**: Sub-millisecond latency synchronization between AgentDB nodes

**Features**:
- QUIC protocol with TLS 1.3 encryption
- Automatic retry and recovery
- Event-based broadcasting
- Batch synchronization (100 patterns/batch)
- <1ms latency between nodes

**Usage**:
```bash
# Start QUIC sync server
python resources/scripts/quic_sync.py --config quic-config.yaml

# With custom peers
python resources/scripts/quic_sync.py \
  --host 0.0.0.0 \
  --port 4433 \
  --peers "192.168.1.10:4433,192.168.1.11:4433"
```

**Dependencies**:
```bash
pip install aioquic asyncio
```

**Configuration**:
See `templates/quic-config.yaml` for full configuration options.

---

### multi_db_manage.sh
**Purpose**: Manage multiple AgentDB instances with sharding, backup, and optimization

**Features**:
- Initialize multiple database shards
- Domain-based shard routing
- Backup/restore all databases
- Merge multiple databases
- Optimize (VACUUM, ANALYZE, reindex)
- Database statistics
- Resharding support

**Usage**:
```bash
# Initialize 5 shards
./resources/scripts/multi_db_manage.sh init 5 shard

# List all databases
./resources/scripts/multi_db_manage.sh list

# Backup all databases
./resources/scripts/multi_db_manage.sh backup

# Restore from backup
./resources/scripts/multi_db_manage.sh restore ./backups/backup_20250101_120000

# Merge databases
./resources/scripts/multi_db_manage.sh merge merged.db shard-0.db shard-1.db shard-2.db

# Optimize all databases
./resources/scripts/multi_db_manage.sh optimize

# Show statistics
./resources/scripts/multi_db_manage.sh stats

# Reshard from 5 to 8 shards
./resources/scripts/multi_db_manage.sh reshard 5 8
```

**Environment Variables**:
```bash
export AGENTDB_BASE_DIR=".agentdb"     # Base directory for databases
export BACKUP_DIR="$AGENTDB_BASE_DIR/backups"  # Backup location
```

---

### custom_metrics.py
**Purpose**: Custom distance metrics for domain-specific similarity

**Features**:
- 15+ distance metrics (cosine, euclidean, manhattan, etc.)
- Weighted metrics with custom importance
- Time-weighted similarity
- Hierarchical similarity
- Semantic drift detection
- Metric factory for registration

**Usage**:
```python
from resources.scripts.custom_metrics import CustomMetrics, MetricFactory

# Basic usage
vec1 = np.random.rand(384)
vec2 = np.random.rand(384)

# Cosine similarity
similarity = CustomMetrics.cosine_similarity(vec1, vec2)

# Weighted Euclidean
weights = np.array([1.0, 2.0, 1.5, ...])
distance = CustomMetrics.weighted_euclidean(vec1, vec2, weights)

# Use factory
factory = MetricFactory()
distance = factory.compute('euclidean', vec1, vec2)

# Register custom metric
factory.register('my_metric', my_custom_function)
```

**Available Metrics**:
- Standard: cosine, euclidean, manhattan, chebyshev, minkowski
- Advanced: mahalanobis, canberra, angular, bhattacharyya, hellinger
- Statistical: pearson, spearman
- Set-based: hamming, jaccard
- Domain-specific: time_weighted, hierarchical, semantic_drift

---

## Templates

### quic-config.yaml
**Purpose**: QUIC synchronization configuration

**Key Sections**:
- Server configuration (host, port, TLS)
- Peer nodes (addresses, priorities)
- Sync settings (interval, batch size, retries)
- QUIC protocol settings (streams, timeout, 0-RTT)
- Pattern filtering (domains, confidence thresholds)
- Conflict resolution strategies
- Performance tuning
- Monitoring and logging
- Security (authentication, encryption)
- High availability (failover, health checks)

**Usage**:
```bash
# Copy template
cp resources/templates/quic-config.yaml my-config.yaml

# Edit configuration
vim my-config.yaml

# Use with QUIC sync
python resources/scripts/quic_sync.py --config my-config.yaml
```

---

### distributed-db.json
**Purpose**: Distributed AgentDB deployment configuration

**Key Sections**:
- Topology (mesh, hierarchical, ring, star)
- Node configuration (roles, regions, shards)
- Sharding strategy (consistent hashing, replication)
- Replication settings (async/sync, consistency levels)
- Database definitions (knowledge, conversation, code, reasoning)
- Load balancing (strategies, health checks, circuit breakers)
- Caching (distributed Redis, local cache)
- Performance tuning (connection pooling, batch operations, HNSW)
- Monitoring (metrics, tracing, logging)
- Security (TLS, authentication, authorization, encryption)
- Backup and disaster recovery
- Auto-scaling configuration
- Data lifecycle management

**Usage**:
```javascript
import fs from 'fs';

// Load configuration
const config = JSON.parse(fs.readFileSync('resources/templates/distributed-db.json'));

// Initialize distributed AgentDB
const nodes = await initializeDistributedCluster(config);

// Access specific database
const knowledgeDB = nodes.getDatabaseByName('knowledge_base');
```

---

### hybrid-search.yaml
**Purpose**: Hybrid search configuration (vector + metadata)

**Key Sections**:
- Vector search configuration (metric, k, HNSW, MMR)
- Metadata filtering (numeric, string, boolean, temporal, geographic)
- Hybrid scoring (component weights, aggregation methods)
- Boosting (recency, popularity, verification)
- Re-ranking (cross-encoder models)
- Query expansion (synonyms, embeddings, history)
- Result diversification (MMR, cluster-based, category-based)
- Performance optimization (caching, pagination, early termination)
- Analytics and A/B testing
- Example use cases (research papers, e-commerce, code search)

**Usage**:
```python
import yaml

# Load configuration
with open('resources/templates/hybrid-search.yaml') as f:
    config = yaml.safe_load(f)

# Apply hybrid search
result = await adapter.retrieveWithReasoning(query_embedding, {
    'domain': 'research-papers',
    'k': config['vector_search']['k'],
    'filters': config['metadata_filters']['numeric'],
    'hybridWeights': config['hybrid_scoring']['components'],
    'useMMR': config['vector_search']['mmr']['enabled']
})
```

---

## Integration with AgentDB

### QUIC Synchronization Integration
```typescript
import { createAgentDBAdapter } from 'agentic-flow/reasoningbank';

const adapter = await createAgentDBAdapter({
  dbPath: '.agentdb/distributed.db',
  enableQUICSync: true,
  syncPort: 4433,
  syncPeers: ['192.168.1.10:4433', '192.168.1.11:4433'],
  syncInterval: 1000,
  syncBatchSize: 100
});

// Patterns automatically sync across peers
await adapter.insertPattern({
  type: 'knowledge',
  domain: 'distributed-sync',
  pattern_data: JSON.stringify({ /* ... */ })
});
```

### Multi-Database Sharding
```typescript
// Shard by domain for horizontal scaling
const shards = {
  'knowledge': await createAgentDBAdapter({ dbPath: '.agentdb/shard-knowledge.db' }),
  'code': await createAgentDBAdapter({ dbPath: '.agentdb/shard-code.db' }),
  'conversation': await createAgentDBAdapter({ dbPath: '.agentdb/shard-conversation.db' })
};

function getDBForDomain(domain: string) {
  const shardKey = domain.split('-')[0];
  return shards[shardKey] || shards['knowledge'];
}

// Route to correct shard
const db = getDBForDomain('knowledge-ml');
await db.insertPattern({ /* ... */ });
```

### Hybrid Search Integration
```typescript
// Hybrid search: vector similarity + metadata filters
const result = await adapter.retrieveWithReasoning(queryEmbedding, {
  domain: 'research-papers',
  k: 20,
  metric: 'cosine',
  filters: {
    year: { $gte: 2023 },
    citations: { $gte: 50 },
    is_peer_reviewed: true
  },
  hybridWeights: {
    vectorSimilarity: 0.6,
    metadataScore: 0.4
  },
  useMMR: true,
  mmrLambda: 0.5
});
```

---

## Production Deployment

### Prerequisites
- AgentDB v1.0.7+
- Node.js 18+ / Python 3.8+
- Network connectivity between nodes
- SSL/TLS certificates (for QUIC sync)
- Sufficient disk space (for backups)

### Deployment Steps

1. **Configure QUIC Synchronization**
   ```bash
   # Copy and edit QUIC config
   cp resources/templates/quic-config.yaml production-quic.yaml
   vim production-quic.yaml

   # Generate production certificates
   openssl req -x509 -newkey rsa:4096 -keyout prod-key.pem -out prod-cert.pem -days 365 -nodes

   # Start QUIC sync on each node
   python resources/scripts/quic_sync.py --config production-quic.yaml
   ```

2. **Initialize Multi-Database Shards**
   ```bash
   # Set production directory
   export AGENTDB_BASE_DIR="/var/lib/agentdb"
   export BACKUP_DIR="/var/backups/agentdb"

   # Initialize shards
   ./resources/scripts/multi_db_manage.sh init 8 prod-shard

   # Set up automated backups (cron)
   echo "0 2 * * * /path/to/multi_db_manage.sh backup" | crontab -
   ```

3. **Configure Hybrid Search**
   ```bash
   # Copy hybrid search config
   cp resources/templates/hybrid-search.yaml production-search.yaml

   # Tune for your use case
   vim production-search.yaml

   # Load in application
   # (application code loads and uses YAML config)
   ```

4. **Enable Monitoring**
   ```bash
   # QUIC sync metrics
   curl http://localhost:9090/metrics

   # Database statistics
   ./resources/scripts/multi_db_manage.sh stats

   # Set up alerts (Prometheus/Grafana)
   ```

---

## Performance Tuning

### QUIC Synchronization
- **Latency**: Reduce `sync_interval` to 500ms for lower latency
- **Throughput**: Increase `batch_size` to 200 for higher throughput
- **Reliability**: Increase `max_retries` to 5 for flaky networks
- **Bandwidth**: Enable compression for limited bandwidth

### Multi-Database Management
- **Sharding**: Use 8-16 shards for optimal parallelism
- **Caching**: Set `cache_size` to 5000 for frequently accessed patterns
- **Optimization**: Run `optimize` weekly to reclaim space
- **Backups**: Use incremental backups for large databases

### Hybrid Search
- **Recall**: Increase `k` to 50 and re-rank to final 10
- **Precision**: Use stricter filters and higher thresholds
- **Diversity**: Enable MMR with `lambda=0.5` for balanced diversity
- **Latency**: Use local caching with 5-minute TTL

---

## Troubleshooting

### QUIC Sync Issues
```bash
# Check firewall
sudo ufw allow 4433/udp

# Verify peer connectivity
ping <peer-host>

# Enable debug logging
export DEBUG=agentdb:quic
python resources/scripts/quic_sync.py --config config.yaml
```

### Multi-Database Issues
```bash
# Check disk space
df -h

# Verify database integrity
sqlite3 .agentdb/shard-0.db "PRAGMA integrity_check"

# Fix corruption
./resources/scripts/multi_db_manage.sh backup
./resources/scripts/multi_db_manage.sh restore <backup_path>
```

### Hybrid Search Issues
```python
# Debug filters
print("Available metadata:", list(pattern.metadata.keys()))

# Relax filters
result = await adapter.retrieveWithReasoning(query, {'k': 100, 'filters': {}})

# Check vector dimensions
print("Query dim:", len(query_embedding))
print("Pattern dim:", len(pattern.embedding))
```

---

## Additional Resources

- **AgentDB Documentation**: https://agentdb.ruv.io
- **QUIC Protocol**: https://www.rfc-editor.org/rfc/rfc9000.html
- **Vector Search Best Practices**: https://www.pinecone.io/learn/vector-search/
- **Hybrid Search Guide**: See `../SKILL.md` for detailed examples

---

**Last Updated**: 2025-01-01
**Version**: 1.0.0
**Maintainer**: AgentDB Advanced Team
