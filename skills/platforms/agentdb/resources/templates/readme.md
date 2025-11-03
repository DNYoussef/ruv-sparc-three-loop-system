# AgentDB Configuration Templates

Production-ready configuration templates for AgentDB vector database deployment.

## 📁 Templates Overview

### 1. `vector-config.yaml`
**Purpose**: Complete AgentDB configuration for production deployment

**Sections**:
- **Collection**: HNSW index parameters, distance metrics
- **Embedding**: Model selection, batch processing
- **Quantization**: Memory optimization settings
- **Performance**: Caching, memory mapping, timeouts
- **Memory Patterns**: Triple-layer retention (24h/7d/30d)
- **Search**: Result limits, similarity thresholds
- **Monitoring**: Metrics collection and export
- **Persistence**: Backups, write-ahead logging
- **Integration**: Memory-MCP, RAG, API endpoints

**Quick Start**:
```yaml
collection:
  name: "my_agentdb"
  metadata:
    hnsw:space: "cosine"
    hnsw:construction_ef: 200
    hnsw:M: 16

embedding:
  model: "all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 128
```

### 2. `collection-schema.json`
**Purpose**: JSON Schema for AgentDB collection definitions

**Features**:
- **Type validation**: Enforce correct data types
- **Required fields**: Ensure critical fields present
- **Metadata schema**: Define custom field types
- **Validation rules**: Min/max, patterns, enums
- **Retention policies**: TTL and document limits
- **Search configuration**: Default parameters

**Schema Structure**:
```json
{
  "collection": {
    "name": "agent_memory",
    "dimension": 384,
    "distance_metric": "cosine"
  },
  "metadata_schema": {
    "fields": [
      {
        "name": "timestamp",
        "type": "datetime",
        "required": true,
        "indexed": true
      }
    ]
  }
}
```

## 🚀 Usage Examples

### Basic Setup

```python
import chromadb
import yaml

# Load configuration
with open('vector-config.yaml') as f:
    config = yaml.safe_load(f)

# Create client
client = chromadb.Client()

# Create collection with config
collection = client.create_collection(
    name=config['collection']['name'],
    metadata=config['collection']['metadata']
)
```

### With Schema Validation

```python
import json
from jsonschema import validate

# Load schema
with open('collection-schema.json') as f:
    schema = json.load(f)

# Validate configuration
user_config = {
    "collection": {
        "name": "test_collection",
        "dimension": 384,
        "distance_metric": "cosine"
    },
    "embedding": {
        "model_name": "all-MiniLM-L6-v2"
    },
    "metadata_schema": {
        "fields": [
            {
                "name": "timestamp",
                "type": "datetime",
                "required": True
            }
        ]
    }
}

# Validate against schema
validate(instance=user_config, schema=schema)
print("✅ Configuration valid")
```

## 🎯 Configuration Presets

### Small Dataset (< 1K documents)

```yaml
collection:
  metadata:
    hnsw:M: 8
    hnsw:construction_ef: 100
    hnsw:search_ef: 50

performance:
  cache_size_mb: 256
  use_mmap: false
```

**Use Cases**:
- Proof of concept
- Development/testing
- Small knowledge bases

### Medium Dataset (1K-10K documents) ⭐ Default

```yaml
collection:
  metadata:
    hnsw:M: 16
    hnsw:construction_ef: 200
    hnsw:search_ef: 100

performance:
  cache_size_mb: 1024
  use_mmap: true
```

**Use Cases**:
- Production agents
- Document search
- RAG applications

### Large Dataset (10K-100K documents)

```yaml
collection:
  metadata:
    hnsw:M: 32
    hnsw:construction_ef: 400
    hnsw:search_ef: 200

performance:
  cache_size_mb: 4096
  use_mmap: true
  prefetch_size: 200
```

**Use Cases**:
- Large knowledge bases
- Enterprise search
- Multi-tenant systems

### X-Large Dataset (> 100K documents)

```yaml
collection:
  metadata:
    hnsw:M: 48
    hnsw:construction_ef: 500
    hnsw:search_ef: 300

performance:
  cache_size_mb: 8192
  use_mmap: true
  prefetch_size: 500

quantization:
  enabled: true
  method: "int8"
```

**Use Cases**:
- Web-scale search
- Massive document collections
- Memory-constrained environments

## 📊 Memory Patterns

### Triple-Layer Retention

```yaml
memory:
  retention:
    short_term:
      enabled: true
      duration_hours: 24
      max_items: 1000

    mid_term:
      enabled: true
      duration_days: 7
      max_items: 5000

    long_term:
      enabled: true
      duration_days: 30
      max_items: 50000
```

**Integration with Memory-MCP**:
- Short-term: Recent context (execution mode)
- Mid-term: Session history (planning mode)
- Long-term: Knowledge base (brainstorming mode)

### Metadata Schema for Agent Memory

```json
{
  "metadata_schema": {
    "fields": [
      {
        "name": "timestamp",
        "type": "datetime",
        "required": true,
        "indexed": true,
        "description": "When memory was created"
      },
      {
        "name": "agent_id",
        "type": "string",
        "required": true,
        "indexed": true,
        "description": "Agent identifier"
      },
      {
        "name": "layer",
        "type": "string",
        "required": true,
        "indexed": true,
        "validation": {
          "enum": ["short_term", "mid_term", "long_term"]
        }
      },
      {
        "name": "priority",
        "type": "integer",
        "required": false,
        "indexed": true,
        "validation": {
          "min": 1,
          "max": 5
        }
      }
    ]
  }
}
```

## 🔧 Performance Tuning

### High Accuracy (95%+ recall)

```yaml
collection:
  metadata:
    hnsw:search_ef: 200

search:
  min_similarity: 0.7
  reranking: "cross_encoder"
```

### High Speed (<1ms latency)

```yaml
collection:
  metadata:
    hnsw:search_ef: 50

performance:
  cache_queries: true
  cache_size_mb: 2048

quantization:
  enabled: true
  method: "int8"
```

### Balanced (Default)

```yaml
collection:
  metadata:
    hnsw:search_ef: 100

search:
  default_k: 10
  min_similarity: 0.5

performance:
  cache_size_mb: 1024
```

## 🔐 Production Checklist

- [ ] HNSW parameters optimized for dataset size
- [ ] Embedding model selected (384d recommended)
- [ ] Metadata schema defined with validation
- [ ] Retention policies configured
- [ ] Monitoring enabled with metrics export
- [ ] Backup schedule configured (24h recommended)
- [ ] Query timeout set (30s default)
- [ ] Cache size allocated (1GB+ recommended)
- [ ] Distance metric validated (cosine for normalized)
- [ ] Integration endpoints configured (Memory-MCP, API)

## 📈 Monitoring Configuration

### Metrics to Track

```yaml
monitoring:
  enabled: true
  track:
    - query_latency      # P50, P95, P99
    - query_throughput   # QPS
    - index_size         # MB
    - memory_usage       # MB
    - cache_hit_rate     # %
    - error_rate         # errors/sec

  export_format: "json"
  export_interval: 60
```

### Integration with Prometheus

```yaml
monitoring:
  export_format: "prometheus"
  api:
    enabled: true
    host: "0.0.0.0"
    port: 9090
```

## 🔗 Integration Examples

### Memory-MCP Triple System

```yaml
integration:
  memory_mcp:
    enabled: true
    endpoint: "http://localhost:3000"

memory:
  retention:
    short_term:
      duration_hours: 24
    mid_term:
      duration_days: 7
    long_term:
      duration_days: 30
```

### RAG Pipeline

```yaml
integration:
  rag:
    enabled: true
    chunk_size: 512
    chunk_overlap: 50

search:
  default_k: 5
  min_similarity: 0.6
  reranking: "semantic"
```

### REST API

```yaml
integration:
  api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    cors_enabled: true

performance:
  query_timeout: 30
  cache_queries: true
```

## 📚 Schema Field Types

### Supported Types

| Type     | Description          | Example                    |
|----------|----------------------|----------------------------|
| string   | Text data            | "agent_memory"             |
| integer  | Whole numbers        | 42                         |
| float    | Decimal numbers      | 3.14                       |
| boolean  | True/False           | true                       |
| datetime | Timestamps           | "2025-11-02T12:00:00Z"     |
| array    | Lists                | ["tag1", "tag2"]           |
| object   | Nested structures    | {"key": "value"}           |

### Validation Rules

```json
{
  "validation": {
    "min": 1,
    "max": 100,
    "pattern": "^[a-zA-Z0-9_]+$",
    "enum": ["option1", "option2", "option3"]
  }
}
```

## 🐛 Troubleshooting

### Issue: High memory usage
**Solution**: Enable quantization or reduce cache size
```yaml
quantization:
  enabled: true
  method: "int8"

performance:
  cache_size_mb: 512
```

### Issue: Slow queries
**Solution**: Increase cache and reduce ef_search
```yaml
collection:
  metadata:
    hnsw:search_ef: 50

performance:
  cache_size_mb: 2048
  cache_queries: true
```

### Issue: Low recall
**Solution**: Increase ef_search and min_similarity
```yaml
collection:
  metadata:
    hnsw:search_ef: 200

search:
  min_similarity: 0.3
```

## 🔗 Related Resources

- See `../scripts/` for optimization tools
- See `../../tests/` for validation tests
- See `../../examples/` for code samples
