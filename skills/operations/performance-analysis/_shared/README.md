# Performance Skill - Enhanced Tier

**Status**: Enhanced Tier ✅
**Version**: 2.0.0
**Last Updated**: 2025-01-02

## Overview

Comprehensive performance optimization skill with profiling, bottleneck detection, and automated optimization suggestions. Includes 4 production-ready scripts, 3 configuration templates, 3 test suites, and 3 detailed examples (150-300 lines each).

## Structure

```
performance/
├── resources/                  # Production scripts & templates
│   ├── profiler.py            # Python CPU/memory/I/O profiler
│   ├── bottleneck-detector.js # Node.js bottleneck detection
│   ├── memory-analyzer.sh     # Bash memory analysis
│   ├── optimization-suggester.py # AI optimization recommendations
│   ├── perf-config.yaml       # Performance configuration template
│   ├── benchmark-template.json # Benchmarking template
│   └── optimization-checklist.yaml # Systematic optimization checklist
├── tests/                      # Comprehensive test suites
│   ├── test-profiler.py       # Profiler tests (16 test cases)
│   ├── test-bottleneck-detector.js # Bottleneck detector tests
│   └── test-optimization-suggester.py # Suggester tests
├── examples/                   # Production examples
│   ├── cpu-profiling-example.py (215 lines)
│   ├── memory-optimization-example.py (287 lines)
│   └── latency-reduction-example.js (298 lines)
└── when-*/                     # Existing sub-skills
    ├── when-analyzing-performance-use-performance-analysis/
    └── when-profiling-performance-use-performance-profiler/
```

## Resources (4 Scripts + 3 Templates)

### Scripts

#### 1. `profiler.py` (Python)
**Features**:
- CPU profiling with cProfile and tracemalloc
- Memory profiling with heap snapshots
- I/O profiling (disk, network)
- System resource monitoring
- Process profiling

**Usage**:
```bash
# System monitoring
python profiler.py --mode system --duration 60

# I/O profiling
python profiler.py --mode io --duration 120

# Profile Python script
python profiler.py --target script.py --mode all
```

#### 2. `bottleneck-detector.js` (Node.js)
**Features**:
- Event loop lag detection
- Memory leak detection
- Slow query tracking
- Slow request monitoring
- Async operation profiling
- Automated recommendations

**Usage**:
```javascript
const BottleneckDetector = require('./bottleneck-detector');

const detector = new BottleneckDetector({
  eventLoopThreshold: 100,
  memoryLeakThreshold: 10,
  slowQueryThreshold: 100
});

detector.on('bottleneck', (bottleneck) => {
  console.log(`[${bottleneck.severity}] ${bottleneck.message}`);
});

detector.start();
```

#### 3. `memory-analyzer.sh` (Bash)
**Features**:
- System memory analysis
- Process memory tracking
- Memory growth detection
- OOM killer history
- Swap usage analysis
- Memory leak detection

**Usage**:
```bash
# Full analysis
./memory-analyzer.sh full

# Track process
./memory-analyzer.sh track <PID> 300

# Check for leaks
./memory-analyzer.sh leaks
```

#### 4. `optimization-suggester.py` (Python)
**Features**:
- AI-powered optimization recommendations
- CPU, memory, I/O, database analysis
- Severity prioritization
- Code examples for fixes
- Estimated improvement calculations

**Usage**:
```bash
# Generate suggestions from profile
python optimization-suggester.py profile_data.json --output suggestions.json

# View recommendations
python optimization-suggester.py profile_data.json --format text
```

### Templates

#### 1. `perf-config.yaml`
Comprehensive configuration for:
- Profiling modes (quick, standard, deep, continuous)
- Threshold settings (CPU, memory, I/O, event loop)
- Bottleneck detection rules
- Output formats
- Integration settings (APM tools, notifications)

#### 2. `benchmark-template.json`
Standardized benchmarking with:
- Multiple test scenarios
- Baseline vs optimized comparisons
- Stress testing configuration
- Metrics definitions
- Reporting templates

#### 3. `optimization-checklist.yaml`
Systematic 10-phase optimization checklist:
1. Baseline & Measurement
2. Analysis & Bottleneck Identification
3. Algorithmic Optimization
4. Memory Optimization
5. Caching Strategies
6. Database Optimization
7. Parallelization & Concurrency
8. Network Optimization
9. Validation & Benchmarking
10. Documentation & Maintenance

## Tests (3 Comprehensive Suites)

### 1. `test-profiler.py`
**Coverage**:
- CPU profiling accuracy
- Memory profiling with tracemalloc
- I/O metrics collection
- System monitoring
- Report generation
- Edge cases (fast functions, exceptions)

**Run**: `python test-profiler.py`

### 2. `test-bottleneck-detector.js`
**Coverage**:
- Event loop lag detection
- Memory growth tracking
- Query tracking
- Request monitoring
- Analysis and recommendations
- Report generation

**Run**: `npm test` or `mocha test-bottleneck-detector.js`

### 3. `test-optimization-suggester.py`
**Coverage**:
- CPU profile analysis
- Memory hotspot detection
- I/O optimization suggestions
- Bottleneck analysis
- Severity sorting
- Report generation

**Run**: `python test-optimization-suggester.py`

## Examples (3 Production-Ready Demos)

### 1. CPU Profiling (215 lines)
**File**: `cpu-profiling-example.py`

**Demonstrates**:
- Inefficient vs optimized algorithms
- Bubble Sort O(n²) → Quick Sort O(n log n)
- Recursive Fibonacci O(2^n) → Memoized O(n)
- Linear Search O(n) → Binary Search O(log n)

**Results**:
- Sorting: 98%+ improvement, 50-100x speedup
- Fibonacci: 99%+ improvement, 1000x+ speedup
- Search: 95%+ improvement, 20-100x speedup

**Run**: `python cpu-profiling-example.py`

### 2. Memory Optimization (287 lines)
**File**: `memory-optimization-example.py`

**Demonstrates**:
- Memory leak detection and fixing
- Generator vs list for large datasets
- `__slots__` for reduced memory
- String interning for duplicates
- Streaming file processing

**Results**:
- File loading: 95%+ memory reduction
- Data structures: 40-60% memory reduction with `__slots__`
- Leak detection: Automated circular reference breaking

**Run**: `python memory-optimization-example.py`

### 3. Latency Reduction (298 lines)
**File**: `latency-reduction-example.js`

**Demonstrates**:
- Sequential → Parallel fetching
- Synchronous → Async processing
- N+1 queries → Batched queries
- No cache → LRU caching

**Results**:
- Parallelization: 90%+ latency reduction, 10x speedup
- Async processing: 20-40% reduction
- Batching: 85%+ reduction, 60-80x speedup
- Caching: 50%+ reduction with 50% hit rate

**Run**: `node latency-reduction-example.js`

## Quick Start

### 1. Profile Application
```bash
# CPU profiling
python resources/profiler.py --mode cpu --target app.py

# Memory profiling
python resources/profiler.py --mode memory --target app.py

# Full profiling
python resources/profiler.py --mode all --target app.py --duration 300
```

### 2. Detect Bottlenecks
```javascript
// In your Node.js app
const BottleneckDetector = require('./resources/bottleneck-detector');
const detector = new BottleneckDetector();

detector.on('bottleneck', (b) => {
  if (b.severity === 'high') {
    console.error(`BOTTLENECK: ${b.message}`);
  }
});

detector.start();
```

### 3. Analyze Memory
```bash
# Monitor system memory
./resources/memory-analyzer.sh system

# Track process memory growth
./resources/memory-analyzer.sh track <PID> 600

# Generate full report
./resources/memory-analyzer.sh full
```

### 4. Get Optimization Suggestions
```bash
# From profiling data
python resources/optimization-suggester.py \
  performance-profile.json \
  --output suggestions.json

# View prioritized recommendations
cat suggestions.json | jq '.suggestions[] | select(.severity == "high")'
```

## Integration with SPARC Skills

### Performance Analysis Skill
Located: `when-analyzing-performance-use-performance-analysis/`

Use for:
- Swarm-level performance analysis
- Bottleneck detection across agents
- Coordination overhead measurement
- Topology optimization

### Performance Profiler Skill
Located: `when-profiling-performance-use-performance-profiler/`

Use for:
- Multi-dimensional profiling (CPU, memory, I/O, network)
- Flame graph generation
- Agent-level optimization
- Production profiling

## Best Practices

1. **Profile First**: Always measure before optimizing
2. **Focus on Bottlenecks**: 80/20 rule - optimize the 20% that matters
3. **Validate Improvements**: Benchmark before and after
4. **Maintain Correctness**: Tests must pass after optimization
5. **Document Changes**: Record all optimizations and their impact
6. **Monitor Production**: Track performance over time
7. **Set Budgets**: Define performance targets (latency, memory, CPU)

## Performance Targets

### API/Backend
- **Latency**: P50 < 100ms, P95 < 500ms, P99 < 1000ms
- **Throughput**: > 1000 req/s
- **Error Rate**: < 0.1%
- **CPU Usage**: < 70%
- **Memory Usage**: < 80%

### Database
- **Query Time**: P50 < 10ms, P95 < 50ms, P99 < 100ms
- **Connection Pool**: < 80% utilization

### Frontend
- **TTFB**: < 200ms
- **FCP**: < 1.8s
- **LCP**: < 2.5s
- **TTI**: < 3.8s
- **CLS**: < 0.1

## Troubleshooting

### High CPU Usage
1. Run CPU profiler
2. Identify hot paths (>10% cumulative time)
3. Apply algorithmic optimizations
4. Consider parallelization
5. Benchmark improvements

### Memory Leaks
1. Run memory profiler
2. Take heap snapshots over time
3. Identify growing allocations
4. Break circular references
5. Use weak references where appropriate

### High Latency
1. Run bottleneck detector
2. Profile async operations
3. Parallelize independent tasks
4. Implement caching
5. Batch database queries

## License

MIT - Part of SPARC Three-Loop System

## Contributing

See parent skill documentation for contribution guidelines.
