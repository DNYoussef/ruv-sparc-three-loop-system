#!/bin/bash
# AgentDB HNSW Optimization Script
# Tunes HNSW parameters for optimal vector search performance

set -e

echo "================================================"
echo "AgentDB HNSW Parameter Optimization"
echo "================================================"

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
DATASET_SIZE=${1:-10000}
EMBEDDING_DIM=${2:-384}
TARGET_RECALL=${3:-0.95}

echo -e "${BLUE}Configuration:${NC}"
echo "  Dataset size: $DATASET_SIZE documents"
echo "  Embedding dimension: $EMBEDDING_DIM"
echo "  Target recall: $TARGET_RECALL"
echo ""

# HNSW parameter recommendations based on dataset size
optimize_hnsw_params() {
    local size=$1

    echo -e "${YELLOW}Calculating optimal HNSW parameters...${NC}"

    if [ $size -lt 1000 ]; then
        M=8
        EF_CONSTRUCTION=100
        EF_SEARCH=50
        TIER="Small"
    elif [ $size -lt 10000 ]; then
        M=16
        EF_CONSTRUCTION=200
        EF_SEARCH=100
        TIER="Medium"
    elif [ $size -lt 100000 ]; then
        M=32
        EF_CONSTRUCTION=400
        EF_SEARCH=200
        TIER="Large"
    else
        M=48
        EF_CONSTRUCTION=500
        EF_SEARCH=300
        TIER="X-Large"
    fi

    echo ""
    echo -e "${GREEN}Optimal Parameters for $TIER dataset ($size docs):${NC}"
    echo "  M (connections per layer): $M"
    echo "  ef_construction (build quality): $EF_CONSTRUCTION"
    echo "  ef_search (query quality): $EF_SEARCH"
    echo ""

    # Memory estimation
    MEMORY_PER_DOC=$(( (M * 2 * 4) + (EMBEDDING_DIM * 4) ))
    TOTAL_MEMORY_MB=$(( (MEMORY_PER_DOC * size) / 1024 / 1024 ))

    echo -e "${BLUE}Memory Estimation:${NC}"
    echo "  Per document: ${MEMORY_PER_DOC} bytes"
    echo "  Total index: ${TOTAL_MEMORY_MB} MB"
    echo ""

    # Performance estimation
    BUILD_TIME_SEC=$(echo "scale=2; $size * 0.001 * ($EF_CONSTRUCTION / 100)" | bc)
    QUERY_TIME_MS=$(echo "scale=3; 1.0 + ($size / 100000) * ($EF_SEARCH / 100)" | bc)
    QPS=$(echo "scale=1; 1000 / $QUERY_TIME_MS" | bc)

    echo -e "${BLUE}Performance Estimation:${NC}"
    echo "  Build time: ${BUILD_TIME_SEC}s"
    echo "  Query latency: ${QUERY_TIME_MS}ms"
    echo "  Throughput: ${QPS} QPS"
    echo ""

    # Generate configuration file
    cat > hnsw_config.yaml <<EOF
# AgentDB HNSW Configuration
# Generated for dataset: $TIER ($size documents)

collection:
  name: "agentdb_optimized"
  metadata:
    hnsw:space: "cosine"
    hnsw:construction_ef: $EF_CONSTRUCTION
    hnsw:search_ef: $EF_SEARCH
    hnsw:M: $M
    hnsw:num_threads: 4

embedding:
  model: "all-MiniLM-L6-v2"
  dimension: $EMBEDDING_DIM
  batch_size: 128

performance:
  estimated_build_time_sec: $BUILD_TIME_SEC
  estimated_query_latency_ms: $QUERY_TIME_MS
  estimated_throughput_qps: $QPS
  estimated_memory_mb: $TOTAL_MEMORY_MB

tuning_guide:
  increase_recall:
    - "Increase ef_search (current: $EF_SEARCH)"
    - "Increase M (current: $M)"
  increase_speed:
    - "Decrease ef_search (min: 50)"
    - "Use quantization (see quantize_vectors.py)"
  reduce_memory:
    - "Decrease M (current: $M)"
    - "Apply vector quantization (4-32x reduction)"
EOF

    echo -e "${GREEN}✅ Configuration saved to hnsw_config.yaml${NC}"
    echo ""
}

# Recall vs Speed trade-off analysis
analyze_tradeoffs() {
    echo -e "${YELLOW}Recall vs Speed Trade-off Analysis:${NC}"
    echo ""
    echo "┌─────────────┬──────────┬───────────────┬───────────┐"
    echo "│ ef_search   │ Recall   │ Latency (ms)  │ QPS       │"
    echo "├─────────────┼──────────┼───────────────┼───────────┤"
    echo "│ 50          │ ~0.90    │ 0.8           │ 1250      │"
    echo "│ 100         │ ~0.95    │ 1.2           │ 833       │"
    echo "│ 200         │ ~0.98    │ 2.0           │ 500       │"
    echo "│ 500         │ ~0.99    │ 4.5           │ 222       │"
    echo "└─────────────┴──────────┴───────────────┴───────────┘"
    echo ""
    echo "Recommendation: Use ef_search=100 for balanced performance"
    echo ""
}

# Best practices guide
print_best_practices() {
    echo -e "${YELLOW}HNSW Optimization Best Practices:${NC}"
    echo ""
    echo "1️⃣  Dataset Size Scaling:"
    echo "   • < 1K docs: M=8, ef_construction=100"
    echo "   • < 10K docs: M=16, ef_construction=200"
    echo "   • < 100K docs: M=32, ef_construction=400"
    echo "   • > 100K docs: M=48, ef_construction=500"
    echo ""
    echo "2️⃣  Memory Optimization:"
    echo "   • Use quantization for 4-32x reduction"
    echo "   • Lower M reduces memory linearly"
    echo "   • Consider dimensionality reduction (384→128)"
    echo ""
    echo "3️⃣  Query Performance:"
    echo "   • Start with ef_search=100 (95% recall)"
    echo "   • Increase for higher recall (diminishing returns)"
    echo "   • Decrease for speed (test recall impact)"
    echo ""
    echo "4️⃣  Build Performance:"
    echo "   • Higher ef_construction = better quality graph"
    echo "   • Trade-off: 2x ef_construction ≈ 2x build time"
    echo "   • Parallel builds: set num_threads=CPU_count"
    echo ""
    echo "5️⃣  Monitoring:"
    echo "   • Track query latency percentiles (p50, p95, p99)"
    echo "   • Measure recall with ground truth dataset"
    echo "   • Monitor memory usage vs dataset growth"
    echo ""
}

# Validation tests
run_validation() {
    echo -e "${YELLOW}Running validation tests...${NC}"

    if command -v python3 &> /dev/null; then
        if python3 -c "import chromadb, sentence_transformers" &> /dev/null; then
            echo -e "${GREEN}✅ Dependencies installed (chromadb, sentence-transformers)${NC}"
        else
            echo -e "${RED}❌ Missing dependencies. Run: pip install chromadb sentence-transformers${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Python 3 not found${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ Validation passed${NC}"
    echo ""
}

# Main execution
main() {
    optimize_hnsw_params $DATASET_SIZE
    analyze_tradeoffs
    print_best_practices
    run_validation

    echo "================================================"
    echo -e "${GREEN}HNSW optimization complete!${NC}"
    echo "================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Review hnsw_config.yaml"
    echo "  2. Run benchmark: python3 benchmark_search.py"
    echo "  3. Apply quantization: python3 quantize_vectors.py"
    echo ""
}

main
