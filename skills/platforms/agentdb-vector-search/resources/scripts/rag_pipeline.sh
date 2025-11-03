#!/bin/bash
# RAG Pipeline for AgentDB Vector Search
# Complete Retrieval-Augmented Generation workflow
# Performance: <100µs search + LLM generation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AGENTDB_PATH="${AGENTDB_PATH:-.agentdb/vectors.db}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
DIMENSION="${DIMENSION:-384}"
METRIC="${METRIC:-cosine}"
TOP_K="${TOP_K:-5}"
THRESHOLD="${THRESHOLD:-0.7}"
USE_MMR="${USE_MMR:-true}"
QUANTIZATION="${QUANTIZATION:-binary}"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    local deps=("node" "npx" "python3" "jq")
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        exit 1
    fi

    # Check AgentDB
    if ! npx agentdb@latest --version &> /dev/null; then
        log_error "AgentDB not available. Install with: npm install -g agentdb"
        exit 1
    fi

    log_success "All dependencies available"
}

# Initialize vector database
init_database() {
    log_info "Initializing AgentDB vector database..."

    if [ -f "$AGENTDB_PATH" ]; then
        log_warning "Database already exists: $AGENTDB_PATH"
        read -p "Overwrite? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi

    npx agentdb@latest init "$AGENTDB_PATH" \
        --dimension "$DIMENSION" \
        --preset medium

    log_success "Database initialized: $AGENTDB_PATH"
}

# Ingest documents into vector store
ingest_documents() {
    local docs_dir="${1:-.}"
    log_info "Ingesting documents from: $docs_dir"

    # Find all text/markdown files
    local files=($(find "$docs_dir" -type f \( -name "*.txt" -o -name "*.md" \) 2>/dev/null))

    if [ ${#files[@]} -eq 0 ]; then
        log_warning "No documents found in $docs_dir"
        return
    fi

    log_info "Found ${#files[@]} documents"

    # Process each file
    local count=0
    for file in "${files[@]}"; do
        log_info "Processing: $file"

        # Read content
        local content=$(cat "$file")

        # Generate embedding using Python
        local embedding=$(python3 -c "
from sentence_transformers import SentenceTransformer
import json
model = SentenceTransformer('$EMBEDDING_MODEL')
embedding = model.encode('$content', convert_to_numpy=True)
print(json.dumps(embedding.tolist()))
" 2>/dev/null)

        # Store in AgentDB
        local metadata="{\"file\": \"$file\", \"size\": $(wc -c < "$file")}"

        # Insert using AgentDB API
        npx agentdb@latest insert "$AGENTDB_PATH" \
            --embedding "$embedding" \
            --metadata "$metadata" \
            &> /dev/null

        ((count++))
        echo -ne "\rIngested: $count/${#files[@]}"
    done

    echo
    log_success "Ingested $count documents"
}

# Retrieve relevant context
retrieve_context() {
    local query="$1"
    log_info "Retrieving context for query: $query"

    # Generate query embedding
    local query_embedding=$(python3 -c "
from sentence_transformers import SentenceTransformer
import json
model = SentenceTransformer('$EMBEDDING_MODEL')
embedding = model.encode('$query', convert_to_numpy=True)
print(json.dumps(embedding.tolist()))
" 2>/dev/null)

    # Search vector database
    local results=$(npx agentdb@latest query "$AGENTDB_PATH" \
        "$query_embedding" \
        -k "$TOP_K" \
        -t "$THRESHOLD" \
        -m "$METRIC" \
        -f json 2>/dev/null)

    # Parse and display results
    echo "$results" | jq -r '.[] | "[\(.score | tonumber | . * 100 | round / 100)] \(.metadata.file)"' || true

    # Return formatted context
    echo "$results" | jq -r '.[] | .text' | head -n "$TOP_K"
}

# RAG query with LLM
rag_query() {
    local question="$1"
    log_info "RAG Query: $question"

    # Step 1: Retrieve relevant context
    log_info "Step 1: Retrieving relevant documents..."
    local context=$(retrieve_context "$question")

    if [ -z "$context" ]; then
        log_warning "No relevant context found"
        return 1
    fi

    # Step 2: Build prompt with context
    local prompt="Context:
$context

Question: $question

Answer based on the context above:"

    log_info "Step 2: Generating answer..."

    # Check if Claude API key is available
    if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        # Use Claude API
        local response=$(curl -s https://api.anthropic.com/v1/messages \
            -H "x-api-key: $ANTHROPIC_API_KEY" \
            -H "anthropic-version: 2023-06-01" \
            -H "content-type: application/json" \
            -d "{
                \"model\": \"claude-3-5-sonnet-20241022\",
                \"max_tokens\": 1024,
                \"messages\": [{
                    \"role\": \"user\",
                    \"content\": $(echo "$prompt" | jq -Rs .)
                }]
            }" 2>/dev/null)

        # Extract answer
        local answer=$(echo "$response" | jq -r '.content[0].text' 2>/dev/null || echo "Error generating answer")

        log_success "Answer:"
        echo
        echo "$answer"
    else
        log_warning "ANTHROPIC_API_KEY not set. Showing context only:"
        echo
        echo "$context"
    fi
}

# Benchmark search performance
benchmark_search() {
    log_info "Benchmarking search performance..."

    local queries=(
        "quantum computing"
        "machine learning"
        "database optimization"
        "neural networks"
        "vector embeddings"
    )

    local total_time=0
    local iterations=${#queries[@]}

    for query in "${queries[@]}"; do
        local start=$(date +%s%N)
        retrieve_context "$query" > /dev/null 2>&1
        local end=$(date +%s%N)

        local duration=$(( (end - start) / 1000000 )) # Convert to ms
        total_time=$((total_time + duration))

        log_info "Query '$query': ${duration}ms"
    done

    local avg_time=$((total_time / iterations))
    log_success "Average search time: ${avg_time}ms"

    if [ $avg_time -lt 100 ]; then
        log_success "✅ Sub-100ms performance achieved!"
    else
        log_warning "⚠️ Performance slower than expected (<100ms target)"
    fi
}

# Export vector index
export_index() {
    local output="${1:-vector_index_backup.json}"
    log_info "Exporting vector index to: $output"

    npx agentdb@latest export "$AGENTDB_PATH" "$output"

    local size=$(du -h "$output" | cut -f1)
    log_success "Exported index ($size): $output"
}

# Import vector index
import_index() {
    local input="${1:-vector_index_backup.json}"
    log_info "Importing vector index from: $input"

    if [ ! -f "$input" ]; then
        log_error "File not found: $input"
        exit 1
    fi

    npx agentdb@latest import "$input"
    log_success "Imported index: $input"
}

# Show database statistics
show_stats() {
    log_info "Database Statistics:"
    npx agentdb@latest stats "$AGENTDB_PATH"
}

# Main menu
show_menu() {
    echo
    echo "========================================="
    echo "  AgentDB RAG Pipeline"
    echo "========================================="
    echo "1. Initialize database"
    echo "2. Ingest documents"
    echo "3. Retrieve context"
    echo "4. RAG query (with LLM)"
    echo "5. Benchmark search"
    echo "6. Show statistics"
    echo "7. Export index"
    echo "8. Import index"
    echo "0. Exit"
    echo "========================================="
}

# Main execution
main() {
    check_dependencies

    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option: " choice

            case $choice in
                1)
                    init_database
                    ;;
                2)
                    read -p "Enter documents directory [.]: " docs_dir
                    docs_dir=${docs_dir:-.}
                    ingest_documents "$docs_dir"
                    ;;
                3)
                    read -p "Enter query: " query
                    retrieve_context "$query"
                    ;;
                4)
                    read -p "Enter question: " question
                    rag_query "$question"
                    ;;
                5)
                    benchmark_search
                    ;;
                6)
                    show_stats
                    ;;
                7)
                    read -p "Enter output file [vector_index_backup.json]: " output
                    output=${output:-vector_index_backup.json}
                    export_index "$output"
                    ;;
                8)
                    read -p "Enter input file [vector_index_backup.json]: " input
                    input=${input:-vector_index_backup.json}
                    import_index "$input"
                    ;;
                0)
                    log_info "Exiting..."
                    exit 0
                    ;;
                *)
                    log_error "Invalid option"
                    ;;
            esac
        done
    else
        # CLI mode
        local command="$1"
        shift

        case $command in
            init)
                init_database
                ;;
            ingest)
                ingest_documents "${1:-.}"
                ;;
            retrieve)
                retrieve_context "$@"
                ;;
            query)
                rag_query "$@"
                ;;
            benchmark)
                benchmark_search
                ;;
            stats)
                show_stats
                ;;
            export)
                export_index "${1:-vector_index_backup.json}"
                ;;
            import)
                import_index "${1:-vector_index_backup.json}"
                ;;
            *)
                log_error "Unknown command: $command"
                echo "Usage: $0 {init|ingest|retrieve|query|benchmark|stats|export|import}"
                exit 1
                ;;
        esac
    fi
}

main "$@"
