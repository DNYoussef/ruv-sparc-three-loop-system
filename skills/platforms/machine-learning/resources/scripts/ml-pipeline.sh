#!/bin/bash
###############################################################################
# Complete ML Pipeline Script
# End-to-end workflow from data preprocessing to model deployment
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
DATA_PATH=""
CONFIG_PATH="resources/templates/training-config.yaml"
OUTPUT_DIR="output"
EXPERIMENT_NAME="ml_experiment_$(date +%Y%m%d_%H%M%S)"
SKIP_PREPROCESSING=false
SKIP_TRAINING=false
SKIP_EVALUATION=false
DEPLOY=false

###############################################################################
# Functions
###############################################################################

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Complete ML pipeline: preprocessing -> training -> evaluation -> deployment

OPTIONS:
    -d, --data PATH          Path to raw data file (required)
    -c, --config PATH        Path to training config (default: resources/templates/training-config.yaml)
    -o, --output DIR         Output directory (default: output)
    -n, --name NAME          Experiment name (default: ml_experiment_TIMESTAMP)
    --skip-preprocessing     Skip data preprocessing step
    --skip-training          Skip model training step
    --skip-evaluation        Skip model evaluation step
    --deploy                 Deploy model after successful evaluation
    -h, --help               Show this help message

EXAMPLES:
    # Full pipeline
    $0 --data data/raw/dataset.csv --output results/

    # Skip preprocessing (use existing processed data)
    $0 --data data/processed/train.csv --skip-preprocessing

    # Train and deploy
    $0 --data data/raw/dataset.csv --deploy

EOF
}

check_dependencies() {
    log_info "Checking dependencies..."

    local missing_deps=()

    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi

    # Check required Python packages
    if ! python3 -c "import torch" 2>/dev/null; then
        missing_deps+=("pytorch")
    fi

    if ! python3 -c "import pandas" 2>/dev/null; then
        missing_deps+=("pandas")
    fi

    if ! python3 -c "import sklearn" 2>/dev/null; then
        missing_deps+=("scikit-learn")
    fi

    # Check Node.js for evaluation
    if ! command -v node &> /dev/null; then
        missing_deps+=("node.js")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install with: pip install torch pandas scikit-learn"
        exit 1
    fi

    log_info "All dependencies satisfied"
}

setup_environment() {
    log_info "Setting up environment for experiment: $EXPERIMENT_NAME"

    # Create directory structure
    mkdir -p "$OUTPUT_DIR/$EXPERIMENT_NAME"/{models,logs,metrics,checkpoints}

    # Setup Python path
    export PYTHONPATH="${PYTHONPATH:-.}:$(pwd)"

    # Create experiment config
    cat > "$OUTPUT_DIR/$EXPERIMENT_NAME/config.json" << EOF
{
    "experiment_name": "$EXPERIMENT_NAME",
    "timestamp": "$(date -Iseconds)",
    "data_path": "$DATA_PATH",
    "config_path": "$CONFIG_PATH",
    "output_dir": "$OUTPUT_DIR/$EXPERIMENT_NAME"
}
EOF

    log_info "Environment setup complete"
}

preprocess_data() {
    if [ "$SKIP_PREPROCESSING" = true ]; then
        log_warn "Skipping data preprocessing"
        return 0
    fi

    log_info "Starting data preprocessing..."

    python3 resources/scripts/data-preprocessor.py \
        --input "$DATA_PATH" \
        --output "$OUTPUT_DIR/$EXPERIMENT_NAME/processed_data" \
        --config "$CONFIG_PATH" \
        2>&1 | tee "$OUTPUT_DIR/$EXPERIMENT_NAME/logs/preprocessing.log"

    if [ $? -eq 0 ]; then
        log_info "Data preprocessing completed successfully"
    else
        log_error "Data preprocessing failed"
        exit 1
    fi
}

train_model() {
    if [ "$SKIP_TRAINING" = true ]; then
        log_warn "Skipping model training"
        return 0
    fi

    log_info "Starting model training..."

    # Check for GPU
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        log_info "GPU available - using CUDA"
        export CUDA_VISIBLE_DEVICES=0
    else
        log_warn "No GPU detected - training on CPU"
    fi

    python3 resources/scripts/model-trainer.py \
        --config "$CONFIG_PATH" \
        --data "$OUTPUT_DIR/$EXPERIMENT_NAME/processed_data" \
        --output "$OUTPUT_DIR/$EXPERIMENT_NAME/models" \
        --checkpoint-dir "$OUTPUT_DIR/$EXPERIMENT_NAME/checkpoints" \
        2>&1 | tee "$OUTPUT_DIR/$EXPERIMENT_NAME/logs/training.log"

    if [ $? -eq 0 ]; then
        log_info "Model training completed successfully"
    else
        log_error "Model training failed"
        exit 1
    fi
}

evaluate_model() {
    if [ "$SKIP_EVALUATION" = true ]; then
        log_warn "Skipping model evaluation"
        return 0
    fi

    log_info "Starting model evaluation..."

    # Find best model checkpoint
    BEST_MODEL=$(find "$OUTPUT_DIR/$EXPERIMENT_NAME/models" -name "best_model.pth" | head -1)

    if [ -z "$BEST_MODEL" ]; then
        log_error "No trained model found"
        exit 1
    fi

    log_info "Evaluating model: $BEST_MODEL"

    # Run evaluation
    node resources/scripts/model-evaluator.js \
        --model "$BEST_MODEL" \
        --data "$OUTPUT_DIR/$EXPERIMENT_NAME/processed_data/test.csv" \
        --output "$OUTPUT_DIR/$EXPERIMENT_NAME/metrics" \
        2>&1 | tee "$OUTPUT_DIR/$EXPERIMENT_NAME/logs/evaluation.log"

    if [ $? -eq 0 ]; then
        log_info "Model evaluation completed successfully"
    else
        log_error "Model evaluation failed"
        exit 1
    fi
}

deploy_model() {
    if [ "$DEPLOY" = false ]; then
        log_info "Deployment not requested - skipping"
        return 0
    fi

    log_info "Deploying model..."

    # Create deployment package
    DEPLOY_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME/deployment"
    mkdir -p "$DEPLOY_DIR"

    # Copy model
    cp "$OUTPUT_DIR/$EXPERIMENT_NAME/models/best_model.pth" "$DEPLOY_DIR/"

    # Copy preprocessor
    if [ -f "$OUTPUT_DIR/$EXPERIMENT_NAME/processed_data/preprocessor.pkl" ]; then
        cp "$OUTPUT_DIR/$EXPERIMENT_NAME/processed_data/preprocessor.pkl" "$DEPLOY_DIR/"
    fi

    # Create deployment config
    cat > "$DEPLOY_DIR/deployment_config.json" << EOF
{
    "model_path": "best_model.pth",
    "preprocessor_path": "preprocessor.pkl",
    "timestamp": "$(date -Iseconds)",
    "experiment_name": "$EXPERIMENT_NAME"
}
EOF

    # Create Dockerfile
    cat > "$DEPLOY_DIR/Dockerfile" << 'EOF'
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    log_info "Deployment package created: $DEPLOY_DIR"
    log_info "Build with: docker build -t ml-model $DEPLOY_DIR"
    log_info "Run with: docker run -p 8000:8000 ml-model"
}

generate_report() {
    log_info "Generating final report..."

    REPORT_FILE="$OUTPUT_DIR/$EXPERIMENT_NAME/REPORT.md"

    cat > "$REPORT_FILE" << EOF
# ML Pipeline Report

**Experiment**: $EXPERIMENT_NAME
**Date**: $(date)

## Configuration

- **Data**: $DATA_PATH
- **Config**: $CONFIG_PATH
- **Output**: $OUTPUT_DIR/$EXPERIMENT_NAME

## Pipeline Stages

EOF

    if [ "$SKIP_PREPROCESSING" = false ]; then
        echo "- [x] Data Preprocessing" >> "$REPORT_FILE"
    else
        echo "- [ ] Data Preprocessing (skipped)" >> "$REPORT_FILE"
    fi

    if [ "$SKIP_TRAINING" = false ]; then
        echo "- [x] Model Training" >> "$REPORT_FILE"
    else
        echo "- [ ] Model Training (skipped)" >> "$REPORT_FILE"
    fi

    if [ "$SKIP_EVALUATION" = false ]; then
        echo "- [x] Model Evaluation" >> "$REPORT_FILE"
    else
        echo "- [ ] Model Evaluation (skipped)" >> "$REPORT_FILE"
    fi

    if [ "$DEPLOY" = true ]; then
        echo "- [x] Model Deployment" >> "$REPORT_FILE"
    else
        echo "- [ ] Model Deployment (skipped)" >> "$REPORT_FILE"
    fi

    cat >> "$REPORT_FILE" << EOF

## Results

See detailed results in:
- Preprocessing logs: \`logs/preprocessing.log\`
- Training logs: \`logs/training.log\`
- Evaluation metrics: \`metrics/\`
- Model checkpoints: \`checkpoints/\`

## Next Steps

1. Review evaluation metrics
2. Analyze model performance
3. Consider hyperparameter tuning
4. Deploy to production if satisfactory

EOF

    log_info "Report generated: $REPORT_FILE"
}

###############################################################################
# Main Pipeline
###############################################################################

main() {
    log_info "=== ML Pipeline Starting ==="

    # Check dependencies
    check_dependencies

    # Setup environment
    setup_environment

    # Run pipeline stages
    preprocess_data
    train_model
    evaluate_model
    deploy_model

    # Generate report
    generate_report

    log_info "=== ML Pipeline Completed Successfully ==="
    log_info "Results saved to: $OUTPUT_DIR/$EXPERIMENT_NAME"
}

###############################################################################
# Parse Arguments
###############################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --skip-preprocessing)
            SKIP_PREPROCESSING=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            shift
            ;;
        --deploy)
            DEPLOY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATA_PATH" ]; then
    log_error "Data path is required"
    usage
    exit 1
fi

# Run main pipeline
main
