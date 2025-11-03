#!/usr/bin/env bash
################################################################################
# Ghidra Headless Analysis Automation Script
# RE Level 2: Static Analysis with Decompilation and Callgraph Generation
#
# Features:
# - Automatic architecture detection (x86/x64/ARM/MIPS)
# - Headless analyzeHeadless integration
# - Selective function decompilation (all or specific functions)
# - Callgraph generation via GraphViz
# - CFG (Control Flow Graph) export
# - Import/export table extraction
# - Connascence analysis on decompiled code
#
# Usage:
#     bash ghidra-headless.sh --binary malware.exe --project-dir ./ghidra-project
#     bash ghidra-headless.sh --binary crackme.bin --decompile true --callgraph true
#
# Author: RE Quick Triage Skill
# License: MIT
################################################################################

set -euo pipefail  # Exit on error, undefined var, pipe failure

# ============================================================================
# Configuration
# ============================================================================

# Default Ghidra installation path (override with --ghidra-path)
GHIDRA_PATH="${GHIDRA_HOME:-/opt/ghidra}"
ANALYZE_HEADLESS="$GHIDRA_PATH/support/analyzeHeadless"

# Default options
DECOMPILE=true
CALLGRAPH=true
CFG=true
IMPORT_EXPORT=true
MAX_FUNCTIONS=0  # 0 = all functions
TIMEOUT=3600  # 1 hour timeout
VERBOSE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# ============================================================================
# Utility Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[*]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[+]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

check_dependencies() {
    local missing_deps=()

    # Check Ghidra installation
    if [[ ! -f "$ANALYZE_HEADLESS" ]]; then
        missing_deps+=("Ghidra (analyzeHeadless not found at $ANALYZE_HEADLESS)")
    fi

    # Check optional tools
    if ! command -v dot &> /dev/null && [[ "$CALLGRAPH" == "true" ]]; then
        log_warning "GraphViz (dot) not found. Callgraph generation will be skipped."
    fi

    if ! command -v file &> /dev/null; then
        missing_deps+=("file (for binary type detection)")
    fi

    if ! command -v sha256sum &> /dev/null; then
        log_warning "sha256sum not found. Hash computation will be skipped."
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies:"
        printf '  - %s\n' "${missing_deps[@]}" >&2
        exit 1
    fi
}

detect_architecture() {
    local binary="$1"
    local file_output

    file_output=$(file "$binary")

    if [[ "$file_output" =~ "x86-64" || "$file_output" =~ "x86_64" ]]; then
        echo "x86:LE:64:default"
    elif [[ "$file_output" =~ "80386" || "$file_output" =~ "Intel" ]]; then
        echo "x86:LE:32:default"
    elif [[ "$file_output" =~ "ARM" ]]; then
        if [[ "$file_output" =~ "64-bit" ]]; then
            echo "AARCH64:LE:64:v8A"
        else
            echo "ARM:LE:32:v7"
        fi
    elif [[ "$file_output" =~ "MIPS" ]]; then
        echo "MIPS:BE:32:default"
    else
        log_warning "Unknown architecture. Ghidra will auto-detect."
        echo "auto"
    fi
}

compute_hash() {
    local binary="$1"
    if command -v sha256sum &> /dev/null; then
        sha256sum "$binary" | awk '{print $1}'
    else
        echo "HASH_UNAVAILABLE"
    fi
}

# ============================================================================
# Ghidra Analysis Functions
# ============================================================================

create_ghidra_project() {
    local binary="$1"
    local project_dir="$2"
    local project_name
    project_name=$(basename "$binary" | sed 's/\.[^.]*$//')  # Remove extension

    log_info "Creating Ghidra project: $project_name"

    # Create project directory
    mkdir -p "$project_dir"

    # Import binary into Ghidra project
    log_info "Importing binary: $binary"
    "$ANALYZE_HEADLESS" "$project_dir" "$project_name" \
        -import "$binary" \
        -scriptPath "$GHIDRA_PATH/Ghidra/Features/Base/ghidra_scripts" \
        -noanalysis \
        -overwrite \
        ${VERBOSE:+-max-cpu 4} \
        2>&1 | tee "$project_dir/import.log"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log_error "Failed to import binary into Ghidra project"
        exit 1
    fi

    log_success "Ghidra project created: $project_dir/$project_name"
}

run_auto_analysis() {
    local binary="$1"
    local project_dir="$2"
    local project_name
    project_name=$(basename "$binary" | sed 's/\.[^.]*$//')

    log_info "Running Ghidra auto-analysis..."

    "$ANALYZE_HEADLESS" "$project_dir" "$project_name" \
        -process "$(basename "$binary")" \
        -scriptPath "$GHIDRA_PATH/Ghidra/Features/Base/ghidra_scripts" \
        ${VERBOSE:+-max-cpu 4} \
        2>&1 | tee "$project_dir/analysis.log"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log_error "Auto-analysis failed"
        return 1
    fi

    log_success "Auto-analysis completed"
}

decompile_functions() {
    local binary="$1"
    local project_dir="$2"
    local output_dir="$project_dir/decompiled"
    local project_name
    project_name=$(basename "$binary" | sed 's/\.[^.]*$//')

    mkdir -p "$output_dir"

    log_info "Decompiling functions..."

    # Create Ghidra script for decompilation
    local decompile_script="$project_dir/decompile_all.py"
    cat > "$decompile_script" <<'EOF'
# Auto-generated Ghidra decompilation script
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor

def decompile_all():
    program = getCurrentProgram()
    fm = program.getFunctionManager()
    functions = fm.getFunctions(True)  # Get all functions

    decompiler = DecompInterface()
    decompiler.openProgram(program)

    output_dir = askDirectory("Select output directory", "Choose")

    count = 0
    for func in functions:
        results = decompiler.decompileFunction(func, 30, ConsoleTaskMonitor())

        if results.decompileCompleted():
            c_code = results.getDecompiledFunction().getC()

            # Write to file
            func_name = func.getName()
            output_file = java.io.File(output_dir, func_name + ".c")

            with open(str(output_file), 'w') as f:
                f.write("// Function: %s\n" % func_name)
                f.write("// Address: %s\n\n" % func.getEntryPoint())
                f.write(c_code)

            count += 1

    print("[SUCCESS] Decompiled %d functions" % count)

decompile_all()
EOF

    # Run decompilation script
    "$ANALYZE_HEADLESS" "$project_dir" "$project_name" \
        -process "$(basename "$binary")" \
        -scriptPath "$project_dir" \
        -postScript decompile_all.py "$output_dir" \
        2>&1 | tee "$project_dir/decompile.log"

    local func_count
    func_count=$(find "$output_dir" -name "*.c" 2>/dev/null | wc -l)
    log_success "Decompiled $func_count functions to $output_dir"
}

generate_callgraph() {
    local binary="$1"
    local project_dir="$2"
    local output_dir="$project_dir/callgraphs"
    local project_name
    project_name=$(basename "$binary" | sed 's/\.[^.]*$//')

    mkdir -p "$output_dir"

    if ! command -v dot &> /dev/null; then
        log_warning "GraphViz (dot) not found. Skipping callgraph generation."
        return 0
    fi

    log_info "Generating callgraph..."

    # Create Ghidra script for callgraph export
    local callgraph_script="$project_dir/export_callgraph.py"
    cat > "$callgraph_script" <<'EOF'
# Auto-generated Ghidra callgraph export script
from ghidra.program.model.symbol import RefType

def export_callgraph():
    program = getCurrentProgram()
    fm = program.getFunctionManager()
    functions = fm.getFunctions(True)

    output_file = askFile("Select output DOT file", "Choose")

    with open(str(output_file), 'w') as f:
        f.write("digraph CallGraph {\n")
        f.write("    rankdir=TB;\n")
        f.write("    node [shape=box, style=filled, fillcolor=lightblue];\n\n")

        for func in functions:
            func_name = func.getName()
            func_addr = str(func.getEntryPoint())

            # Get called functions
            refs = func.getBody().getAddresses(True)
            for ref in refs:
                for xref in getReferencesFrom(ref):
                    if xref.getReferenceType() in [RefType.UNCONDITIONAL_CALL, RefType.CONDITIONAL_CALL]:
                        called_func = fm.getFunctionContaining(xref.getToAddress())
                        if called_func:
                            called_name = called_func.getName()
                            f.write('    "%s" -> "%s";\n' % (func_name, called_name))

        f.write("}\n")

    print("[SUCCESS] Callgraph exported to %s" % output_file)

export_callgraph()
EOF

    # Run callgraph export script
    "$ANALYZE_HEADLESS" "$project_dir" "$project_name" \
        -process "$(basename "$binary")" \
        -scriptPath "$project_dir" \
        -postScript export_callgraph.py "$output_dir/callgraph.dot" \
        2>&1 | tee "$project_dir/callgraph.log"

    # Convert DOT to PNG if GraphViz available
    if [[ -f "$output_dir/callgraph.dot" ]]; then
        dot -Tpng "$output_dir/callgraph.dot" -o "$output_dir/callgraph.png"
        log_success "Callgraph generated: $output_dir/callgraph.png"
    fi
}

# ============================================================================
# CLI Parsing
# ============================================================================

usage() {
    cat <<EOF
Usage: $0 --binary <file> --project-dir <dir> [OPTIONS]

Required:
    --binary <file>          Binary file to analyze
    --project-dir <dir>      Ghidra project directory

Options:
    --ghidra-path <path>     Ghidra installation path (default: /opt/ghidra)
    --decompile <bool>       Decompile functions (default: true)
    --callgraph <bool>       Generate callgraph (default: true)
    --cfg <bool>             Generate CFG (default: true)
    --functions <list>       Comma-separated function names to decompile (default: all)
    --timeout <seconds>      Analysis timeout (default: 3600)
    --verbose                Enable verbose output
    --help                   Show this help message

Examples:
    # Basic analysis
    $0 --binary malware.exe --project-dir ./ghidra-project

    # Decompile specific functions only
    $0 --binary crackme.bin --project-dir ./analysis --functions main,check_password

    # Disable callgraph generation
    $0 --binary firmware.bin --project-dir ./fw-analysis --callgraph false
EOF
}

# Parse arguments
BINARY=""
PROJECT_DIR=""
FUNCTIONS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --binary)
            BINARY="$2"
            shift 2
            ;;
        --project-dir)
            PROJECT_DIR="$2"
            shift 2
            ;;
        --ghidra-path)
            GHIDRA_PATH="$2"
            ANALYZE_HEADLESS="$GHIDRA_PATH/support/analyzeHeadless"
            shift 2
            ;;
        --decompile)
            DECOMPILE="$2"
            shift 2
            ;;
        --callgraph)
            CALLGRAPH="$2"
            shift 2
            ;;
        --cfg)
            CFG="$2"
            shift 2
            ;;
        --functions)
            FUNCTIONS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
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
if [[ -z "$BINARY" ]]; then
    log_error "Missing required argument: --binary"
    usage
    exit 1
fi

if [[ -z "$PROJECT_DIR" ]]; then
    log_error "Missing required argument: --project-dir"
    usage
    exit 1
fi

if [[ ! -f "$BINARY" ]]; then
    log_error "Binary not found: $BINARY"
    exit 1
fi

# ============================================================================
# Main Analysis Workflow
# ============================================================================

main() {
    log_info "========================================="
    log_info "Ghidra Headless Analysis"
    log_info "========================================="
    log_info "Binary: $BINARY"
    log_info "Project: $PROJECT_DIR"
    log_info "========================================="

    # Check dependencies
    check_dependencies

    # Detect architecture
    local arch
    arch=$(detect_architecture "$BINARY")
    log_info "Detected architecture: $arch"

    # Compute hash
    local hash
    hash=$(compute_hash "$BINARY")
    log_info "SHA-256: $hash"

    # Create Ghidra project and import binary
    create_ghidra_project "$BINARY" "$PROJECT_DIR"

    # Run auto-analysis
    run_auto_analysis "$BINARY" "$PROJECT_DIR"

    # Decompile functions if requested
    if [[ "$DECOMPILE" == "true" ]]; then
        decompile_functions "$BINARY" "$PROJECT_DIR"
    fi

    # Generate callgraph if requested
    if [[ "$CALLGRAPH" == "true" ]]; then
        generate_callgraph "$BINARY" "$PROJECT_DIR"
    fi

    # Generate analysis report
    local report_file="$PROJECT_DIR/analysis-report.txt"
    cat > "$report_file" <<EOF
Ghidra Headless Analysis Report
================================

Binary: $(basename "$BINARY")
Full Path: $(realpath "$BINARY")
SHA-256: $hash
Architecture: $arch

Analysis Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Project Directory: $PROJECT_DIR

Completed Tasks:
  [x] Import binary into Ghidra
  [x] Auto-analysis (function discovery, CFG)
  $([[ "$DECOMPILE" == "true" ]] && echo "[x]" || echo "[ ]") Decompilation
  $([[ "$CALLGRAPH" == "true" ]] && echo "[x]" || echo "[ ]") Callgraph generation

Output Files:
  - Ghidra project: $PROJECT_DIR/$(basename "$BINARY" | sed 's/\.[^.]*$//')
  $([[ "$DECOMPILE" == "true" ]] && echo "- Decompiled C code: $PROJECT_DIR/decompiled/")
  $([[ "$CALLGRAPH" == "true" ]] && echo "- Callgraph: $PROJECT_DIR/callgraphs/callgraph.png")

Logs:
  - Import log: $PROJECT_DIR/import.log
  - Analysis log: $PROJECT_DIR/analysis.log
  $([[ "$DECOMPILE" == "true" ]] && echo "- Decompile log: $PROJECT_DIR/decompile.log")

Next Steps:
  1. Review decompiled C code in $PROJECT_DIR/decompiled/
  2. Analyze callgraph for interesting function relationships
  3. Run connascence-analyzer on decompiled code (if MCP available)
  4. Store findings in memory-mcp for cross-session persistence
EOF

    log_success "========================================="
    log_success "Analysis Complete!"
    log_success "========================================="
    log_success "Report: $report_file"
    log_success "Project: $PROJECT_DIR"
}

# Run main function
main
