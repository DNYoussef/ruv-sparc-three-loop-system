#!/bin/bash
# hook-installer.sh - Install and configure Claude Flow hooks
# Usage: bash hook-installer.sh [--config path] [--dry-run] [--verbose]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOKS_DIR="${HOOKS_DIR:-$HOME/.claude-flow/hooks}"
CONFIG_FILE="${1:-$SCRIPT_DIR/../templates/pre-task-hook.yaml}"
DRY_RUN=false
VERBOSE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --config PATH    Path to hook configuration file"
      echo "  --dry-run        Show what would be done without executing"
      echo "  --verbose        Enable verbose output"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Logging functions
log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

log_verbose() {
  if [[ "$VERBOSE" == "true" ]]; then
    echo -e "[DEBUG] $1"
  fi
}

# Check prerequisites
check_prerequisites() {
  log_info "Checking prerequisites..."

  # Check for Node.js
  if ! command -v node &> /dev/null; then
    log_error "Node.js is not installed. Please install Node.js 18+ first."
    exit 1
  fi

  # Check for npm
  if ! command -v npm &> /dev/null; then
    log_error "npm is not installed. Please install npm first."
    exit 1
  fi

  # Check for Claude Flow
  if ! command -v npx &> /dev/null; then
    log_error "npx is not available. Please reinstall Node.js."
    exit 1
  fi

  log_verbose "Node.js version: $(node --version)"
  log_verbose "npm version: $(npm --version)"

  log_info "Prerequisites check passed"
}

# Create hooks directory structure
setup_directories() {
  log_info "Setting up hooks directories..."

  if [[ "$DRY_RUN" == "true" ]]; then
    log_verbose "Would create: $HOOKS_DIR"
    log_verbose "Would create: $HOOKS_DIR/pre-task"
    log_verbose "Would create: $HOOKS_DIR/post-edit"
    log_verbose "Would create: $HOOKS_DIR/session"
    return
  fi

  mkdir -p "$HOOKS_DIR"/{pre-task,post-edit,post-task,session,git}
  log_verbose "Created hooks directories at $HOOKS_DIR"

  log_info "Directories created successfully"
}

# Install Claude Flow if not present
install_claude_flow() {
  log_info "Checking Claude Flow installation..."

  if npx --yes claude-flow@alpha --version &> /dev/null; then
    log_info "Claude Flow is already installed"
    return
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    log_verbose "Would install: claude-flow@alpha"
    return
  fi

  log_info "Installing Claude Flow..."
  npm install -g claude-flow@alpha

  log_info "Claude Flow installed successfully"
}

# Copy hook templates
install_hook_templates() {
  log_info "Installing hook templates..."

  local templates_dir="$SCRIPT_DIR/../templates"

  if [[ ! -d "$templates_dir" ]]; then
    log_error "Templates directory not found: $templates_dir"
    exit 1
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    log_verbose "Would copy templates from $templates_dir to $HOOKS_DIR"
    return
  fi

  # Copy pre-task hook
  if [[ -f "$templates_dir/pre-task-hook.yaml" ]]; then
    cp "$templates_dir/pre-task-hook.yaml" "$HOOKS_DIR/pre-task/config.yaml"
    log_verbose "Installed pre-task hook template"
  fi

  # Copy post-edit hook
  if [[ -f "$templates_dir/post-edit-hook.json" ]]; then
    cp "$templates_dir/post-edit-hook.json" "$HOOKS_DIR/post-edit/config.json"
    log_verbose "Installed post-edit hook template"
  fi

  # Copy session hook
  if [[ -f "$templates_dir/session-hooks.yaml" ]]; then
    cp "$templates_dir/session-hooks.yaml" "$HOOKS_DIR/session/config.yaml"
    log_verbose "Installed session hook template"
  fi

  log_info "Hook templates installed successfully"
}

# Create hook wrapper scripts
create_hook_wrappers() {
  log_info "Creating hook wrapper scripts..."

  if [[ "$DRY_RUN" == "true" ]]; then
    log_verbose "Would create hook wrappers in $HOOKS_DIR"
    return
  fi

  # Pre-task hook wrapper
  cat > "$HOOKS_DIR/pre-task/run.sh" <<'EOF'
#!/bin/bash
# Pre-task hook wrapper
npx claude-flow@alpha hooks pre-task --description "$1"
EOF
  chmod +x "$HOOKS_DIR/pre-task/run.sh"

  # Post-edit hook wrapper
  cat > "$HOOKS_DIR/post-edit/run.sh" <<'EOF'
#!/bin/bash
# Post-edit hook wrapper
npx claude-flow@alpha hooks post-edit --file "$1" --memory-key "hooks/post-edit/$2"
EOF
  chmod +x "$HOOKS_DIR/post-edit/run.sh"

  # Session restore wrapper
  cat > "$HOOKS_DIR/session/restore.sh" <<'EOF'
#!/bin/bash
# Session restore hook wrapper
npx claude-flow@alpha hooks session-restore --session-id "$1"
EOF
  chmod +x "$HOOKS_DIR/session/restore.sh"

  log_verbose "Created hook wrappers with execute permissions"
  log_info "Hook wrappers created successfully"
}

# Test hook installation
test_hooks() {
  log_info "Testing hook installation..."

  if [[ "$DRY_RUN" == "true" ]]; then
    log_verbose "Would test hook execution"
    return
  fi

  # Test pre-task hook
  if [[ -x "$HOOKS_DIR/pre-task/run.sh" ]]; then
    log_verbose "Pre-task hook is executable"
  else
    log_warn "Pre-task hook is not executable"
  fi

  # Test post-edit hook
  if [[ -x "$HOOKS_DIR/post-edit/run.sh" ]]; then
    log_verbose "Post-edit hook is executable"
  else
    log_warn "Post-edit hook is not executable"
  fi

  log_info "Hook installation tests completed"
}

# Main installation flow
main() {
  log_info "Starting hooks installation..."
  log_verbose "Hooks directory: $HOOKS_DIR"
  log_verbose "Dry run: $DRY_RUN"

  check_prerequisites
  setup_directories
  install_claude_flow
  install_hook_templates
  create_hook_wrappers
  test_hooks

  log_info "Hooks installation completed successfully!"
  echo ""
  log_info "Next steps:"
  echo "  1. Validate installation: python $SCRIPT_DIR/hook-validator.py"
  echo "  2. Test hooks: bash $SCRIPT_DIR/hook-tester.sh"
  echo "  3. Review configuration: $HOOKS_DIR"
  echo ""
  log_info "Hook directory: $HOOKS_DIR"
}

# Run main installation
main
