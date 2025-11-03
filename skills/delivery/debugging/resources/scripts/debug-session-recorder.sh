#!/bin/bash

###############################################################################
# Debug Session Recorder
#
# Records debugging session including:
# - Commands executed
# - Test results
# - Code changes
# - System state snapshots
#
# Usage:
#   ./debug-session-recorder.sh start --issue "BUG-123"
#   ./debug-session-recorder.sh stop
#   ./debug-session-recorder.sh report
###############################################################################

set -euo pipefail

# Configuration
SESSION_DIR="${DEBUG_SESSION_DIR:-.debug-sessions}"
CURRENT_SESSION_FILE="${SESSION_DIR}/.current-session"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1" >&2
}

# Initialize session directory
init_session_dir() {
    mkdir -p "${SESSION_DIR}"
}

# Start a new debug session
start_session() {
    local issue_id="${1:-unknown}"
    local timestamp=$(date +%Y%m%d-%H%M%S)
    local session_id="${issue_id}-${timestamp}"
    local session_path="${SESSION_DIR}/${session_id}"

    # Check if session already running
    if [ -f "${CURRENT_SESSION_FILE}" ]; then
        log_error "Debug session already running. Stop it first with: $0 stop"
        exit 1
    fi

    # Create session directory
    mkdir -p "${session_path}"

    # Create session metadata
    cat > "${session_path}/metadata.json" <<EOF
{
  "session_id": "${session_id}",
  "issue_id": "${issue_id}",
  "start_time": "$(date -Iseconds)",
  "user": "${USER}",
  "hostname": "${HOSTNAME}",
  "working_directory": "$(pwd)",
  "git_branch": "$(git branch --show-current 2>/dev/null || echo 'not a git repo')",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
}
EOF

    # Save current session reference
    echo "${session_id}" > "${CURRENT_SESSION_FILE}"

    # Initialize logs
    echo "Debug Session Started: $(date)" > "${session_path}/session.log"
    echo "[]" > "${session_path}/commands.json"
    echo "[]" > "${session_path}/snapshots.json"

    # Capture initial system state
    capture_snapshot "${session_path}" "initial"

    log_success "Debug session started: ${session_id}"
    log_info "Session directory: ${session_path}"
    log_info "To log commands, use: $0 log-command <command>"
}

# Stop current session
stop_session() {
    if [ ! -f "${CURRENT_SESSION_FILE}" ]; then
        log_error "No active debug session"
        exit 1
    fi

    local session_id=$(cat "${CURRENT_SESSION_FILE}")
    local session_path="${SESSION_DIR}/${session_id}"

    # Update metadata with end time
    python3 -c "
import json
import sys
from datetime import datetime

with open('${session_path}/metadata.json', 'r') as f:
    metadata = json.load(f)

metadata['end_time'] = datetime.now().isoformat()

with open('${session_path}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

    # Capture final snapshot
    capture_snapshot "${session_path}" "final"

    # Append to session log
    echo "Debug Session Stopped: $(date)" >> "${session_path}/session.log"

    # Clear current session
    rm "${CURRENT_SESSION_FILE}"

    log_success "Debug session stopped: ${session_id}"
    log_info "Generate report with: $0 report ${session_id}"
}

# Log a command execution
log_command() {
    local command="$1"
    local exit_code="${2:-0}"

    if [ ! -f "${CURRENT_SESSION_FILE}" ]; then
        log_warning "No active debug session. Command not logged."
        return
    fi

    local session_id=$(cat "${CURRENT_SESSION_FILE}")
    local session_path="${SESSION_DIR}/${session_id}"

    # Append command to log
    python3 -c "
import json
import sys
from datetime import datetime

with open('${session_path}/commands.json', 'r') as f:
    commands = json.load(f)

commands.append({
    'timestamp': datetime.now().isoformat(),
    'command': '''${command}''',
    'exit_code': ${exit_code},
    'working_directory': '$(pwd)'
})

with open('${session_path}/commands.json', 'w') as f:
    json.dump(commands, f, indent=2)
"

    echo "[$(date -Iseconds)] Command: ${command} (exit: ${exit_code})" >> "${session_path}/session.log"
}

# Capture system state snapshot
capture_snapshot() {
    local session_path="$1"
    local snapshot_type="${2:-checkpoint}"
    local snapshot_dir="${session_path}/snapshots/$(date +%Y%m%d-%H%M%S)-${snapshot_type}"

    mkdir -p "${snapshot_dir}"

    # Capture git status if in git repo
    if git rev-parse --git-dir > /dev/null 2>&1; then
        git status > "${snapshot_dir}/git-status.txt" 2>&1 || true
        git diff > "${snapshot_dir}/git-diff.txt" 2>&1 || true
        git log -1 --stat > "${snapshot_dir}/git-last-commit.txt" 2>&1 || true
    fi

    # Capture system info
    {
        echo "=== System Information ==="
        uname -a
        echo ""
        echo "=== Memory Usage ==="
        free -h 2>/dev/null || vm_stat 2>/dev/null || true
        echo ""
        echo "=== Disk Usage ==="
        df -h .
        echo ""
        echo "=== Process Info ==="
        ps aux | head -20
    } > "${snapshot_dir}/system-info.txt"

    # Capture environment variables (sanitize sensitive data)
    env | grep -v -E '(PASSWORD|SECRET|TOKEN|KEY)' | sort > "${snapshot_dir}/environment.txt"

    # Log snapshot
    python3 -c "
import json
from datetime import datetime

with open('${session_path}/snapshots.json', 'r') as f:
    snapshots = json.load(f)

snapshots.append({
    'timestamp': datetime.now().isoformat(),
    'type': '${snapshot_type}',
    'path': '${snapshot_dir}'
})

with open('${session_path}/snapshots.json', 'w') as f:
    json.dump(snapshots, f, indent=2)
"

    log_info "Snapshot captured: ${snapshot_type}"
}

# Generate debug report
generate_report() {
    local session_id="${1:-$(cat ${CURRENT_SESSION_FILE} 2>/dev/null || echo '')}"

    if [ -z "${session_id}" ]; then
        log_error "No session ID provided and no active session"
        exit 1
    fi

    local session_path="${SESSION_DIR}/${session_id}"

    if [ ! -d "${session_path}" ]; then
        log_error "Session not found: ${session_id}"
        exit 1
    fi

    local report_file="${session_path}/debug-report.md"

    log_info "Generating debug report..."

    # Generate report
    python3 -c "
import json
from datetime import datetime
from pathlib import Path

session_path = Path('${session_path}')

# Load metadata
with open(session_path / 'metadata.json', 'r') as f:
    metadata = json.load(f)

# Load commands
with open(session_path / 'commands.json', 'r') as f:
    commands = json.load(f)

# Load snapshots
with open(session_path / 'snapshots.json', 'r') as f:
    snapshots = json.load(f)

# Generate report
report = []
report.append('# Debug Session Report')
report.append('')
report.append(f'**Session ID:** {metadata[\"session_id\"]}')
report.append(f'**Issue ID:** {metadata[\"issue_id\"]}')
report.append(f'**Start Time:** {metadata[\"start_time\"]}')
if 'end_time' in metadata:
    report.append(f'**End Time:** {metadata[\"end_time\"]}')
report.append(f'**User:** {metadata[\"user\"]}@{metadata[\"hostname\"]}')
report.append(f'**Working Directory:** {metadata[\"working_directory\"]}')
report.append(f'**Git Branch:** {metadata[\"git_branch\"]}')
report.append(f'**Git Commit:** {metadata[\"git_commit\"]}')
report.append('')

report.append('## Summary')
report.append('')
report.append(f'- Total commands executed: {len(commands)}')
report.append(f'- Failed commands: {sum(1 for cmd in commands if cmd[\"exit_code\"] != 0)}')
report.append(f'- Snapshots captured: {len(snapshots)}')
report.append('')

report.append('## Commands Executed')
report.append('')
for i, cmd in enumerate(commands, 1):
    status = '✓' if cmd['exit_code'] == 0 else '✗'
    report.append(f'{i}. [{status}] `{cmd[\"command\"]}` (exit: {cmd[\"exit_code\"]})')
    report.append(f'   - Time: {cmd[\"timestamp\"]}')
    report.append(f'   - Directory: {cmd[\"working_directory\"]}')
    report.append('')

report.append('## Snapshots')
report.append('')
for snapshot in snapshots:
    report.append(f'- **{snapshot[\"type\"]}** - {snapshot[\"timestamp\"]}')
    report.append(f'  - Path: {snapshot[\"path\"]}')
    report.append('')

report.append('## Next Steps')
report.append('')
report.append('- [ ] Review failed commands')
report.append('- [ ] Analyze snapshots for system state changes')
report.append('- [ ] Document root cause')
report.append('- [ ] Create test case to prevent regression')
report.append('')

# Write report
with open('${report_file}', 'w') as f:
    f.write('\n'.join(report))

print('Report generated successfully')
"

    log_success "Report generated: ${report_file}"
    echo ""
    cat "${report_file}"
}

# List all sessions
list_sessions() {
    log_info "Debug sessions:"
    echo ""

    if [ ! -d "${SESSION_DIR}" ] || [ -z "$(ls -A ${SESSION_DIR} 2>/dev/null)" ]; then
        log_warning "No debug sessions found"
        return
    fi

    for session_path in "${SESSION_DIR}"/*; do
        if [ -d "${session_path}" ]; then
            local session_id=$(basename "${session_path}")
            local metadata_file="${session_path}/metadata.json"

            if [ -f "${metadata_file}" ]; then
                python3 -c "
import json
with open('${metadata_file}', 'r') as f:
    metadata = json.load(f)
    status = 'ACTIVE' if '${session_id}' == '$(cat ${CURRENT_SESSION_FILE} 2>/dev/null || echo '')' else 'COMPLETED'
    print(f'  [{status}] {metadata[\"session_id\"]}')
    print(f'      Issue: {metadata[\"issue_id\"]}')
    print(f'      Start: {metadata[\"start_time\"]}')
    if 'end_time' in metadata:
        print(f'      End: {metadata[\"end_time\"]}')
    print()
"
            fi
        fi
    done
}

# Main command dispatcher
main() {
    init_session_dir

    local command="${1:-}"

    case "${command}" in
        start)
            shift
            local issue_id="${1:-unknown}"
            start_session "${issue_id}"
            ;;
        stop)
            stop_session
            ;;
        log-command)
            shift
            log_command "$@"
            ;;
        snapshot)
            if [ ! -f "${CURRENT_SESSION_FILE}" ]; then
                log_error "No active debug session"
                exit 1
            fi
            local session_id=$(cat "${CURRENT_SESSION_FILE}")
            capture_snapshot "${SESSION_DIR}/${session_id}" "manual"
            ;;
        report)
            shift
            generate_report "$@"
            ;;
        list)
            list_sessions
            ;;
        *)
            echo "Debug Session Recorder"
            echo ""
            echo "Usage:"
            echo "  $0 start [--issue ISSUE-ID]   Start new debug session"
            echo "  $0 stop                        Stop current session"
            echo "  $0 log-command <command>       Log a command"
            echo "  $0 snapshot                    Capture manual snapshot"
            echo "  $0 report [SESSION-ID]         Generate debug report"
            echo "  $0 list                        List all sessions"
            exit 1
            ;;
    esac
}

main "$@"
