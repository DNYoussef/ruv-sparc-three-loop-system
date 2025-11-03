#!/usr/bin/env bash
#
# GitHub Pull Request Automation Script
# Automated PR workflows including creation, review, merge, and cleanup
# Supports bulk operations and multi-repo coordination
#

set -euo pipefail

# Configuration
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GITHUB_API_URL="${GITHUB_API_URL:-https://api.github.com}"
DRY_RUN="${DRY_RUN:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
log_debug() { [[ "${VERBOSE:-false}" == "true" ]] && echo -e "${BLUE}[DEBUG]${NC} $*" || true; }

# Check prerequisites
check_prerequisites() {
    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "GITHUB_TOKEN not set"
        log_info "Set with: export GITHUB_TOKEN=<token>"
        exit 1
    fi

    for cmd in jq curl git; do
        if ! command -v "$cmd" &>/dev/null; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
}

# API request helper
api_request() {
    local method="$1"
    local endpoint="$2"
    local data="${3:-}"

    local args=(
        -X "$method"
        -H "Authorization: Bearer $GITHUB_TOKEN"
        -H "Accept: application/vnd.github+json"
        -H "X-GitHub-Api-Version: 2022-11-28"
        -s -w "\n%{http_code}"
    )

    [[ -n "$data" ]] && args+=(-H "Content-Type: application/json" -d "$data")

    local response
    response=$(curl "${args[@]}" "${GITHUB_API_URL}${endpoint}")

    local http_code body
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    if [[ "$http_code" -ge 400 ]]; then
        log_error "API request failed: $http_code"
        echo "$body" | jq '.' >&2 2>/dev/null || echo "$body" >&2
        return 1
    fi

    echo "$body"
}

# Create pull request
create_pr() {
    local repo="$1"
    local head="$2"
    local base="$3"
    local title="$4"
    local body="${5:-}"
    local draft="${6:-false}"

    log_info "Creating PR: $head -> $base in $repo"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "DRY RUN - Would create PR: $title"
        return 0
    fi

    local data
    data=$(jq -n \
        --arg head "$head" \
        --arg base "$base" \
        --arg title "$title" \
        --arg body "$body" \
        --argjson draft "$draft" \
        '{head: $head, base: $base, title: $title, body: $body, draft: $draft}')

    api_request POST "/repos/$repo/pulls" "$data"
}

# Auto-merge pull request
auto_merge_pr() {
    local repo="$1"
    local pr_number="$2"
    local merge_method="${3:-squash}" # merge, squash, rebase

    log_info "Auto-merging PR #$pr_number with method: $merge_method"

    # Check if PR is mergeable
    local pr_data
    pr_data=$(api_request GET "/repos/$repo/pulls/$pr_number")

    local mergeable mergeable_state
    mergeable=$(echo "$pr_data" | jq -r '.mergeable')
    mergeable_state=$(echo "$pr_data" | jq -r '.mergeable_state')

    if [[ "$mergeable" != "true" ]]; then
        log_error "PR #$pr_number is not mergeable (state: $mergeable_state)"
        return 1
    fi

    # Check required checks
    local required_checks_passed
    required_checks_passed=$(check_required_status "$repo" "$pr_number")

    if [[ "$required_checks_passed" != "true" ]]; then
        log_error "Required checks have not passed for PR #$pr_number"
        return 1
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "DRY RUN - Would merge PR #$pr_number"
        return 0
    fi

    # Merge PR
    local data
    data=$(jq -n --arg method "$merge_method" '{merge_method: $method}')

    api_request PUT "/repos/$repo/pulls/$pr_number/merge" "$data"
    log_info "Successfully merged PR #$pr_number"
}

# Check required status checks
check_required_status() {
    local repo="$1"
    local pr_number="$2"

    local pr_data
    pr_data=$(api_request GET "/repos/$repo/pulls/$pr_number")

    local head_sha
    head_sha=$(echo "$pr_data" | jq -r '.head.sha')

    # Get commit status
    local status
    status=$(api_request GET "/repos/$repo/commits/$head_sha/status")

    local state
    state=$(echo "$status" | jq -r '.state')

    [[ "$state" == "success" ]] && echo "true" || echo "false"
}

# Add reviewers to PR
add_reviewers() {
    local repo="$1"
    local pr_number="$2"
    shift 2
    local reviewers=("$@")

    log_info "Adding reviewers to PR #$pr_number: ${reviewers[*]}"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "DRY RUN - Would add reviewers: ${reviewers[*]}"
        return 0
    fi

    local data
    data=$(jq -n --arg reviewers "${reviewers[*]}" '{reviewers: ($reviewers | split(" "))}')

    api_request POST "/repos/$repo/pulls/$pr_number/requested_reviewers" "$data"
}

# Sync PR branch with base
sync_pr_branch() {
    local repo="$1"
    local pr_number="$2"

    log_info "Syncing PR #$pr_number with base branch"

    local pr_data
    pr_data=$(api_request GET "/repos/$repo/pulls/$pr_number")

    local head_ref base_ref
    head_ref=$(echo "$pr_data" | jq -r '.head.ref')
    base_ref=$(echo "$pr_data" | jq -r '.base.ref')

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "DRY RUN - Would sync $head_ref with $base_ref"
        return 0
    fi

    # Update branch via API
    api_request PUT "/repos/$repo/pulls/$pr_number/update-branch" "{}"
    log_info "Successfully synced PR #$pr_number"
}

# Close stale PRs
close_stale_prs() {
    local repo="$1"
    local days="${2:-30}"
    local dry_run="${DRY_RUN:-true}"

    log_info "Closing PRs older than $days days in $repo"

    local cutoff_date
    cutoff_date=$(date -d "$days days ago" -I)

    # Get open PRs
    local prs
    prs=$(api_request GET "/repos/$repo/pulls?state=open&per_page=100")

    local stale_count=0
    while IFS= read -r pr; do
        local pr_number updated_at title
        pr_number=$(echo "$pr" | jq -r '.number')
        updated_at=$(echo "$pr" | jq -r '.updated_at')
        title=$(echo "$pr" | jq -r '.title')

        if [[ "$updated_at" < "$cutoff_date" ]]; then
            log_warn "Stale PR #$pr_number: $title (last updated: $updated_at)"

            if [[ "$dry_run" != "true" ]]; then
                # Add stale label and comment
                api_request POST "/repos/$repo/issues/$pr_number/labels" '{"labels":["stale"]}'

                local comment="This PR has been automatically marked as stale because it has not had recent activity. It will be closed in 7 days if no further activity occurs."
                local data
                data=$(jq -n --arg body "$comment" '{body: $body}')
                api_request POST "/repos/$repo/issues/$pr_number/comments" "$data"

                ((stale_count++))
            fi
        fi
    done < <(echo "$prs" | jq -c '.[]')

    log_info "Found $stale_count stale PR(s)"
}

# Bulk PR creation from branch list
bulk_create_prs() {
    local repo="$1"
    local base_branch="$2"
    local pr_template="$3"
    local branch_list_file="$4"

    log_info "Creating PRs from branch list: $branch_list_file"

    if [[ ! -f "$branch_list_file" ]]; then
        log_error "Branch list file not found: $branch_list_file"
        return 1
    fi

    local created=0 failed=0
    while IFS= read -r branch; do
        [[ -z "$branch" || "$branch" =~ ^# ]] && continue

        local title body
        title="[Automated] Update from $branch"
        body=$(cat "$pr_template" 2>/dev/null || echo "Automated PR from $branch")

        if create_pr "$repo" "$branch" "$base_branch" "$title" "$body" "false" &>/dev/null; then
            log_info "Created PR for branch: $branch"
            ((created++))
        else
            log_error "Failed to create PR for branch: $branch"
            ((failed++))
        fi
    done < "$branch_list_file"

    log_info "Bulk PR creation complete: $created created, $failed failed"
}

# Usage information
usage() {
    cat <<EOF
GitHub Pull Request Automation

USAGE:
    $0 <command> [options]

COMMANDS:
    create <repo> <head> <base> <title> [body] [draft]
        Create a pull request

    auto-merge <repo> <pr-number> [merge-method]
        Automatically merge a PR (after checks pass)

    add-reviewers <repo> <pr-number> <reviewer1> [reviewer2...]
        Add reviewers to a PR

    sync <repo> <pr-number>
        Sync PR branch with base branch

    close-stale <repo> [days]
        Close or mark stale PRs (default: 30 days)

    bulk-create <repo> <base-branch> <template-file> <branch-list-file>
        Create PRs from list of branches

ENVIRONMENT:
    GITHUB_TOKEN    GitHub personal access token (required)
    DRY_RUN         Set to "true" for dry-run mode
    VERBOSE         Set to "true" for debug output

EXAMPLES:
    # Create PR
    $0 create owner/repo feature-branch main "Add new feature" "Description here"

    # Auto-merge PR #123
    $0 auto-merge owner/repo 123 squash

    # Close stale PRs older than 60 days
    $0 close-stale owner/repo 60

EOF
}

# Main command dispatcher
main() {
    local command="${1:-}"
    shift || true

    check_prerequisites

    case "$command" in
        create)
            create_pr "$@"
            ;;
        auto-merge)
            auto_merge_pr "$@"
            ;;
        add-reviewers)
            add_reviewers "$@"
            ;;
        sync)
            sync_pr_branch "$@"
            ;;
        close-stale)
            close_stale_prs "$@"
            ;;
        bulk-create)
            bulk_create_prs "$@"
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

main "$@"
