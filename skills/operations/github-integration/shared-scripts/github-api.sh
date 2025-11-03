#!/usr/bin/env bash
#
# GitHub API Utility Script
# Comprehensive GitHub REST API and GraphQL interactions
# Used across all GitHub integration skills
#

set -euo pipefail

# Configuration
GITHUB_API_URL="${GITHUB_API_URL:-https://api.github.com}"
GITHUB_GRAPHQL_URL="${GITHUB_GRAPHQL_URL:-https://api.github.com/graphql}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Check if GitHub token is set
check_token() {
    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "GITHUB_TOKEN environment variable not set"
        log_info "Set token: export GITHUB_TOKEN=<your-token>"
        exit 1
    fi
}

# Make authenticated API request
api_request() {
    local method="$1"
    local endpoint="$2"
    local data="${3:-}"

    check_token

    local args=(
        -X "$method"
        -H "Authorization: Bearer $GITHUB_TOKEN"
        -H "Accept: application/vnd.github+json"
        -H "X-GitHub-Api-Version: 2022-11-28"
        -s -w "\n%{http_code}"
    )

    if [[ -n "$data" ]]; then
        args+=(-H "Content-Type: application/json" -d "$data")
    fi

    local response
    response=$(curl "${args[@]}" "${GITHUB_API_URL}${endpoint}")

    local http_code
    http_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | sed '$d')

    if [[ "$http_code" -ge 400 ]]; then
        log_error "API request failed with status $http_code"
        echo "$body" | jq '.' >&2 2>/dev/null || echo "$body" >&2
        return 1
    fi

    echo "$body"
}

# GraphQL query
graphql_query() {
    local query="$1"
    local variables="${2:-{}}"

    check_token

    local data
    data=$(jq -n \
        --arg query "$query" \
        --argjson variables "$variables" \
        '{query: $query, variables: $variables}')

    curl -X POST \
        -H "Authorization: Bearer $GITHUB_TOKEN" \
        -H "Content-Type: application/json" \
        -s "$GITHUB_GRAPHQL_URL" \
        -d "$data"
}

# Fetch pull request details
fetch_pr() {
    local repo="$1"
    local pr_number="$2"
    local include_comments="${3:-false}"

    log_info "Fetching PR #$pr_number from $repo"

    local pr_data
    pr_data=$(api_request GET "/repos/$repo/pulls/$pr_number")

    if [[ "$include_comments" == "true" ]]; then
        local comments
        comments=$(api_request GET "/repos/$repo/issues/$pr_number/comments")

        echo "$pr_data" | jq \
            --argjson comments "$comments" \
            '. + {comments: $comments}'
    else
        echo "$pr_data"
    fi
}

# Fetch issue details
fetch_issue() {
    local repo="$1"
    local issue_number="$2"
    local include_comments="${3:-false}"

    log_info "Fetching issue #$issue_number from $repo"

    local issue_data
    issue_data=$(api_request GET "/repos/$repo/issues/$issue_number")

    if [[ "$include_comments" == "true" ]]; then
        local comments
        comments=$(api_request GET "/repos/$repo/issues/$issue_number/comments")

        echo "$issue_data" | jq \
            --argjson comments "$comments" \
            '. + {comments: $comments}'
    else
        echo "$issue_data"
    fi
}

# Post review to pull request
post_review() {
    local repo="$1"
    local pr_number="$2"
    local event="$3"  # APPROVE, REQUEST_CHANGES, COMMENT
    local body="$4"

    log_info "Posting review to PR #$pr_number: $event"

    local data
    data=$(jq -n \
        --arg body "$body" \
        --arg event "$event" \
        '{body: $body, event: $event}')

    api_request POST "/repos/$repo/pulls/$pr_number/reviews" "$data"
}

# Add inline comment to PR
add_pr_comment() {
    local repo="$1"
    local pr_number="$2"
    local path="$3"
    local line="$4"
    local body="$5"

    log_info "Adding comment to $path:$line"

    # Get PR details to find commit SHA
    local pr_data
    pr_data=$(api_request GET "/repos/$repo/pulls/$pr_number")
    local commit_id
    commit_id=$(echo "$pr_data" | jq -r '.head.sha')

    local data
    data=$(jq -n \
        --arg body "$body" \
        --arg commit_id "$commit_id" \
        --arg path "$path" \
        --argjson line "$line" \
        '{body: $body, commit_id: $commit_id, path: $path, line: $line}')

    api_request POST "/repos/$repo/pulls/$pr_number/comments" "$data"
}

# Add labels to issue/PR
add_labels() {
    local repo="$1"
    local issue_number="$2"
    shift 2
    local labels=("$@")

    log_info "Adding labels to #$issue_number: ${labels[*]}"

    local data
    data=$(jq -n --arg labels "${labels[*]}" '{labels: ($labels | split(" "))}')

    api_request POST "/repos/$repo/issues/$issue_number/labels" "$data"
}

# Assign issue to user
assign_issue() {
    local repo="$1"
    local issue_number="$2"
    local assignee="$3"

    log_info "Assigning #$issue_number to $assignee"

    local data
    data=$(jq -n --arg assignee "$assignee" '{assignees: [$assignee]}')

    api_request POST "/repos/$repo/issues/$issue_number/assignees" "$data"
}

# Set milestone
set_milestone() {
    local repo="$1"
    local issue_number="$2"
    local milestone_title="$3"

    log_info "Setting milestone for #$issue_number to '$milestone_title'"

    # Find milestone by title
    local milestones
    milestones=$(api_request GET "/repos/$repo/milestones")
    local milestone_number
    milestone_number=$(echo "$milestones" | jq -r \
        --arg title "$milestone_title" \
        '.[] | select(.title == $title) | .number')

    if [[ -z "$milestone_number" ]]; then
        log_error "Milestone '$milestone_title' not found"
        return 1
    fi

    local data
    data=$(jq -n --argjson milestone "$milestone_number" '{milestone: $milestone}')

    api_request PATCH "/repos/$repo/issues/$issue_number" "$data"
}

# Create milestone
create_milestone() {
    local repo="$1"
    local title="$2"
    local due_date="$3"
    local description="$4"

    log_info "Creating milestone: $title"

    local data
    data=$(jq -n \
        --arg title "$title" \
        --arg due_on "$due_date" \
        --arg description "$description" \
        '{title: $title, due_on: $due_on, description: $description}')

    api_request POST "/repos/$repo/milestones" "$data"
}

# Create release
create_release() {
    local repo="$1"
    local tag="$2"
    local name="$3"
    local body="$4"
    local draft="${5:-false}"
    local prerelease="${6:-false}"

    log_info "Creating release: $tag"

    local data
    data=$(jq -n \
        --arg tag "$tag" \
        --arg name "$name" \
        --arg body "$body" \
        --argjson draft "$draft" \
        --argjson prerelease "$prerelease" \
        '{tag_name: $tag, name: $name, body: $body, draft: $draft, prerelease: $prerelease}')

    api_request POST "/repos/$repo/releases" "$data"
}

# Publish release
publish_release() {
    local repo="$1"
    local tag="$2"

    log_info "Publishing release: $tag"

    # Find release by tag
    local releases
    releases=$(api_request GET "/repos/$repo/releases")
    local release_id
    release_id=$(echo "$releases" | jq -r \
        --arg tag "$tag" \
        '.[] | select(.tag_name == $tag) | .id')

    if [[ -z "$release_id" ]]; then
        log_error "Release with tag '$tag' not found"
        return 1
    fi

    local data='{"draft": false}'

    api_request PATCH "/repos/$repo/releases/$release_id" "$data"
}

# List issues with filters
list_issues() {
    local repo="$1"
    local state="${2:-open}"
    local labels="${3:-}"
    local per_page="${4:-100}"

    log_info "Listing issues from $repo (state: $state)"

    local endpoint="/repos/$repo/issues?state=$state&per_page=$per_page"

    if [[ -n "$labels" ]]; then
        endpoint="${endpoint}&labels=${labels}"
    fi

    api_request GET "$endpoint"
}

# Create pull request
create_pr() {
    local repo="$1"
    local head="$2"
    local base="$3"
    local title="$4"
    local body="$5"

    log_info "Creating PR: $head -> $base"

    local data
    data=$(jq -n \
        --arg head "$head" \
        --arg base "$base" \
        --arg title "$title" \
        --arg body "$body" \
        '{head: $head, base: $base, title: $title, body: $body}')

    api_request POST "/repos/$repo/pulls" "$data"
}

# Get workflow run details
get_workflow_run() {
    local repo="$1"
    local run_id="$2"
    local include_logs="${3:-false}"

    log_info "Fetching workflow run: $run_id"

    local run_data
    run_data=$(api_request GET "/repos/$repo/actions/runs/$run_id")

    if [[ "$include_logs" == "true" ]]; then
        local logs_url
        logs_url=$(echo "$run_data" | jq -r '.logs_url')

        local logs
        logs=$(curl -L -H "Authorization: Bearer $GITHUB_TOKEN" -s "$logs_url")

        echo "$run_data" | jq --arg logs "$logs" '. + {logs: $logs}'
    else
        echo "$run_data"
    fi
}

# List repository branches
list_branches() {
    local repo="$1"
    local per_page="${2:-100}"

    log_info "Listing branches from $repo"

    api_request GET "/repos/$repo/branches?per_page=$per_page"
}

# Get repository details
get_repo() {
    local repo="$1"

    log_info "Fetching repository: $repo"

    api_request GET "/repos/$repo"
}

# Main command dispatcher
main() {
    local command="${1:-}"
    shift || true

    case "$command" in
        fetch-pr)
            fetch_pr "$@"
            ;;
        fetch-issue)
            fetch_issue "$@"
            ;;
        post-review)
            post_review "$@"
            ;;
        add-comment)
            add_pr_comment "$@"
            ;;
        add-labels)
            add_labels "$@"
            ;;
        assign-issue)
            assign_issue "$@"
            ;;
        set-milestone)
            set_milestone "$@"
            ;;
        create-milestone)
            create_milestone "$@"
            ;;
        create-release)
            create_release "$@"
            ;;
        publish-release)
            publish_release "$@"
            ;;
        list-issues)
            list_issues "$@"
            ;;
        create-pr)
            create_pr "$@"
            ;;
        get-workflow-run)
            get_workflow_run "$@"
            ;;
        list-branches)
            list_branches "$@"
            ;;
        get-repo)
            get_repo "$@"
            ;;
        *)
            log_error "Unknown command: $command"
            echo "Usage: $0 <command> [args...]"
            echo ""
            echo "Commands:"
            echo "  fetch-pr <repo> <pr-number> [include-comments]"
            echo "  fetch-issue <repo> <issue-number> [include-comments]"
            echo "  post-review <repo> <pr-number> <event> <body>"
            echo "  add-comment <repo> <pr-number> <path> <line> <body>"
            echo "  add-labels <repo> <issue-number> <label1> [label2...]"
            echo "  assign-issue <repo> <issue-number> <assignee>"
            echo "  set-milestone <repo> <issue-number> <milestone-title>"
            echo "  create-milestone <repo> <title> <due-date> <description>"
            echo "  create-release <repo> <tag> <name> <body> [draft] [prerelease]"
            echo "  publish-release <repo> <tag>"
            echo "  list-issues <repo> [state] [labels] [per-page]"
            echo "  create-pr <repo> <head> <base> <title> <body>"
            echo "  get-workflow-run <repo> <run-id> [include-logs]"
            echo "  list-branches <repo> [per-page]"
            echo "  get-repo <repo>"
            exit 1
            ;;
    esac
}

main "$@"
