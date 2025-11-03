#!/usr/bin/env bash
#
# Multi-Repository Management Script
# Utilities for coordinating operations across multiple GitHub repositories
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/github-api.sh" 2>/dev/null || true

# Clone multiple repositories
clone_all() {
    local repos_file="$1"
    local target_dir="${2:-.}"

    log_info "Cloning repositories from $repos_file"

    while IFS= read -r repo || [[ -n "$repo" ]]; then
        [[ "$repo" =~ ^#.*$ ]] && continue  # Skip comments
        [[ -z "$repo" ]] && continue  # Skip empty lines

        local repo_name
        repo_name=$(basename "$repo" .git)
        local clone_path="${target_dir}/${repo_name}"

        if [[ -d "$clone_path" ]]; then
            log_warn "Repository $repo_name already exists, skipping"
            continue
        fi

        log_info "Cloning $repo to $clone_path"
        git clone "https://github.com/${repo}.git" "$clone_path"
    done < "$repos_file"
}

# Create PRs across multiple repositories
create_prs() {
    local repos="$1"
    local title="$2"
    local body_template="$3"
    local labels="${4:-}"

    IFS=',' read -ra REPO_ARRAY <<< "$repos"

    for repo in "${REPO_ARRAY[@]}"; do
        log_info "Creating PR for $repo"

        # Get current branch name
        local branch
        branch=$(git -C "$repo" rev-parse --abbrev-ref HEAD)

        # Read body from template
        local body
        body=$(<"$body_template")

        # Create PR
        local pr_data
        pr_data=$(create_pr "$repo" "$branch" "main" "$title" "$body")

        local pr_number
        pr_number=$(echo "$pr_data" | jq -r '.number')

        log_info "Created PR #$pr_number for $repo"

        # Add labels if specified
        if [[ -n "$labels" ]]; then
            IFS=',' read -ra LABEL_ARRAY <<< "$labels"
            add_labels "$repo" "$pr_number" "${LABEL_ARRAY[@]}"
        fi
    done
}

# Check PR status across repositories
pr_status() {
    local repos="$1"

    IFS=',' read -ra REPO_ARRAY <<< "$repos"

    echo "Repository,PR Number,Status,Checks,Reviews"

    for repo in "${REPO_ARRAY[@]}"; do
        # Get open PRs
        local prs
        prs=$(list_issues "$repo" "open" "" 100 | jq -r '.[] | select(.pull_request != null)')

        echo "$prs" | jq -r --arg repo "$repo" \
            '"\($repo),\(.number),\(.state),\(.mergeable_state // "unknown"),\(.reviews_count // 0)"'
    done
}

# Execute synchronized release
synchronized_release() {
    local repos="$1"
    local version_strategy="$2"  # major, minor, patch
    local release_notes_dir="$3"

    IFS=',' read -ra REPO_ARRAY <<< "$repos"

    # Calculate next version for each repo
    declare -A versions

    for repo in "${REPO_ARRAY[@]}"; do
        local current_version
        current_version=$(get_latest_tag "$repo")

        local next_version
        next_version=$(calculate_next_version "$current_version" "$version_strategy")

        versions["$repo"]="$next_version"

        log_info "Planning $repo: $current_version -> $next_version"
    done

    # Create releases in dependency order
    for repo in "${REPO_ARRAY[@]}"; do
        local version="${versions[$repo]}"
        local release_notes="${release_notes_dir}/${repo##*/}-${version}.md"

        if [[ ! -f "$release_notes" ]]; then
            log_error "Release notes not found: $release_notes"
            continue
        fi

        log_info "Creating release for $repo: $version"

        create_release "$repo" "v${version}" "Version ${version}" \
            "$(<"$release_notes")" false false
    done
}

# Determine merge sequence based on dependencies
merge_sequence() {
    local dependency_graph="$1"

    log_info "Calculating merge sequence from dependency graph"

    # Use topological sort to determine order
    # This is simplified - real implementation would use proper graph traversal
    cat "$dependency_graph" | jq -r '.nodes | sort_by(.depth) | .[].name'
}

# Get latest tag from repository
get_latest_tag() {
    local repo="$1"

    api_request GET "/repos/$repo/tags" | jq -r '.[0].name // "v0.0.0"'
}

# Calculate next semantic version
calculate_next_version() {
    local current="$1"
    local strategy="$2"

    # Remove 'v' prefix if present
    current="${current#v}"

    IFS='.' read -r major minor patch <<< "$current"

    case "$strategy" in
        major)
            echo "$((major + 1)).0.0"
            ;;
        minor)
            echo "${major}.$((minor + 1)).0"
            ;;
        patch)
            echo "${major}.${minor}.$((patch + 1))"
            ;;
        *)
            log_error "Unknown version strategy: $strategy"
            return 1
            ;;
    esac
}

# Build repository dependency graph
build_graph() {
    local repos="$1"
    local output="${2:-dependency-graph.dot}"

    log_info "Building dependency graph"

    IFS=',' read -ra REPO_ARRAY <<< "$repos"

    echo "digraph dependencies {" > "$output"
    echo "  rankdir=LR;" >> "$output"
    echo "  node [shape=box];" >> "$output"

    # Analyze each repository for dependencies
    for repo in "${REPO_ARRAY[@]}"; do
        local repo_name="${repo##*/}"

        # Fetch package.json or Cargo.toml to find dependencies
        local package_file
        package_file=$(api_request GET "/repos/$repo/contents/package.json" 2>/dev/null || echo "{}")

        if [[ $(echo "$package_file" | jq -r '.content // ""') != "" ]]; then
            # Decode base64 content
            local content
            content=$(echo "$package_file" | jq -r '.content' | base64 -d)

            # Extract dependencies that match other repos in the list
            local deps
            deps=$(echo "$content" | jq -r '.dependencies // {} | keys[]')

            for dep in $deps; do
                # Check if dependency is one of our repos
                for target_repo in "${REPO_ARRAY[@]}"; do
                    local target_name="${target_repo##*/}"
                    if [[ "$dep" == "$target_name" ]]; then
                        echo "  \"$repo_name\" -> \"$target_name\";" >> "$output"
                    fi
                done
            done
        fi
    done

    echo "}" >> "$output"

    log_info "Dependency graph saved to $output"
}

# Main command dispatcher
main() {
    local command="${1:-}"
    shift || true

    case "$command" in
        clone-all)
            clone_all "$@"
            ;;
        create-prs)
            create_prs "$@"
            ;;
        pr-status)
            pr_status "$@"
            ;;
        synchronized-release)
            synchronized_release "$@"
            ;;
        merge-sequence)
            merge_sequence "$@"
            ;;
        build-graph)
            build_graph "$@"
            ;;
        *)
            log_error "Unknown command: $command"
            echo "Usage: $0 <command> [args...]"
            echo ""
            echo "Commands:"
            echo "  clone-all <repos-file> [target-dir]"
            echo "  create-prs <repos-csv> <title> <body-template> [labels-csv]"
            echo "  pr-status <repos-csv>"
            echo "  synchronized-release <repos-csv> <strategy> <release-notes-dir>"
            echo "  merge-sequence <dependency-graph.dot>"
            echo "  build-graph <repos-csv> [output-file]"
            exit 1
            ;;
    esac
}

main "$@"
