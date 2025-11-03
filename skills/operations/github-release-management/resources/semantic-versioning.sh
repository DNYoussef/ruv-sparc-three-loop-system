#!/usr/bin/env bash
# Semantic Versioning Analysis and Automation
# Analyzes commit history and suggests appropriate version bumps

set -euo pipefail

VERSION_FILE="${VERSION_FILE:-package.json}"
CURRENT_VERSION=""
SUGGESTED_VERSION=""
BUMP_TYPE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Get current version from package.json or git tag
get_current_version() {
    if [[ -f "$VERSION_FILE" ]]; then
        CURRENT_VERSION=$(grep -oP '"version":\s*"\K[^"]+' "$VERSION_FILE" 2>/dev/null || echo "")
    fi

    if [[ -z "$CURRENT_VERSION" ]]; then
        CURRENT_VERSION=$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "0.0.0")
    fi

    log_info "Current version: $CURRENT_VERSION"
}

# Analyze commits since last tag for breaking changes, features, fixes
analyze_commits() {
    local last_tag
    last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

    if [[ -z "$last_tag" ]]; then
        log_warning "No previous tags found, analyzing all commits"
        last_tag="HEAD~10"
    fi

    local commits
    commits=$(git log "$last_tag"..HEAD --pretty=format:"%s" 2>/dev/null || echo "")

    if [[ -z "$commits" ]]; then
        log_warning "No commits found since last tag"
        BUMP_TYPE="none"
        return
    fi

    # Check for breaking changes (BREAKING CHANGE or !)
    if echo "$commits" | grep -qiE "BREAKING CHANGE|^[a-z]+(\([a-z]+\))?!:"; then
        log_warning "Breaking changes detected"
        BUMP_TYPE="major"
        return
    fi

    # Check for features (feat:)
    if echo "$commits" | grep -qiE "^feat(\([a-z]+\))?:"; then
        log_info "New features detected"
        BUMP_TYPE="minor"
        return
    fi

    # Check for fixes (fix:, bugfix:)
    if echo "$commits" | grep -qiE "^(fix|bugfix)(\([a-z]+\))?:"; then
        log_info "Bug fixes detected"
        BUMP_TYPE="patch"
        return
    fi

    # Default to patch for any other commits
    log_info "Regular commits detected"
    BUMP_TYPE="patch"
}

# Calculate new version based on bump type
calculate_new_version() {
    local major minor patch
    IFS='.' read -r major minor patch <<< "$CURRENT_VERSION"

    # Remove any pre-release or build metadata
    patch="${patch%%-*}"
    patch="${patch%%+*}"

    case "$BUMP_TYPE" in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
        none)
            SUGGESTED_VERSION="$CURRENT_VERSION"
            return
            ;;
        *)
            log_error "Unknown bump type: $BUMP_TYPE"
            exit 1
            ;;
    esac

    SUGGESTED_VERSION="$major.$minor.$patch"
    log_success "Suggested version: $CURRENT_VERSION → $SUGGESTED_VERSION ($BUMP_TYPE)"
}

# Update version in package.json
update_version_file() {
    if [[ -f "$VERSION_FILE" ]]; then
        sed -i.bak "s/\"version\": \"$CURRENT_VERSION\"/\"version\": \"$SUGGESTED_VERSION\"/" "$VERSION_FILE"
        rm -f "$VERSION_FILE.bak"
        log_success "Updated $VERSION_FILE to $SUGGESTED_VERSION"
    fi
}

# Create git tag
create_git_tag() {
    local tag_name="v$SUGGESTED_VERSION"

    if git rev-parse "$tag_name" >/dev/null 2>&1; then
        log_warning "Tag $tag_name already exists"
        return
    fi

    # Generate tag message from recent commits
    local last_tag
    last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

    local tag_message
    if [[ -n "$last_tag" ]]; then
        tag_message=$(git log "$last_tag"..HEAD --pretty=format:"- %s" | head -10)
    else
        tag_message=$(git log --pretty=format:"- %s" | head -10)
    fi

    git tag -a "$tag_name" -m "Release $SUGGESTED_VERSION

$tag_message"

    log_success "Created tag: $tag_name"
}

# Validate semantic version format
validate_version() {
    local version="$1"
    if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9\.]+)?(\+[a-zA-Z0-9\.]+)?$ ]]; then
        log_error "Invalid semantic version format: $version"
        return 1
    fi
    return 0
}

# Show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Semantic versioning analysis and automation tool.

OPTIONS:
    --analyze           Analyze commits and suggest version bump
    --bump TYPE         Bump version (major|minor|patch)
    --set VERSION       Set specific version
    --create-tag        Create git tag for current/suggested version
    --push-tags         Push tags to remote
    --dry-run           Show what would be done without making changes
    -h, --help          Show this help message

EXAMPLES:
    # Analyze and suggest version
    $0 --analyze

    # Bump minor version
    $0 --bump minor

    # Set specific version
    $0 --set 2.0.0

    # Analyze, bump, and create tag
    $0 --analyze --create-tag

    # Full release workflow
    $0 --analyze --create-tag --push-tags

ENVIRONMENT VARIABLES:
    VERSION_FILE        Version file to update (default: package.json)
EOF
}

# Main execution
main() {
    local do_analyze=false
    local do_create_tag=false
    local do_push_tags=false
    local dry_run=false
    local manual_bump=""
    local set_version=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --analyze)
                do_analyze=true
                shift
                ;;
            --bump)
                manual_bump="$2"
                shift 2
                ;;
            --set)
                set_version="$2"
                shift 2
                ;;
            --create-tag)
                do_create_tag=true
                shift
                ;;
            --push-tags)
                do_push_tags=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Get current version
    get_current_version

    # Handle manual version set
    if [[ -n "$set_version" ]]; then
        if ! validate_version "$set_version"; then
            exit 1
        fi
        SUGGESTED_VERSION="$set_version"
        log_info "Setting version to: $SUGGESTED_VERSION"
    elif [[ -n "$manual_bump" ]]; then
        BUMP_TYPE="$manual_bump"
        calculate_new_version
    elif [[ "$do_analyze" == true ]]; then
        analyze_commits
        calculate_new_version
    else
        log_error "No action specified. Use --analyze, --bump, or --set"
        show_help
        exit 1
    fi

    # Exit if no version change
    if [[ "$BUMP_TYPE" == "none" ]] && [[ -z "$set_version" ]]; then
        log_info "No version bump needed"
        exit 0
    fi

    # Apply changes unless dry-run
    if [[ "$dry_run" == true ]]; then
        log_info "DRY RUN - Would update version to: $SUGGESTED_VERSION"
        log_info "DRY RUN - Would update file: $VERSION_FILE"
        if [[ "$do_create_tag" == true ]]; then
            log_info "DRY RUN - Would create tag: v$SUGGESTED_VERSION"
        fi
        if [[ "$do_push_tags" == true ]]; then
            log_info "DRY RUN - Would push tags to remote"
        fi
    else
        update_version_file

        if [[ "$do_create_tag" == true ]]; then
            create_git_tag
        fi

        if [[ "$do_push_tags" == true ]]; then
            git push --tags
            log_success "Pushed tags to remote"
        fi
    fi

    log_success "Version bump complete: $CURRENT_VERSION → $SUGGESTED_VERSION"
}

# Run main if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
