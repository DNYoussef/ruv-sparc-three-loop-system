#!/bin/bash
#
# Template Structure Validator
# Validates generated project templates for compliance with best practices
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Validation results
PASSED=0
FAILED=0
WARNINGS=0

# Print functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

print_error() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

print_header() {
    echo ""
    echo "================================================"
    echo "$1"
    echo "================================================"
}

# Validation functions
check_required_files() {
    print_header "Checking Required Files"

    local required_files=(".gitignore" "README.md")

    for file in "${required_files[@]}"; do
        if [[ -f "$1/$file" ]]; then
            print_success "Found $file"
        else
            print_error "Missing $file"
        fi
    done
}

check_node_structure() {
    print_header "Validating Node.js Project Structure"

    local project_path="$1"

    # Check package.json
    if [[ -f "$project_path/package.json" ]]; then
        print_success "Found package.json"

        # Validate package.json content
        if command -v jq &> /dev/null; then
            # Check for required fields
            local name=$(jq -r '.name' "$project_path/package.json")
            local version=$(jq -r '.version' "$project_path/package.json")
            local scripts=$(jq -r '.scripts' "$project_path/package.json")

            [[ "$name" != "null" ]] && print_success "Package name: $name" || print_error "Missing package name"
            [[ "$version" != "null" ]] && print_success "Package version: $version" || print_error "Missing package version"
            [[ "$scripts" != "null" ]] && print_success "Scripts defined" || print_error "No scripts defined"

            # Check for security: no hardcoded credentials
            if grep -q "password\|secret\|key.*:" "$project_path/package.json" 2>/dev/null; then
                print_warning "Possible credentials in package.json"
            fi
        else
            print_warning "jq not installed, skipping detailed package.json validation"
        fi
    else
        print_error "Missing package.json"
    fi

    # Check directory structure
    [[ -d "$project_path/src" ]] && print_success "src/ directory exists" || print_error "Missing src/ directory"
    [[ -d "$project_path/tests" ]] && print_success "tests/ directory exists" || print_warning "Missing tests/ directory"

    # Check for .env.example
    if [[ -f "$project_path/.env.example" ]]; then
        print_success "Found .env.example"
    else
        print_warning "Missing .env.example"
    fi

    # Check .gitignore includes node_modules
    if grep -q "node_modules" "$project_path/.gitignore" 2>/dev/null; then
        print_success ".gitignore includes node_modules"
    else
        print_error ".gitignore missing node_modules"
    fi
}

check_python_structure() {
    print_header "Validating Python Project Structure"

    local project_path="$1"

    # Check for pyproject.toml or requirements.txt
    if [[ -f "$project_path/pyproject.toml" ]]; then
        print_success "Found pyproject.toml"
    elif [[ -f "$project_path/requirements.txt" ]]; then
        print_success "Found requirements.txt"
    else
        print_error "Missing pyproject.toml or requirements.txt"
    fi

    # Check directory structure
    [[ -d "$project_path/app" ]] || [[ -d "$project_path/src" ]] && \
        print_success "Source directory exists" || \
        print_error "Missing source directory (app/ or src/)"

    [[ -d "$project_path/tests" ]] && print_success "tests/ directory exists" || print_warning "Missing tests/ directory"

    # Check .gitignore includes __pycache__
    if grep -q "__pycache__" "$project_path/.gitignore" 2>/dev/null; then
        print_success ".gitignore includes __pycache__"
    else
        print_error ".gitignore missing __pycache__"
    fi

    # Check for .env.example
    if [[ -f "$project_path/.env.example" ]]; then
        print_success "Found .env.example"
    else
        print_warning "Missing .env.example"
    fi
}

check_go_structure() {
    print_header "Validating Go Project Structure"

    local project_path="$1"

    # Check go.mod
    if [[ -f "$project_path/go.mod" ]]; then
        print_success "Found go.mod"

        # Validate go version
        local go_version=$(grep "^go " "$project_path/go.mod" | awk '{print $2}')
        if [[ ! -z "$go_version" ]]; then
            print_success "Go version: $go_version"
        fi
    else
        print_error "Missing go.mod"
    fi

    # Check directory structure
    [[ -d "$project_path/cmd" ]] && print_success "cmd/ directory exists" || print_warning "Missing cmd/ directory"
    [[ -d "$project_path/internal" ]] && print_success "internal/ directory exists" || print_warning "Missing internal/ directory"

    # Check .gitignore
    if [[ -f "$project_path/.gitignore" ]]; then
        if grep -q "vendor" "$project_path/.gitignore" 2>/dev/null; then
            print_success ".gitignore includes vendor/"
        fi
    fi
}

check_security() {
    print_header "Security Checks"

    local project_path="$1"

    # Check for .env in .gitignore
    if grep -q "\.env" "$project_path/.gitignore" 2>/dev/null; then
        print_success ".env properly gitignored"
    else
        print_error ".env not in .gitignore"
    fi

    # Check for actual .env file (should not exist in template)
    if [[ -f "$project_path/.env" ]]; then
        print_error "Found .env file (should be .env.example only)"
    else
        print_success "No .env file in template"
    fi

    # Check for common secret files
    local secret_files=("id_rsa" "id_dsa" "*.pem" "*.key")
    for pattern in "${secret_files[@]}"; do
        if find "$project_path" -name "$pattern" -type f 2>/dev/null | grep -q .; then
            print_error "Found potential secret file: $pattern"
        fi
    done
}

check_documentation() {
    print_header "Documentation Checks"

    local project_path="$1"

    # Check README
    if [[ -f "$project_path/README.md" ]]; then
        local readme_lines=$(wc -l < "$project_path/README.md")
        if [[ $readme_lines -gt 10 ]]; then
            print_success "README.md has adequate content ($readme_lines lines)"
        else
            print_warning "README.md is very short ($readme_lines lines)"
        fi

        # Check for required sections
        if grep -q "## " "$project_path/README.md"; then
            print_success "README has sections"
        else
            print_warning "README missing sections"
        fi
    fi

    # Check for LICENSE
    if [[ -f "$project_path/LICENSE" ]]; then
        print_success "Found LICENSE file"
    else
        print_warning "No LICENSE file"
    fi
}

check_dependencies() {
    print_header "Dependency Checks"

    local project_path="$1"

    # Node.js dependency count
    if [[ -f "$project_path/package.json" ]] && command -v jq &> /dev/null; then
        local dep_count=$(jq -r '.dependencies | length' "$project_path/package.json" 2>/dev/null || echo "0")
        local dev_dep_count=$(jq -r '.devDependencies | length' "$project_path/package.json" 2>/dev/null || echo "0")

        if [[ $dep_count -lt 10 ]]; then
            print_success "Minimal dependencies: $dep_count production deps"
        else
            print_warning "Many dependencies: $dep_count production deps"
        fi

        print_success "Dev dependencies: $dev_dep_count"
    fi
}

detect_project_type() {
    local project_path="$1"

    if [[ -f "$project_path/package.json" ]]; then
        echo "node"
    elif [[ -f "$project_path/pyproject.toml" ]] || [[ -f "$project_path/requirements.txt" ]]; then
        echo "python"
    elif [[ -f "$project_path/go.mod" ]]; then
        echo "go"
    else
        echo "unknown"
    fi
}

# Main validation
main() {
    if [[ $# -lt 1 ]]; then
        echo "Usage: $0 <project_path>"
        exit 1
    fi

    local project_path="$1"

    if [[ ! -d "$project_path" ]]; then
        echo "Error: Directory $project_path does not exist"
        exit 1
    fi

    echo "Validating template: $project_path"

    # Detect project type
    local project_type=$(detect_project_type "$project_path")
    echo "Detected project type: $project_type"

    # Run common checks
    check_required_files "$project_path"
    check_security "$project_path"
    check_documentation "$project_path"

    # Run type-specific checks
    case "$project_type" in
        node)
            check_node_structure "$project_path"
            check_dependencies "$project_path"
            ;;
        python)
            check_python_structure "$project_path"
            ;;
        go)
            check_go_structure "$project_path"
            ;;
        *)
            print_warning "Unknown project type, skipping type-specific checks"
            ;;
    esac

    # Print summary
    print_header "Validation Summary"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"

    if [[ $FAILED -eq 0 ]]; then
        echo -e "\n${GREEN}✓ Template validation successful!${NC}"
        exit 0
    else
        echo -e "\n${RED}✗ Template validation failed with $FAILED errors${NC}"
        exit 1
    fi
}

main "$@"
