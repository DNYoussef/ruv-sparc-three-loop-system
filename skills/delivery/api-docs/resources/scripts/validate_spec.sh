#!/bin/bash
#
# OpenAPI Specification Validator
#
# Validates OpenAPI 3.0 specifications for compliance, completeness, and best practices.
# Uses multiple validation tools and custom checks.
#
# Usage:
#   ./validate_spec.sh openapi.yaml
#   ./validate_spec.sh openapi.json --strict
#   ./validate_spec.sh openapi.yaml --output report.txt

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SPEC_FILE=""
STRICT_MODE=false
OUTPUT_FILE=""
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --strict)
            STRICT_MODE=true
            shift
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 <spec-file> [options]"
            echo ""
            echo "Options:"
            echo "  --strict        Enable strict validation mode"
            echo "  --output, -o    Save validation report to file"
            echo "  --verbose, -v   Enable verbose output"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            if [[ -z "$SPEC_FILE" ]]; then
                SPEC_FILE="$1"
            fi
            shift
            ;;
    esac
done

# Check if spec file provided
if [[ -z "$SPEC_FILE" ]]; then
    echo -e "${RED}Error: OpenAPI specification file not provided${NC}"
    echo "Usage: $0 <spec-file>"
    exit 1
fi

# Check if file exists
if [[ ! -f "$SPEC_FILE" ]]; then
    echo -e "${RED}Error: File not found: $SPEC_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}=== OpenAPI Specification Validator ===${NC}"
echo -e "File: ${YELLOW}$SPEC_FILE${NC}"
echo ""

# Initialize validation results
ERRORS=0
WARNINGS=0
INFO=0

# Function to log messages
log_error() {
    ((ERRORS++))
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    ((WARNINGS++))
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_info() {
    ((INFO++))
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_verbose() {
    if [[ "$VERBOSE" = true ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Check file format
check_format() {
    echo -e "${BLUE}[1/7] Checking file format...${NC}"

    EXT="${SPEC_FILE##*.}"
    if [[ "$EXT" == "yaml" || "$EXT" == "yml" ]]; then
        log_verbose "Detected YAML format"

        # Validate YAML syntax
        if command -v yamllint &> /dev/null; then
            if yamllint -d relaxed "$SPEC_FILE" &> /dev/null; then
                log_info "Valid YAML syntax"
            else
                log_error "Invalid YAML syntax"
                return 1
            fi
        else
            log_warning "yamllint not installed, skipping YAML validation"
        fi

    elif [[ "$EXT" == "json" ]]; then
        log_verbose "Detected JSON format"

        # Validate JSON syntax
        if command -v jq &> /dev/null; then
            if jq empty "$SPEC_FILE" &> /dev/null; then
                log_info "Valid JSON syntax"
            else
                log_error "Invalid JSON syntax"
                return 1
            fi
        else
            log_warning "jq not installed, skipping JSON validation"
        fi
    else
        log_error "Unsupported file format: $EXT (expected .yaml, .yml, or .json)"
        return 1
    fi
}

# Check OpenAPI version
check_version() {
    echo -e "\n${BLUE}[2/7] Checking OpenAPI version...${NC}"

    if command -v yq &> /dev/null; then
        VERSION=$(yq eval '.openapi' "$SPEC_FILE" 2>/dev/null)

        if [[ "$VERSION" =~ ^3\.0\. ]]; then
            log_info "OpenAPI version: $VERSION"
        elif [[ "$VERSION" =~ ^3\.1\. ]]; then
            log_info "OpenAPI version: $VERSION (3.1.x detected)"
            if [[ "$STRICT_MODE" = true ]]; then
                log_warning "Target version is 3.0.x, but found 3.1.x"
            fi
        else
            log_error "Unsupported OpenAPI version: $VERSION (expected 3.0.x)"
        fi
    else
        log_warning "yq not installed, skipping version check"
    fi
}

# Validate with Swagger CLI
validate_swagger() {
    echo -e "\n${BLUE}[3/7] Validating with Swagger CLI...${NC}"

    if command -v swagger-cli &> /dev/null; then
        if swagger-cli validate "$SPEC_FILE" 2>&1 | tee /tmp/swagger_output.txt; then
            log_info "Swagger CLI validation passed"
        else
            log_error "Swagger CLI validation failed"
            cat /tmp/swagger_output.txt
        fi
    else
        log_warning "swagger-cli not installed, skipping validation"
        log_info "Install with: npm install -g @apidevtools/swagger-cli"
    fi
}

# Validate with OpenAPI Generator
validate_openapi_generator() {
    echo -e "\n${BLUE}[4/7] Validating with OpenAPI Generator...${NC}"

    if command -v openapi-generator-cli &> /dev/null; then
        if openapi-generator-cli validate -i "$SPEC_FILE" 2>&1 | tee /tmp/openapi_gen_output.txt; then
            log_info "OpenAPI Generator validation passed"
        else
            log_error "OpenAPI Generator validation failed"
            cat /tmp/openapi_gen_output.txt
        fi
    else
        log_warning "openapi-generator-cli not installed, skipping validation"
        log_info "Install with: npm install -g @openapitools/openapi-generator-cli"
    fi
}

# Check required fields
check_required_fields() {
    echo -e "\n${BLUE}[5/7] Checking required fields...${NC}"

    if ! command -v yq &> /dev/null; then
        log_warning "yq not installed, skipping required fields check"
        return
    fi

    # Check info section
    TITLE=$(yq eval '.info.title' "$SPEC_FILE" 2>/dev/null)
    VERSION=$(yq eval '.info.version' "$SPEC_FILE" 2>/dev/null)

    if [[ "$TITLE" == "null" || -z "$TITLE" ]]; then
        log_error "Missing required field: info.title"
    else
        log_verbose "info.title: $TITLE"
    fi

    if [[ "$VERSION" == "null" || -z "$VERSION" ]]; then
        log_error "Missing required field: info.version"
    else
        log_verbose "info.version: $VERSION"
    fi

    # Check paths
    PATHS_COUNT=$(yq eval '.paths | length' "$SPEC_FILE" 2>/dev/null)
    if [[ "$PATHS_COUNT" == "0" || "$PATHS_COUNT" == "null" ]]; then
        if [[ "$STRICT_MODE" = true ]]; then
            log_error "No paths defined in specification"
        else
            log_warning "No paths defined in specification"
        fi
    else
        log_info "Found $PATHS_COUNT path(s)"
    fi
}

# Check best practices
check_best_practices() {
    echo -e "\n${BLUE}[6/7] Checking best practices...${NC}"

    if ! command -v yq &> /dev/null; then
        log_warning "yq not installed, skipping best practices check"
        return
    fi

    # Check for description
    DESCRIPTION=$(yq eval '.info.description' "$SPEC_FILE" 2>/dev/null)
    if [[ "$DESCRIPTION" == "null" || -z "$DESCRIPTION" ]]; then
        log_warning "Missing recommended field: info.description"
    else
        log_verbose "info.description present"
    fi

    # Check for servers
    SERVERS_COUNT=$(yq eval '.servers | length' "$SPEC_FILE" 2>/dev/null)
    if [[ "$SERVERS_COUNT" == "0" || "$SERVERS_COUNT" == "null" ]]; then
        log_warning "No servers defined"
    else
        log_info "Found $SERVERS_COUNT server(s)"
    fi

    # Check for security schemes
    SECURITY_SCHEMES=$(yq eval '.components.securitySchemes | length' "$SPEC_FILE" 2>/dev/null)
    if [[ "$SECURITY_SCHEMES" == "0" || "$SECURITY_SCHEMES" == "null" ]]; then
        log_warning "No security schemes defined"
    else
        log_info "Found $SECURITY_SCHEMES security scheme(s)"
    fi

    # Check for tags
    TAGS_COUNT=$(yq eval '.tags | length' "$SPEC_FILE" 2>/dev/null)
    if [[ "$TAGS_COUNT" == "0" || "$TAGS_COUNT" == "null" ]]; then
        log_warning "No tags defined (recommended for organization)"
    else
        log_info "Found $TAGS_COUNT tag(s)"
    fi

    # Check for examples
    if [[ "$STRICT_MODE" = true ]]; then
        log_verbose "Checking for examples in responses..."
        # This is a simplified check; a full implementation would traverse all paths
        if ! grep -q "example" "$SPEC_FILE"; then
            log_warning "No examples found in specification"
        fi
    fi
}

# Generate validation report
generate_report() {
    echo -e "\n${BLUE}[7/7] Generating validation report...${NC}"
    echo ""

    REPORT="=== OpenAPI Validation Report ===\n"
    REPORT+="File: $SPEC_FILE\n"
    REPORT+="Date: $(date '+%Y-%m-%d %H:%M:%S')\n"
    REPORT+="Strict Mode: $STRICT_MODE\n\n"
    REPORT+="Results:\n"
    REPORT+="  Errors:   $ERRORS\n"
    REPORT+="  Warnings: $WARNINGS\n"
    REPORT+="  Info:     $INFO\n\n"

    if [[ $ERRORS -eq 0 ]]; then
        REPORT+="Status: ${GREEN}PASSED${NC}\n"
        echo -e "${GREEN}✓ Validation PASSED${NC}"
    else
        REPORT+="Status: ${RED}FAILED${NC}\n"
        echo -e "${RED}✗ Validation FAILED${NC}"
    fi

    if [[ -n "$OUTPUT_FILE" ]]; then
        echo -e "$REPORT" > "$OUTPUT_FILE"
        echo -e "\nReport saved to: ${YELLOW}$OUTPUT_FILE${NC}"
    fi
}

# Run all checks
check_format
check_version
validate_swagger
validate_openapi_generator
check_required_fields
check_best_practices
generate_report

# Exit with appropriate code
if [[ $ERRORS -gt 0 ]]; then
    exit 1
elif [[ "$STRICT_MODE" = true && $WARNINGS -gt 0 ]]; then
    exit 1
else
    exit 0
fi
