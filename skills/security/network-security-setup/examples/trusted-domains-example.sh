#!/bin/bash
# Example: Trusted Domains Configuration and Management
#
# This comprehensive example demonstrates:
# - Trusted domain whitelist management
# - Domain validation and testing
# - Dynamic domain addition/removal
# - Domain pattern matching
# - DNS resolution and caching
#
# Use Case: Managing trusted domain whitelist for network security

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="/tmp/network-security-example"
TRUSTED_DOMAINS_FILE="$CONFIG_DIR/trusted-domains.conf"
DNS_CACHE_FILE="$CONFIG_DIR/dns-cache.json"
LOG_FILE="$CONFIG_DIR/trusted-domains.log"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[${timestamp}] [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() { log "${BLUE}INFO${NC}" "$@"; }
log_success() { log "${GREEN}SUCCESS${NC}" "$@"; }
log_warn() { log "${YELLOW}WARN${NC}" "$@"; }
log_error() { log "${RED}ERROR${NC}" "$@"; }

# Initialize environment
initialize() {
    log_info "Initializing trusted domains environment..."

    # Create directories
    mkdir -p "$CONFIG_DIR"

    # Initialize trusted domains file
    cat > "$TRUSTED_DOMAINS_FILE" <<'EOF'
# Trusted Domains Configuration
# One domain per line, wildcards supported

# Package Registries
*.npmjs.org
registry.npmjs.org
*.yarnpkg.com
*.pypi.org
pypi.python.org

# Container Registries
*.docker.io
registry.hub.docker.com
ghcr.io
gcr.io

# Source Control
*.github.com
api.github.com
raw.githubusercontent.com
*.gitlab.com

# CDNs
*.cloudfront.net
cdn.jsdelivr.net
unpkg.com

# Development Tools
*.vercel.com
*.netlify.com
*.supabase.co
EOF

    # Initialize DNS cache
    echo '{}' > "$DNS_CACHE_FILE"

    log_success "Environment initialized"
}

# Parse trusted domains
parse_trusted_domains() {
    log_info "Parsing trusted domains..."

    # Extract domains, removing comments and empty lines
    mapfile -t DOMAINS < <(grep -v '^#' "$TRUSTED_DOMAINS_FILE" | grep -v '^$' | sed 's/[[:space:]]*#.*//')

    log_success "Parsed ${#DOMAINS[@]} trusted domains"
}

# Validate domain pattern
validate_domain_pattern() {
    local domain="$1"

    # Check for valid domain pattern
    if [[ ! "$domain" =~ ^(\*\.)?[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$ ]]; then
        log_error "Invalid domain pattern: $domain"
        return 1
    fi

    # Check for overly permissive wildcards
    if [[ "$domain" =~ \*\.\*\. ]]; then
        log_warn "Domain contains multiple wildcards: $domain"
    fi

    return 0
}

# Resolve domain to IP addresses
resolve_domain() {
    local domain="$1"
    local clean_domain="${domain#\*.}"

    log_info "Resolving domain: $clean_domain"

    # Check DNS cache first
    if command -v jq &> /dev/null; then
        local cached=$(jq -r ".\"$clean_domain\" // empty" "$DNS_CACHE_FILE" 2>/dev/null || echo "")

        if [[ -n "$cached" ]]; then
            log_info "Using cached DNS result for $clean_domain"
            echo "$cached"
            return 0
        fi
    fi

    # Resolve using dig if available
    if command -v dig &> /dev/null; then
        local ips=$(dig +short "$clean_domain" A "$clean_domain" AAAA 2>/dev/null | grep -E '^[0-9a-f.:]+$' | head -5)

        if [[ -n "$ips" ]]; then
            log_success "Resolved $clean_domain to: $(echo "$ips" | tr '\n' ', ' | sed 's/,$//')"

            # Cache result
            if command -v jq &> /dev/null; then
                local cache=$(cat "$DNS_CACHE_FILE")
                echo "$cache" | jq ". + {\"$clean_domain\": \"$(echo "$ips" | tr '\n' ',' | sed 's/,$//')\"}" > "$DNS_CACHE_FILE"
            fi

            echo "$ips"
            return 0
        fi
    fi

    # Fallback to getent
    if command -v getent &> /dev/null; then
        local ips=$(getent hosts "$clean_domain" 2>/dev/null | awk '{print $1}' | head -5)

        if [[ -n "$ips" ]]; then
            log_success "Resolved $clean_domain using getent"
            echo "$ips"
            return 0
        fi
    fi

    log_warn "Failed to resolve $clean_domain"
    return 1
}

# Test domain connectivity
test_domain_connectivity() {
    local domain="$1"
    local clean_domain="${domain#\*.}"

    log_info "Testing connectivity to $clean_domain..."

    # Try HTTPS first
    if curl -s -o /dev/null -w "%{http_code}" --max-time 5 "https://$clean_domain" > /dev/null 2>&1; then
        log_success "✓ $clean_domain is accessible via HTTPS"
        return 0
    fi

    # Try ping
    if ping -c 1 -W 3 "$clean_domain" > /dev/null 2>&1; then
        log_success "✓ $clean_domain responds to ping"
        return 0
    fi

    log_warn "✗ $clean_domain is not accessible"
    return 1
}

# Add trusted domain
add_trusted_domain() {
    local domain="$1"

    log_info "Adding trusted domain: $domain"

    # Validate domain pattern
    if ! validate_domain_pattern "$domain"; then
        log_error "Failed to add domain: invalid pattern"
        return 1
    fi

    # Check if domain already exists
    if grep -qxF "$domain" "$TRUSTED_DOMAINS_FILE" 2>/dev/null; then
        log_warn "Domain already in trusted list: $domain"
        return 0
    fi

    # Add domain
    echo "$domain" >> "$TRUSTED_DOMAINS_FILE"
    log_success "Added trusted domain: $domain"

    # Test connectivity
    test_domain_connectivity "$domain" || true
}

# Remove trusted domain
remove_trusted_domain() {
    local domain="$1"

    log_info "Removing trusted domain: $domain"

    # Create temporary file
    local temp_file=$(mktemp)

    # Filter out the domain
    grep -vxF "$domain" "$TRUSTED_DOMAINS_FILE" > "$temp_file" || true

    # Replace original file
    mv "$temp_file" "$TRUSTED_DOMAINS_FILE"

    log_success "Removed trusted domain: $domain"
}

# List trusted domains with details
list_trusted_domains() {
    log_info "Listing trusted domains with details..."

    echo ""
    echo "==========================================="
    echo "Trusted Domains Summary"
    echo "==========================================="

    local count=0
    local accessible=0
    local inaccessible=0

    while IFS= read -r domain; do
        # Skip comments and empty lines
        [[ "$domain" =~ ^#.*$ ]] && continue
        [[ -z "$domain" ]] && continue

        ((count++))

        echo ""
        echo "Domain #$count: $domain"

        # Validate pattern
        if validate_domain_pattern "$domain"; then
            echo "  Pattern: ✓ Valid"
        else
            echo "  Pattern: ✗ Invalid"
            continue
        fi

        # Resolve DNS
        local ips=$(resolve_domain "$domain" 2>/dev/null || echo "")
        if [[ -n "$ips" ]]; then
            echo "  DNS: ✓ Resolved"
            echo "  IPs: $(echo "$ips" | tr '\n' ', ' | sed 's/,$//')"
        else
            echo "  DNS: ✗ Failed to resolve"
        fi

        # Test connectivity
        if test_domain_connectivity "$domain" 2>/dev/null; then
            echo "  Connectivity: ✓ Accessible"
            ((accessible++))
        else
            echo "  Connectivity: ✗ Not accessible"
            ((inaccessible++))
        fi

    done < "$TRUSTED_DOMAINS_FILE"

    echo ""
    echo "==========================================="
    echo "Summary"
    echo "==========================================="
    echo "Total Domains: $count"
    echo "Accessible: $accessible"
    echo "Inaccessible: $inaccessible"
    echo "==========================================="
}

# Audit trusted domains
audit_trusted_domains() {
    log_info "Auditing trusted domains for security issues..."

    local issues=0

    while IFS= read -r domain; do
        # Skip comments and empty lines
        [[ "$domain" =~ ^#.*$ ]] && continue
        [[ -z "$domain" ]] && continue

        # Check for top-level wildcards
        if [[ "$domain" =~ ^\*\.[^.]+$ ]]; then
            log_warn "Security Issue: Top-level wildcard domain: $domain"
            ((issues++))
        fi

        # Check for multiple wildcards
        if [[ $(echo "$domain" | grep -o '\*' | wc -l) -gt 1 ]]; then
            log_warn "Security Issue: Multiple wildcards: $domain"
            ((issues++))
        fi

        # Check for IP addresses (should use hostnames)
        if [[ "$domain" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            log_warn "Security Issue: IP address instead of hostname: $domain"
            ((issues++))
        fi

        # Check for non-standard TLDs
        local tld="${domain##*.}"
        if [[ ! "$tld" =~ ^(com|org|net|io|dev|co|cloud)$ ]] && [[ "$tld" != "$domain" ]]; then
            log_info "Non-standard TLD: $domain (.$tld)"
        fi

    done < "$TRUSTED_DOMAINS_FILE"

    if [[ $issues -eq 0 ]]; then
        log_success "Audit complete: No security issues found"
    else
        log_warn "Audit complete: Found $issues security issues"
    fi

    return $issues
}

# Generate security report
generate_security_report() {
    log_info "Generating security report..."

    local report_file="$CONFIG_DIR/security-report-$(date +%Y%m%d-%H%M%S).txt"

    {
        echo "Trusted Domains Security Report"
        echo "Generated: $(date)"
        echo ""
        echo "Configuration File: $TRUSTED_DOMAINS_FILE"
        echo ""
        echo "====================================="
        echo "Statistics"
        echo "====================================="

        local total=$(grep -v '^#' "$TRUSTED_DOMAINS_FILE" | grep -v '^$' | wc -l)
        local wildcards=$(grep -v '^#' "$TRUSTED_DOMAINS_FILE" | grep '\*' | wc -l)
        local specific=$(grep -v '^#' "$TRUSTED_DOMAINS_FILE" | grep -v '\*' | wc -l)

        echo "Total Domains: $total"
        echo "Wildcard Domains: $wildcards"
        echo "Specific Domains: $specific"
        echo ""

        echo "====================================="
        echo "Audit Results"
        echo "====================================="

        audit_trusted_domains 2>&1

    } | tee "$report_file"

    log_success "Security report saved to: $report_file"
}

# Main menu
show_menu() {
    echo ""
    echo "==========================================="
    echo "Trusted Domains Management"
    echo "==========================================="
    echo "1. List all trusted domains"
    echo "2. Add trusted domain"
    echo "3. Remove trusted domain"
    echo "4. Test domain connectivity"
    echo "5. Audit trusted domains"
    echo "6. Generate security report"
    echo "7. Exit"
    echo "==========================================="
}

# Interactive mode
interactive_mode() {
    while true; do
        show_menu
        read -p "Select an option: " choice

        case $choice in
            1)
                list_trusted_domains
                ;;
            2)
                read -p "Enter domain to add: " domain
                add_trusted_domain "$domain"
                ;;
            3)
                read -p "Enter domain to remove: " domain
                remove_trusted_domain "$domain"
                ;;
            4)
                read -p "Enter domain to test: " domain
                test_domain_connectivity "$domain"
                ;;
            5)
                audit_trusted_domains
                ;;
            6)
                generate_security_report
                ;;
            7)
                log_info "Exiting..."
                exit 0
                ;;
            *)
                log_error "Invalid option"
                ;;
        esac
    done
}

# Main execution
main() {
    echo "==========================================="
    echo "Trusted Domains Example"
    echo "==========================================="

    initialize
    parse_trusted_domains

    # Run automated examples
    log_info "Running automated examples..."

    # Example 1: List domains
    echo ""
    log_info "Example 1: List all trusted domains"
    list_trusted_domains

    # Example 2: Add domain
    echo ""
    log_info "Example 2: Add new trusted domain"
    add_trusted_domain "api.example.com"

    # Example 3: Audit domains
    echo ""
    log_info "Example 3: Audit trusted domains"
    audit_trusted_domains || true

    # Example 4: Generate report
    echo ""
    log_info "Example 4: Generate security report"
    generate_security_report

    # Prompt for interactive mode
    echo ""
    read -p "Enter interactive mode? (y/n): " interactive

    if [[ "$interactive" == "y" ]]; then
        interactive_mode
    fi

    log_success "Example completed successfully"
}

# Run main function
main "$@"
