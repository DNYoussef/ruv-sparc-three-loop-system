#!/bin/bash
# Firewall Configuration Script for Network Security Setup
# Configures iptables/nftables rules for sandbox isolation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
FIREWALL_TYPE="${FIREWALL_TYPE:-iptables}" # iptables or nftables
CONFIG_FILE="${CONFIG_FILE:-/etc/network-security/trusted-domains.conf}"
LOG_FILE="${LOG_FILE:-/var/log/network-security/firewall.log}"
DRY_RUN="${DRY_RUN:-false}"

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."

    if [[ "$EUID" -ne 0 ]]; then
        log_error "This script must be run as root"
        exit 1
    fi

    if [[ "$FIREWALL_TYPE" == "iptables" ]]; then
        if ! command -v iptables &> /dev/null; then
            log_error "iptables is not installed"
            exit 1
        fi
    elif [[ "$FIREWALL_TYPE" == "nftables" ]]; then
        if ! command -v nft &> /dev/null; then
            log_error "nftables is not installed"
            exit 1
        fi
    else
        log_error "Invalid FIREWALL_TYPE: $FIREWALL_TYPE (must be 'iptables' or 'nftables')"
        exit 1
    fi

    log_success "Prerequisites validated"
}

# Parse trusted domains from config file
parse_trusted_domains() {
    log_info "Parsing trusted domains from $CONFIG_FILE..."

    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi

    # Extract domains, removing comments and empty lines
    mapfile -t TRUSTED_DOMAINS < <(grep -v '^#' "$CONFIG_FILE" | grep -v '^$' | sed 's/[[:space:]]*#.*//')

    log_info "Found ${#TRUSTED_DOMAINS[@]} trusted domains"
}

# Resolve domain to IP addresses
resolve_domain() {
    local domain="$1"

    # Handle wildcards by removing the wildcard prefix
    local clean_domain="${domain#\*.}"

    # Resolve using dig (more reliable than host for scripting)
    if command -v dig &> /dev/null; then
        dig +short "$clean_domain" A "$clean_domain" AAAA 2>/dev/null | grep -E '^[0-9a-f.:]+$'
    else
        # Fallback to getent
        getent hosts "$clean_domain" 2>/dev/null | awk '{print $1}'
    fi
}

# Configure iptables rules
configure_iptables() {
    log_info "Configuring iptables firewall rules..."

    # Flush existing rules (in DRY_RUN, just show what would be done)
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would flush existing rules"
    else
        iptables -F OUTPUT
        iptables -F INPUT
        log_info "Flushed existing rules"
    fi

    # Set default policies
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would set default DROP policy for OUTPUT chain"
    else
        iptables -P OUTPUT DROP
        log_info "Set default DROP policy for OUTPUT chain"
    fi

    # Allow loopback
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would allow loopback traffic"
    else
        iptables -A OUTPUT -o lo -j ACCEPT
        iptables -A INPUT -i lo -j ACCEPT
        log_info "Allowed loopback traffic"
    fi

    # Allow established connections
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would allow established/related connections"
    else
        iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
        iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
        log_info "Allowed established/related connections"
    fi

    # Add trusted domains
    local count=0
    for domain in "${TRUSTED_DOMAINS[@]}"; do
        log_info "Processing domain: $domain"

        # Resolve domain to IPs
        mapfile -t ips < <(resolve_domain "$domain")

        if [[ ${#ips[@]} -eq 0 ]]; then
            log_warn "Could not resolve domain: $domain"
            continue
        fi

        for ip in "${ips[@]}"; do
            if [[ "$DRY_RUN" == "true" ]]; then
                log_info "[DRY RUN] Would allow traffic to $ip (from $domain)"
            else
                iptables -A OUTPUT -d "$ip" -j ACCEPT
                log_info "Allowed traffic to $ip (from $domain)"
            fi
            ((count++))
        done
    done

    log_success "Configured $count IP rules for ${#TRUSTED_DOMAINS[@]} domains"
}

# Configure nftables rules
configure_nftables() {
    log_info "Configuring nftables firewall rules..."

    # Create nftables configuration
    local nft_config="/tmp/network-security-nftables.conf"

    cat > "$nft_config" <<'EOF'
#!/usr/sbin/nft -f

# Flush existing rules
flush ruleset

# Define table
table inet filter {
    # Define chain for output filtering
    chain output {
        type filter hook output priority 0; policy drop;

        # Allow loopback
        oifname "lo" accept

        # Allow established/related connections
        ct state established,related accept

        # Trusted domains (will be populated)
EOF

    # Add trusted domain IPs
    local count=0
    for domain in "${TRUSTED_DOMAINS[@]}"; do
        log_info "Processing domain: $domain"

        mapfile -t ips < <(resolve_domain "$domain")

        if [[ ${#ips[@]} -eq 0 ]]; then
            log_warn "Could not resolve domain: $domain"
            continue
        fi

        for ip in "${ips[@]}"; do
            echo "        ip daddr $ip accept # $domain" >> "$nft_config"
            ((count++))
        done
    done

    # Close the configuration
    cat >> "$nft_config" <<'EOF'
    }

    # Input chain (allow established connections)
    chain input {
        type filter hook input priority 0; policy accept;

        iifname "lo" accept
        ct state established,related accept
    }
}
EOF

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would apply nftables configuration:"
        cat "$nft_config"
    else
        nft -f "$nft_config"
        log_success "Applied nftables configuration with $count IP rules"
    fi

    rm -f "$nft_config"
}

# Verify firewall configuration
verify_configuration() {
    log_info "Verifying firewall configuration..."

    if [[ "$FIREWALL_TYPE" == "iptables" ]]; then
        log_info "Current iptables OUTPUT rules:"
        iptables -L OUTPUT -n -v
    elif [[ "$FIREWALL_TYPE" == "nftables" ]]; then
        log_info "Current nftables ruleset:"
        nft list ruleset
    fi
}

# Test connectivity to trusted domains
test_connectivity() {
    log_info "Testing connectivity to trusted domains..."

    local success_count=0
    local fail_count=0

    for domain in "${TRUSTED_DOMAINS[@]}"; do
        # Remove wildcard prefix for testing
        local test_domain="${domain#\*.}"

        if timeout 5 ping -c 1 "$test_domain" &> /dev/null; then
            log_success "✓ Connectivity to $test_domain: OK"
            ((success_count++))
        else
            log_warn "✗ Connectivity to $test_domain: FAILED"
            ((fail_count++))
        fi
    done

    log_info "Connectivity test results: $success_count passed, $fail_count failed"
}

# Save configuration for persistence
save_configuration() {
    log_info "Saving firewall configuration for persistence..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would save configuration"
        return
    fi

    if [[ "$FIREWALL_TYPE" == "iptables" ]]; then
        if command -v iptables-save &> /dev/null; then
            iptables-save > /etc/iptables/rules.v4
            log_success "Saved iptables configuration to /etc/iptables/rules.v4"
        else
            log_warn "iptables-save not found, configuration not persisted"
        fi
    elif [[ "$FIREWALL_TYPE" == "nftables" ]]; then
        nft list ruleset > /etc/nftables.conf
        log_success "Saved nftables configuration to /etc/nftables.conf"
    fi
}

# Main execution
main() {
    log_info "========================================="
    log_info "Network Security Firewall Configuration"
    log_info "========================================="
    log_info "Firewall Type: $FIREWALL_TYPE"
    log_info "Config File: $CONFIG_FILE"
    log_info "Dry Run: $DRY_RUN"
    log_info ""

    validate_prerequisites
    parse_trusted_domains

    if [[ "$FIREWALL_TYPE" == "iptables" ]]; then
        configure_iptables
    elif [[ "$FIREWALL_TYPE" == "nftables" ]]; then
        configure_nftables
    fi

    verify_configuration

    if [[ "$DRY_RUN" != "true" ]]; then
        test_connectivity
        save_configuration
    fi

    log_info ""
    log_success "========================================="
    log_success "Firewall configuration complete!"
    log_success "========================================="
}

# Run main function
main "$@"
