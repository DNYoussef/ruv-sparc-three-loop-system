# Network Security Setup - Enhanced Tier

## Overview

The **network-security-setup** skill provides comprehensive network isolation and security configuration for Claude Code sandboxes. This Enhanced tier implementation includes production-ready scripts, templates, tests, and examples for enterprise-grade network security.

## 📁 Structure

```
network-security-setup/
├── skill.md                           # Main skill definition (11KB)
├── README.md                          # This file
├── resources/
│   ├── scripts/
│   │   ├── firewall-config.sh         # Firewall configuration (iptables/nftables)
│   │   ├── ssl-setup.py               # SSL/TLS certificate management
│   │   ├── network-scanner.js         # Network connectivity scanner
│   │   └── security-audit.py          # Security audit and compliance
│   └── templates/
│       ├── security-policy.yaml       # Security policy template
│       ├── network-rules.json         # Network rules configuration
│       └── ssl-config.yaml            # SSL/TLS configuration
├── tests/
│   ├── test-firewall-config.sh        # Firewall script tests
│   ├── test-ssl-setup.py              # SSL setup tests
│   └── test-network-scanner.js        # Network scanner tests
└── examples/
    ├── sandbox-isolation-example.js   # Complete isolation setup (200+ lines)
    ├── trusted-domains-example.sh     # Domain management (250+ lines)
    └── security-policies-example.py   # Policy configuration (300+ lines)
```

## 🚀 Quick Start

### 1. Basic Network Isolation

```bash
# Configure trusted domains
cat > /etc/network-security/trusted-domains.conf <<EOF
*.npmjs.org
registry.npmjs.org
*.github.com
api.github.com
EOF

# Setup firewall
bash resources/scripts/firewall-config.sh
```

### 2. SSL/TLS Setup

```bash
# Generate CA certificate
python3 resources/scripts/ssl-setup.py \
  --action generate-ca \
  --common-name "Network Security CA" \
  --validity-days 3650

# Generate server certificate
python3 resources/scripts/ssl-setup.py \
  --action generate-server \
  --common-name "sandbox.local" \
  --san-domains "*.sandbox.local" "localhost"
```

### 3. Network Scanning

```bash
# Scan trusted domains
node resources/scripts/network-scanner.js \
  --verbose \
  --output scan-results.json
```

### 4. Security Audit

```bash
# Run comprehensive audit
python3 resources/scripts/security-audit.py \
  --config-dir /etc/network-security \
  --output audit-report.json
```

## 📚 Examples

### Example 1: Sandbox Isolation (200+ lines)

Complete network isolation setup with monitoring:

```javascript
const { SandboxIsolationManager } = require('./examples/sandbox-isolation-example.js');

const manager = new SandboxIsolationManager(config);
await manager.initialize();
await manager.testConnectivity();
```

**Features:**
- Automated directory setup
- Trusted domain configuration
- Firewall rule deployment
- Real-time monitoring
- Metrics collection
- Security reporting

### Example 2: Trusted Domains Management (250+ lines)

Interactive domain whitelist management:

```bash
bash examples/trusted-domains-example.sh
```

**Features:**
- Domain validation
- DNS resolution caching
- Connectivity testing
- Security auditing
- Interactive CLI
- Automated reports

### Example 3: Security Policies (300+ lines)

Enterprise security policy engine:

```python
from examples.security_policies_example import SecurityPolicyManager

manager = SecurityPolicyManager(config_dir)
policy = manager.create_network_isolation_policy()
manager.test_policy(policy)
```

**Features:**
- Rule-based access control
- Policy validation
- Compliance checking
- Test scenarios
- Report generation
- JSON export

## 🧪 Testing

### Run All Tests

```bash
# Firewall configuration tests
bash tests/test-firewall-config.sh

# SSL setup tests
python3 tests/test-ssl-setup.py

# Network scanner tests
node tests/test-network-scanner.js
```

### Test Coverage

- **Firewall Configuration**: 7 tests
  - Script existence
  - Config parsing
  - Dry run mode
  - Domain resolution
  - Log file creation
  - Error handling
  - Firewall type validation

- **SSL Setup**: 10 tests
  - Directory creation
  - Key generation
  - CA certificate creation
  - Server certificate creation
  - Certificate saving
  - Certificate validation
  - Expiration checking
  - Dry run mode
  - CLI functionality

- **Network Scanner**: 7 tests
  - Script existence
  - Config parsing
  - Domain resolution
  - HTTP connectivity
  - Results export
  - CLI interface
  - Error handling

## 📋 Templates

### Security Policy Template

Complete YAML-based security policy with:
- Network isolation modes (none, trusted, custom)
- Trusted domain whitelist
- Blocked domain blacklist
- Network rules (allow/deny)
- Proxy configuration
- Logging and monitoring
- Rate limiting
- Compliance settings

### Network Rules Template

JSON Schema-based firewall rules:
- Rule priorities and ordering
- Source/destination filtering
- Protocol and port specifications
- State-based rules (NEW, ESTABLISHED, RELATED)
- Time-based activation
- Rate limiting
- Logging configuration

### SSL Configuration Template

Comprehensive SSL/TLS settings:
- CA and certificate configuration
- Protocol versions (TLS 1.2+)
- Cipher suites (modern, secure)
- Certificate validation
- Auto-renewal settings
- Trust store management
- OCSP and HSTS
- Security headers

## 🛠️ Resources

### Scripts

#### 1. `firewall-config.sh` (450 lines)

Automated firewall configuration for iptables/nftables:

**Features:**
- Dual firewall support (iptables and nftables)
- Domain to IP resolution
- Rule generation from config
- Dry run mode
- Comprehensive logging
- Configuration persistence
- Connectivity testing

**Usage:**
```bash
FIREWALL_TYPE=iptables \
CONFIG_FILE=/etc/network-security/trusted-domains.conf \
DRY_RUN=false \
bash resources/scripts/firewall-config.sh
```

#### 2. `ssl-setup.py` (500 lines)

SSL/TLS certificate management:

**Features:**
- CA certificate generation
- Server certificate creation
- Certificate validation
- Multi-platform trust store installation
- SAN (Subject Alternative Names)
- Certificate expiration checking
- Dry run mode

**Usage:**
```bash
python3 resources/scripts/ssl-setup.py \
  --action generate-ca \
  --common-name "My CA" \
  --validity-days 3650
```

#### 3. `network-scanner.js` (450 lines)

Network connectivity scanner:

**Features:**
- DNS resolution with caching
- HTTP/HTTPS connectivity testing
- TCP port scanning
- ICMP ping testing
- Parallel batch processing
- JSON results export
- Blocked domain detection

**Usage:**
```bash
CONFIG_FILE=/etc/network-security/trusted-domains.conf \
TIMEOUT=5000 \
PARALLEL=5 \
node resources/scripts/network-scanner.js --verbose --output results.json
```

#### 4. `security-audit.py` (550 lines)

Comprehensive security audit:

**Features:**
- Configuration file validation
- Firewall rule auditing
- SSL certificate checking
- Environment variable scanning
- Compliance scoring
- Severity-based findings
- JSON report generation

**Usage:**
```bash
python3 resources/scripts/security-audit.py \
  --config-dir /etc/network-security \
  --output audit-report.json
```

## 🔒 Security Features

### Network Isolation Modes

1. **Trusted Mode (Recommended)**
   - Pre-approved domain whitelist
   - Deny by default
   - Explicit allow rules

2. **No Network Access**
   - Complete isolation
   - Offline development
   - Maximum security

3. **Custom Access**
   - User-defined whitelist
   - Enterprise registries
   - Internal networks

### Threat Mitigation

- **Prompt Injection → Data Exfiltration**: Blocked by domain whitelist
- **Malicious Package Download**: Only trusted registries allowed
- **Internal Network Scanning**: Network isolation prevents scanning
- **Credential Theft**: Secrets not in sandbox config, network blocked

### Default Trusted Domains

**Package Registries:**
- `*.npmjs.org`, `*.yarnpkg.com`, `*.pypi.org`

**Container Registries:**
- `*.docker.io`, `ghcr.io`, `gcr.io`

**Source Control:**
- `*.github.com`, `*.gitlab.com`

**CDNs:**
- `*.cloudfront.net`, `cdn.jsdelivr.net`

## 📊 Metrics & Monitoring

### Available Metrics

- Connection attempts (total, allowed, blocked)
- Success rate percentage
- DNS resolution cache hits
- Firewall rule matches
- Certificate expiration warnings
- Security audit findings

### Logging Levels

- **DEBUG**: Detailed execution trace
- **INFO**: Standard operational messages
- **WARN**: Potential issues or anomalies
- **ERROR**: Failures requiring attention

## 🏢 Enterprise Features

### Corporate Proxy Support

```json
{
  "sandbox": {
    "network": {
      "customProxy": {
        "enabled": true,
        "http": "http://proxy.company.com:8080",
        "https": "http://proxy.company.com:8080",
        "noProxy": ["localhost", "*.internal"]
      }
    }
  }
}
```

### Internal Registry Configuration

```json
{
  "sandbox": {
    "network": {
      "trustedDomains": [
        "registry.company.internal:5000",
        "npm.company.com",
        "docs.company.com"
      ]
    }
  }
}
```

### Compliance Standards

- **PCI DSS 3.2.1**: Payment card industry compliance
- **NIST SP 800-52**: TLS configuration guidelines
- **CIS Benchmarks**: Security best practices
- **FIPS 140-2**: Cryptographic module validation

## 🔧 Configuration

### Firewall Configuration

```bash
# /etc/network-security/trusted-domains.conf
*.npmjs.org
registry.npmjs.org
*.github.com
api.github.com
```

### SSL Configuration

```yaml
# /etc/network-security/ssl-config.yaml
spec:
  protocols:
    min_version: "TLSv1.2"
    allowed_versions: ["TLSv1.2", "TLSv1.3"]

  cipher_suites:
    preferred:
      - "TLS_AES_256_GCM_SHA384"
      - "TLS_CHACHA20_POLY1305_SHA256"
```

### Security Policy

```yaml
# /etc/network-security/security-policy.yaml
spec:
  isolation:
    mode: trusted
    default_action: deny

  trusted_domains:
    - domain: "*.npmjs.org"
      protocols: ["https"]
      ports: [443]
```

## 📖 Documentation

### Main Skill Document

See `skill.md` for:
- Complete methodology
- Network isolation modes
- Enterprise configuration
- Environment variable management
- Security threat mitigation
- Validation checklists

### API Reference

All scripts provide `--help` documentation:

```bash
bash resources/scripts/firewall-config.sh --help
python3 resources/scripts/ssl-setup.py --help
node resources/scripts/network-scanner.js --help
python3 resources/scripts/security-audit.py --help
```

## 🎯 Use Cases

### 1. Open Source Development

Standard npm/GitHub access with security isolation.

### 2. Enterprise Internal

Corporate proxy, internal registries, compliance requirements.

### 3. Maximum Security

Complete network isolation for sensitive projects.

### 4. Development/Staging

Balanced security with debugging capabilities.

## 🚦 Status

- **Tier**: Enhanced
- **Status**: Production Ready
- **Version**: 1.0.0
- **Last Updated**: 2025-11-02

## 🤝 Integration

### With Other Skills

- **sandbox-configurator**: Complete sandbox security setup
- **security-review**: Security audit workflows
- **cicd-intelligent-recovery**: Automated deployment security

### With MCP Tools

- Flow-Nexus sandbox creation
- Network security validation
- Real-time monitoring

## 📝 License

This skill is part of the SPARC Three-Loop System and follows the project's licensing terms.

## 🔗 References

- [SPARC Methodology](../../README.md)
- [Claude Flow Documentation](https://github.com/ruvnet/claude-flow)
- [Network Security Best Practices](https://www.nist.gov/cybersecurity)
- [TLS Configuration Guidelines](https://www.nist.gov/publications/sp-800-52)

---

**Security Principle**: Deny by default, allow explicitly, verify continuously.
