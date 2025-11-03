# Sandbox Configurator Resources

## Overview
Production-ready resources for configuring E2B sandboxes, Docker containers, and VM environments with comprehensive security isolation.

## Directory Structure

```
resources/
├── scripts/           # Production scripts for sandbox configuration
│   ├── e2b-setup.py              # E2B sandbox initialization (Python)
│   ├── docker-configurator.sh   # Docker container setup (Bash)
│   ├── network-isolation.py     # Network policy configuration (Python)
│   └── resource-limiter.js      # CPU/memory constraints (Node.js)
│
└── templates/         # Configuration templates
    ├── e2b-config.yaml          # E2B sandbox template
    ├── docker-compose.yml       # Multi-container setup
    └── security-policy.json     # Sandbox security config
```

## Scripts

### e2b-setup.py
**Purpose**: Initialize and configure E2B sandboxes with security isolation
**Language**: Python 3.10+
**Dependencies**: `e2b`, `pyyaml`, `requests`
**Usage**:
```bash
python e2b-setup.py --template nodejs --timeout 3600 --env-file .env.sandbox
```

**Features**:
- Template-based sandbox creation (node, python, react, nextjs, claude-code)
- Environment variable injection with validation
- Network policy configuration
- Resource limit enforcement
- Health check monitoring
- Automatic cleanup on failure

### docker-configurator.sh
**Purpose**: Configure Docker containers with security best practices
**Language**: Bash
**Dependencies**: `docker`, `jq`
**Usage**:
```bash
./docker-configurator.sh --config security-policy.json --network isolated
```

**Features**:
- Security profile configuration (AppArmor/SELinux)
- Read-only root filesystem setup
- Capability dropping (CAP_NET_RAW, CAP_SYS_ADMIN)
- Network namespace isolation
- User namespace remapping
- Resource constraints (CPU, memory, PIDs)
- Tmpfs mount for temporary data

### network-isolation.py
**Purpose**: Configure network policies for sandbox isolation
**Language**: Python 3.10+
**Dependencies**: `netfilter`, `iptables`, `pyyaml`
**Usage**:
```bash
python network-isolation.py --mode whitelist --trusted-domains npmjs.org,github.com
```

**Features**:
- Whitelist/blacklist mode for domain filtering
- DNS-based domain resolution
- iptables rule generation
- Local binding control (127.0.0.1:*)
- Unix socket permission management
- Network namespace creation
- Egress traffic filtering

### resource-limiter.js
**Purpose**: Enforce CPU and memory limits for sandbox processes
**Language**: Node.js
**Dependencies**: Native Node.js modules
**Usage**:
```bash
node resource-limiter.js --cpu 2 --memory 2048 --pids 100
```

**Features**:
- cgroups v2 integration
- CPU quota enforcement (percentage or cores)
- Memory limit with swap control
- PID limit to prevent fork bombs
- I/O throttling (disk read/write)
- Real-time monitoring
- Alert thresholds
- Graceful degradation

## Templates

### e2b-config.yaml
E2B sandbox configuration template with:
- Template selection (nodejs, python, react, nextjs, claude-code)
- Environment variable definitions
- Network policy configuration
- Timeout settings
- Package installation lists
- Startup script commands

### docker-compose.yml
Multi-container Docker setup with:
- Service definitions for sandbox runtime
- Network configuration (bridge, isolated)
- Volume mounts (read-only where possible)
- Security options (AppArmor, seccomp, capabilities)
- Resource limits per service
- Health checks

### security-policy.json
Comprehensive security policy defining:
- Sandbox mode (enabled, auto-allow settings)
- Excluded commands (git, docker, etc.)
- Network configuration
  - Trusted domains
  - Local binding permissions
  - Unix socket access
- Environment variable schema
- File system permissions
- Command execution policies

## Usage Patterns

### Pattern 1: E2B Sandbox for Claude Code
```bash
# 1. Configure security policy
cp templates/security-policy.json .claude/settings.local.json

# 2. Initialize E2B sandbox
python scripts/e2b-setup.py --template claude-code --config templates/e2b-config.yaml

# 3. Apply network isolation
python scripts/network-isolation.py --mode whitelist --config templates/e2b-config.yaml

# 4. Set resource limits
node scripts/resource-limiter.js --cpu 2 --memory 2048
```

### Pattern 2: Docker Container Isolation
```bash
# 1. Build secure container
./scripts/docker-configurator.sh --config templates/security-policy.json --build

# 2. Start multi-container environment
docker-compose -f templates/docker-compose.yml up -d

# 3. Apply network policies
python scripts/network-isolation.py --docker-network sandbox_isolated
```

### Pattern 3: Production VM Configuration
```bash
# 1. Apply security hardening
./scripts/docker-configurator.sh --mode vm --apply-hardening

# 2. Configure network boundaries
python scripts/network-isolation.py --mode whitelist --trusted-file trusted-domains.txt

# 3. Enforce resource quotas
node scripts/resource-limiter.js --profile production
```

## Security Best Practices

### 1. Principle of Least Privilege
- Start with maximum security (Level 1)
- Add exclusions only when necessary
- Document reason for each permission escalation
- Regular security audits

### 2. Defense in Depth
- Multiple layers of isolation (sandbox + network + resource limits)
- Fail-safe defaults (deny by default, allow by exception)
- Monitoring and alerting for anomalies
- Automated incident response

### 3. Validation and Testing
- Test security policies before production deployment
- Validate network isolation with port scans
- Confirm resource limits under load
- Regular penetration testing

### 4. Secrets Management
- Never hardcode secrets in configuration files
- Use environment variables for sensitive data
- Rotate credentials regularly
- Encrypt at rest and in transit

## Troubleshooting

### Common Issues

**Issue**: E2B sandbox fails to start
**Solution**: Check template compatibility, verify API key, increase timeout

**Issue**: Docker container cannot access network
**Solution**: Verify trusted domains, check DNS resolution, review iptables rules

**Issue**: Resource limits too restrictive
**Solution**: Monitor actual usage, adjust limits incrementally, use profiling

**Issue**: Permission denied errors
**Solution**: Review excluded commands, check file system permissions, validate user namespaces

## Performance Considerations

### E2B Sandboxes
- Template selection impacts startup time (base < nodejs < claude-code)
- Cold start: 2-5 seconds
- Warm start: <1 second
- Resource overhead: ~100MB base memory

### Docker Containers
- Image size affects pull time (use alpine base images)
- Volume mounts add minimal overhead
- Network isolation: <1ms latency
- Resource limits enforced at kernel level (negligible overhead)

### Network Policies
- iptables rules evaluated in order (optimize rule placement)
- DNS resolution caching reduces latency
- Whitelist mode faster than blacklist (fewer rules)

## Integration with Claude Code

### Hooks Integration
```javascript
// Pre-task hook: Apply sandbox configuration
npx claude-flow@alpha hooks pre-task \
  --run "python resources/scripts/e2b-setup.py --template nodejs"

// Post-edit hook: Validate security compliance
npx claude-flow@alpha hooks post-edit \
  --run "python resources/scripts/network-isolation.py --validate"

// Session-end hook: Cleanup sandbox resources
npx claude-flow@alpha hooks session-end \
  --run "python resources/scripts/e2b-setup.py --cleanup"
```

### Memory MCP Integration
Store sandbox configurations for reuse:
```bash
npx claude-flow@alpha memory store \
  --key "sandbox/config/production" \
  --value "$(cat .claude/settings.local.json)"
```

## Version History

**v1.0.0** (2025-11-02)
- Initial Gold tier release
- 4 production scripts (Python, Bash, Node.js)
- 3 configuration templates
- Comprehensive documentation
- Security best practices guide

## License

MIT License - See project root LICENSE file

## Support

- Documentation: This README and inline code comments
- Issues: Report via project issue tracker
- Security: Report vulnerabilities privately to security team
