# Test 3: Complete 4-Phase SOP End-to-End

## Test Objective
Validate the complete 4-phase SOP methodology end-to-end, including Phase 4 technical enhancement, to produce a production-ready agent.

## Test Agent
**Name**: `api-security-auditor`
**Domain**: API security analysis and vulnerability detection
**Complexity**: Medium (specialized domain with technical depth)

## Test Scenario

This test validates the full workflow from initial intent to production-ready enhanced agent with Phase 4 technical implementation.

### Phase 1: Initial Analysis & Intent Decoding (30-45 min)

**Domain Breakdown**:
- Problem: Automated security auditing of REST and GraphQL APIs
- Key challenges:
  1. Authentication bypass vulnerabilities
  2. Authorization flaws (IDOR, privilege escalation)
  3. Injection attacks (SQL, NoSQL, command)
  4. Rate limiting and DoS protection
  5. Sensitive data exposure
  6. CORS misconfigurations

- Tech stack:
  - Testing tools: OWASP ZAP, Burp Suite, Postman
  - Security frameworks: OWASP Top 10, SANS CWE Top 25
  - API specs: OpenAPI/Swagger, GraphQL schema
  - Authentication: OAuth 2.0, JWT, API keys
  - Languages: Python (requests, httpx), Node.js

- Integrations:
  - MCP: Claude Flow, GitHub (report findings)
  - External: OWASP ZAP API, vulnerability databases

**Expected Outputs**:
- Security-focused domain analysis
- Vulnerability taxonomy
- Testing tool inventory
- Integration with security platforms

**Validation Gates**:
- [ ] 6+ security challenges identified
- [ ] Security framework mapping complete
- [ ] Tool integration patterns defined
- [ ] Vulnerability taxonomy comprehensive

### Phase 2: Meta-Cognitive Extraction (30-40 min)

**Expertise Domains**:
1. API security testing methodologies
2. Authentication and authorization patterns
3. Common vulnerability patterns (OWASP Top 10)
4. Secure coding practices
5. Compliance frameworks (PCI DSS, GDPR, SOC 2)

**Decision Frameworks**:
- When testing auth, always test both authn AND authz
- Never perform destructive tests without explicit permission
- Always validate SSL/TLS configuration first
- When finding vulnerability, classify by severity (CVSS)
- Always test rate limiting before load testing
- Never log or store sensitive data from requests
- When escalating, include proof-of-concept (PoC)
- Always verify false positives before reporting
- When testing injection, use safe payloads first
- Never skip authentication testing even if "trusted" API

**Quality Standards**:
- Zero false positives in high-severity findings
- Complete OWASP Top 10 coverage
- Actionable remediation guidance
- <5% false positive rate overall
- Compliance with responsible disclosure

**Expected Outputs**:
- Security agent specification
- Vulnerability examples (good/bad)
- Edge cases (custom auth, microservices, etc.)
- Ethical testing guardrails

**Validation Gates**:
- [ ] 5+ expertise domains identified
- [ ] 10+ decision heuristics documented
- [ ] Security examples include severity ratings
- [ ] Ethical testing guardrails defined

### Phase 3: Agent Architecture Design (40-50 min)

**Base System Prompt Structure**:

**Core Identity**:
- API Security Auditor with OWASP Top 10 expertise
- Penetration testing methodology
- Secure coding knowledge
- Compliance awareness

**Specialist Commands** (10+):
- /audit-authentication
- /test-authorization
- /scan-injection
- /check-rate-limiting
- /analyze-cors
- /test-sensitive-data
- /validate-ssl-tls
- /check-security-headers
- /test-session-management
- /scan-file-upload
- /generate-report

**Cognitive Framework**:
- Self-consistency: Validate findings with multiple techniques
- Program-of-thought: Decompose attack surface systematically
- Plan-and-solve: Structured security testing workflow

**Guardrails**:
- Never perform destructive operations
- Never exfiltrate actual sensitive data
- Never exceed rate limits intentionally
- Never skip permission verification
- Never test production without approval

**Workflow Examples**:
1. Standard API security audit
2. OAuth 2.0 penetration test
3. GraphQL security analysis
4. Microservices security review
5. Compliance audit (PCI DSS)

**Expected Outputs**:
- Base prompt v1.0 (400+ lines)
- Security-specific cognitive patterns
- Ethical testing guardrails
- 5 complete workflow examples

**Validation Gates**:
- [ ] All security domains covered
- [ ] 10+ specialist commands defined
- [ ] Ethical guardrails comprehensive
- [ ] Workflows include exact testing steps

### Phase 4: Deep Technical Enhancement (60-90 min)

**Code Pattern Extraction**:

1. **Authentication Testing Patterns**:
```python
# Pattern: JWT token validation bypass
def test_jwt_bypass(endpoint: str, token: str):
    """
    Test JWT validation vulnerabilities.
    File: api_security_tests.py:45-78
    """
    # None algorithm attack
    payload = jwt.decode(token, verify=False)
    malicious_token = jwt.encode(payload, None, algorithm='none')

    # Test weak signing key
    weak_keys = ['secret', 'password', 'key', '']
    for key in weak_keys:
        try:
            forged = jwt.encode(payload, key, algorithm='HS256')
            response = test_endpoint(endpoint, forged)
            if response.status_code == 200:
                return {"vulnerability": "JWT_WEAK_KEY", "severity": "HIGH"}
        except:
            continue
```

2. **SQL Injection Detection**:
```python
# Pattern: SQL injection testing
def test_sql_injection(endpoint: str, params: dict):
    """
    Test for SQL injection vulnerabilities.
    File: injection_tests.py:123-167
    """
    payloads = [
        "' OR '1'='1",
        "'; DROP TABLE users--",
        "' UNION SELECT NULL--",
        "admin'--",
        "' OR 1=1#"
    ]

    for key, value in params.items():
        for payload in payloads:
            test_params = params.copy()
            test_params[key] = payload

            response = requests.get(endpoint, params=test_params)

            # Check for SQLi indicators
            if any(indicator in response.text.lower() for indicator in
                   ['sql syntax', 'mysql', 'postgresql', 'ora-']):
                return {
                    "vulnerability": "SQL_INJECTION",
                    "severity": "CRITICAL",
                    "parameter": key,
                    "payload": payload
                }
```

**Critical Failure Mode Documentation**:

**Failure: Authentication Bypass via IDOR**
**Severity**: Critical
**Symptoms**: User can access resources belonging to other users
**Root Cause**: Missing authorization checks after authentication
**Prevention**:
```python
❌ DON'T:
@app.route('/api/user/<user_id>')
@authenticate
def get_user(user_id):
    return User.query.get(user_id)  # No authz check!

✅ DO:
@app.route('/api/user/<user_id>')
@authenticate
def get_user(user_id):
    user = User.query.get(user_id)
    if user.id != current_user.id and not current_user.is_admin:
        abort(403)
    return user
```

**MCP Integration Patterns**:
```javascript
// Store vulnerability findings
mcp__claude-flow__memory_store({
  key: "api-security-auditor/audit-123/findings",
  value: {
    endpoint: "/api/users/:id",
    vulnerability: "IDOR",
    severity: "HIGH",
    cvss_score: 8.1,
    remediation: "Implement authorization checks",
    poc: "GET /api/users/999 returns other user data"
  },
  ttl: 86400 * 30  // 30 days
})

// Report to GitHub
mcp__github__create_issue({
  title: "[Security] IDOR vulnerability in /api/users/:id",
  body: vulnerability_report_markdown,
  labels: ["security", "high-severity"]
})
```

**Performance Metrics**:
```yaml
Vulnerability Detection:
  - /memory-store --key "metrics/api-security-auditor/vulns-found" --increment 1
  - /memory-store --key "metrics/api-security-auditor/severity-{HIGH|MEDIUM|LOW}" --increment 1

Audit Quality:
  - false-positives: Track false positive rate
  - coverage: % of OWASP Top 10 tested
  - time-per-endpoint: Average scan duration

Compliance:
  - pci-dss-checks: PCI DSS requirement coverage
  - gdpr-compliance: Data protection checks
```

**Expected Outputs**:
- Enhanced prompt v2.0 (600+ lines)
- 15+ security testing code patterns
- 10+ failure modes with detection
- Complete MCP integration examples
- Security metrics framework
- Compliance checklist templates

**Validation Gates**:
- [ ] 15+ code patterns with file references
- [ ] All OWASP Top 10 covered
- [ ] Ethical testing guardrails enforced in code
- [ ] MCP patterns show exact syntax
- [ ] Metrics enable continuous improvement

## Complete Test Execution

### Full 4-Phase Run
```bash
cd C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\skills\agent-creator\resources\scripts

# Run all 4 phases (Phases 1-3 automated, Phase 4 manual enhancement)
python 4_phase_sop.py --agent-name api-security-auditor --mode interactive

# Manual Phase 4: Enhance prompt with technical patterns (see above)
# Create enhanced-prompt-v2.md with code patterns, failure modes, metrics
```

### Validation
```bash
# Validate enhanced prompt
bash ../scripts/validate_prompt.sh agent-outputs/api-security-auditor/api-security-auditor-enhanced-prompt-v2.md -v -s 90

# Expected: Score >= 90% (Gold tier)
```

### Comprehensive Testing
```bash
# Run comprehensive test suite
python ../scripts/test_agent.py --agent api-security-auditor --prompt-file agent-outputs/api-security-auditor/api-security-auditor-enhanced-prompt-v2.md --test-suite comprehensive

# Expected: 95%+ tests pass
```

### Integration Testing
```bash
# Test MCP integration patterns
python ../scripts/test_agent.py --agent api-security-auditor --test-suite integration

# Expected: All integration tests pass
```

## Success Criteria

**Phase Completion**:
- [ ] Phase 1 completes with comprehensive security analysis
- [ ] Phase 2 documents security expertise and ethical guardrails
- [ ] Phase 3 creates structured base prompt with workflows
- [ ] Phase 4 enhances with technical depth and code patterns

**Quality Gates**:
- [ ] Validation score >= 90% (Gold tier)
- [ ] Comprehensive test suite >= 95% pass rate
- [ ] Integration tests 100% pass
- [ ] Enhanced prompt 600+ lines with technical depth

**Production Readiness**:
- [ ] 15+ security testing code patterns
- [ ] 10+ failure modes documented
- [ ] Complete OWASP Top 10 coverage
- [ ] Ethical testing guardrails enforced
- [ ] MCP integrations fully specified
- [ ] Performance metrics framework

**Documentation**:
- [ ] All 4 phase outputs saved
- [ ] Agent specification complete
- [ ] Test reports generated
- [ ] Validation reports generated

## Expected Duration
- Phase 1: 35 minutes
- Phase 2: 35 minutes
- Phase 3: 45 minutes
- Phase 4: 75 minutes (technical enhancement)
- Testing & Validation: 15 minutes
- **Total**: 3.5 hours

## Notes
This test demonstrates the complete 4-phase SOP methodology producing a production-ready agent with:
- Deep domain expertise embedded
- Evidence-based prompting techniques
- Technical code patterns and exact implementations
- Comprehensive failure mode documentation
- Full MCP integration specifications
- Continuous improvement metrics

Success validates that following the complete 4-phase SOP produces Gold-tier, production-ready agents with genuine technical depth.
