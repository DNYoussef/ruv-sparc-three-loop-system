# Test 2: Complex Agent Creation

## Test Objective
Validate that the 4-phase SOP can handle a complex agent with multiple domains, extensive integrations, and sophisticated workflows.

## Test Agent
**Name**: `devops-orchestrator`
**Domains**: Cloud infrastructure, CI/CD, monitoring, security, containerization
**Complexity**: High (multi-domain, complex workflows, extensive tool integration)

## Test Scenario

### Phase 1: Initial Analysis & Intent Decoding
**Expected Inputs**:
- Problem: Orchestrate end-to-end DevOps workflows from code commit to production deployment
- Key challenges:
  1. Multi-cloud environment management (AWS, Azure, GCP)
  2. Zero-downtime deployments
  3. Security scanning and compliance
  4. Rollback strategies
  5. Monitoring and alerting integration
  6. Secret management
  7. Cost optimization
  8. Multi-team coordination

- Tech stack:
  - Container orchestration: Kubernetes, Docker, Helm
  - CI/CD: GitHub Actions, GitLab CI, Jenkins
  - Cloud providers: AWS (ECS, EKS, Lambda), Azure, GCP
  - IaC: Terraform, CloudFormation, Pulumi
  - Monitoring: Prometheus, Grafana, Datadog
  - Security: Trivy, Snyk, OWASP ZAP
  - Secret management: HashiCorp Vault, AWS Secrets Manager

- MCP servers:
  - Claude Flow (agent coordination)
  - GitHub integration (CI/CD triggers)
  - Custom cloud MCP servers

**Expected Outputs**:
- Comprehensive domain analysis (2+ pages)
- Extensive technology stack mapping
- Complex integration requirements
- Multiple coordination patterns

**Validation**:
- [ ] 8+ key challenges identified
- [ ] 10+ tools/frameworks documented
- [ ] 5+ integration points defined
- [ ] Multi-agent coordination patterns specified

### Phase 2: Meta-Cognitive Extraction
**Expected Inputs**:
- Expertise domains:
  1. Cloud infrastructure architecture
  2. Container orchestration
  3. CI/CD pipeline design
  4. Security and compliance
  5. Monitoring and observability
  6. Incident response
  7. Cost optimization

- Decision frameworks (10+ heuristics):
  - When deploying, always run security scans first
  - Never deploy directly to production without staging validation
  - Always implement canary deployments for critical services
  - When rollback needed, prioritize speed over perfect cleanup
  - Always validate Terraform plans before applying
  - Never store secrets in code or CI/CD logs
  - When scaling, consider cost implications first
  - Always implement circuit breakers for external dependencies
  - When incident occurs, coordinate with monitoring agent
  - Never skip smoke tests after deployment

- Quality standards:
  - Zero production incidents from deployments
  - <5 minute rollback time
  - 99.9% deployment success rate
  - All security scans pass
  - Cost variance <10%

**Expected Outputs**:
- Detailed agent specification (3+ pages)
- Multiple good/bad examples per domain
- Extensive edge cases (network failures, partial deployments, etc.)
- Failure mode catalog (10+ scenarios)

**Validation**:
- [ ] 5+ expertise domains identified
- [ ] 10+ decision heuristics documented
- [ ] Examples cover multiple domains
- [ ] 10+ failure modes documented

### Phase 3: Agent Architecture Design
**Expected Outputs**:
- Base system prompt v1.0 (comprehensive):
  - Core Identity covering all 5 domains
  - Universal commands with DevOps-specific usage patterns
  - 15+ specialist commands:
    - /deploy-canary
    - /rollback-deployment
    - /run-security-scan
    - /validate-terraform
    - /scale-service
    - /create-monitoring
    - /trigger-pipeline
    - /rotate-secrets
    - /cost-analysis
    - /incident-response
    - /validate-helm-chart
    - /test-smoke
    - /backup-state
    - /audit-compliance
    - /optimize-resources

  - 5+ MCP integrations:
    - Claude Flow (coordination)
    - GitHub (CI/CD)
    - AWS MCP (cloud operations)
    - Terraform MCP (IaC)
    - Monitoring MCP (observability)

  - Cognitive framework with domain-specific patterns:
    - Self-consistency: Validate across security, cost, performance
    - Program-of-thought: Deployment orchestration decomposition
    - Plan-and-solve: Multi-stage deployment workflow

  - 10+ guardrails across domains
  - 5+ workflow examples:
    1. Standard deployment workflow
    2. Emergency rollback workflow
    3. Security incident response
    4. Multi-region deployment
    5. Cost optimization audit

**Validation**:
- [ ] Base prompt exceeds 500 lines
- [ ] All 5 domains represented
- [ ] 15+ specialist commands defined
- [ ] 5+ MCP tools integrated
- [ ] 5+ complete workflow examples

### Phase 4: Technical Enhancement
**Expected Outputs**:
- Enhanced prompt v2.0 (production-ready):
  - Exact Kubernetes manifest patterns
  - Terraform module structures
  - AWS CDK code examples
  - Security scanning configurations
  - Monitoring query templates
  - Rollback automation scripts
  - Cost optimization strategies
  - Incident response playbooks

**Validation**:
- [ ] 20+ code patterns extracted
- [ ] 10+ failure modes with detection code
- [ ] MCP integration with exact syntax
- [ ] Performance metrics framework

## Test Execution

### Setup
```bash
cd C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\skills\agent-creator\resources\scripts
python 4_phase_sop.py --agent-name devops-orchestrator --mode interactive
```

### Phase-by-Phase Execution
Execute all 3 phases interactively with comprehensive inputs.

### Validation
```bash
bash ../scripts/validate_prompt.sh agent-outputs/devops-orchestrator/devops-orchestrator-base-prompt-v1.md -v
```
Expected: Score >= 85% (higher bar for complex agents)

### Comprehensive Testing
```bash
python ../scripts/test_agent.py --agent devops-orchestrator --test-suite comprehensive
```
Expected: 90%+ tests pass

## Success Criteria

- [ ] All 3 phases complete with complex inputs
- [ ] Phase 1 identifies 8+ challenges
- [ ] Phase 2 documents 5+ domains, 10+ heuristics
- [ ] Phase 3 creates comprehensive prompt (500+ lines)
- [ ] System prompt includes 15+ specialist commands
- [ ] 5+ complete workflow examples
- [ ] Validation score >= 85%
- [ ] Comprehensive test suite passes >= 90%
- [ ] Agent handles multi-domain coordination
- [ ] Evidence-based patterns applied to all domains

## Expected Duration
- Phase 1: 45-60 minutes (complex domain analysis)
- Phase 2: 40-50 minutes (extensive expertise extraction)
- Phase 3: 50-70 minutes (comprehensive prompt design)
- **Total**: 2.25-3 hours

## Notes
This test validates that the 4-phase SOP scales to production-grade agents with:
- Multiple interacting domains
- Complex decision-making
- Extensive tool integration
- Sophisticated workflows
- Production-level quality requirements

Success demonstrates the SOP's capability for enterprise-grade agent creation.
