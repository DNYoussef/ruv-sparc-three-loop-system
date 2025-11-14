# ADR-001: Agent Manifest Format (agent.yaml)

**Status**: ✅ Accepted
**Date**: 2025-11-01
**Decision Makers**: System Architecture Designer
**Related**: Quick Win #1

## Context

The Claude Flow ecosystem currently has 86 agent types with varying structures, configurations, and deployment patterns. There is no standardized way to:

1. Define agent metadata and capabilities
2. Declare dependencies and backing services
3. Configure agents across environments
4. Document agent requirements
5. Validate agent specifications
6. Enable agent portability and scaling

This lack of standardization creates challenges:
- Inconsistent agent behavior across environments
- Difficulty in migrating agents between deployments
- No clear understanding of agent requirements
- Limited observability and monitoring
- Complex manual configuration
- No automated validation

## Decision

We will adopt a **12-Factor compliant agent manifest specification** using `agent.yaml` files that:

1. **Define agents declaratively** using YAML format for human readability
2. **Validate against JSON Schema** (Draft 7) for machine verification
3. **Follow 12-Factor App principles** adapted for AI agents
4. **Support gradual adoption** without breaking existing agents
5. **Enable enhanced features** (observability, coordination, constraints)
6. **Provide comprehensive tooling** for validation and migration

### File Format: YAML

**Rationale**:
- ✅ Human-readable and writable
- ✅ Supports comments for documentation
- ✅ Industry standard for configuration (Kubernetes, Docker Compose, GitHub Actions)
- ✅ Better than JSON for large, nested structures
- ✅ Easy to validate with JSON Schema via conversion

**Alternatives Considered**:
- ❌ JSON: Less readable, no comments
- ❌ TOML: Less common, limited tooling
- ❌ XML: Too verbose, outdated

### Validation: JSON Schema Draft 7

**Rationale**:
- ✅ Industry standard with excellent tooling (AJV, validator libraries)
- ✅ Comprehensive validation rules (patterns, enums, conditionals)
- ✅ Self-documenting with descriptions and examples
- ✅ Supports complex validation logic (oneOf, anyOf, allOf)
- ✅ Wide language support (JavaScript, Python, Go, Rust)

**Alternatives Considered**:
- ❌ Custom validation: Reinventing the wheel
- ❌ TypeScript types: Not runtime-validated
- ❌ Zod/Yup: Less standard, JavaScript-only

### 12-Factor Methodology

**Rationale**:
- ✅ Proven methodology for cloud-native applications
- ✅ Ensures portability across environments
- ✅ Enables horizontal scaling
- ✅ Promotes stateless design
- ✅ Standardizes configuration management
- ✅ Industry best practice

**Adaptations for AI Agents**:
1. **Codebase**: Support for inline prompts in addition to Git
2. **Dependencies**: Add MCP servers and agent dependencies
3. **Port Binding**: Add MCP protocol alongside HTTP/gRPC
4. **Processes**: Focus on stateless event-driven models
5. **Extended Factors**: Add observability, capabilities, coordination

## Consequences

### Positive

✅ **Standardization**: All agents follow the same structure
✅ **Portability**: Agents can move between environments easily
✅ **Validation**: Automated checking prevents configuration errors
✅ **Documentation**: Self-documenting manifests with inline comments
✅ **Observability**: Built-in support for metrics and health checks
✅ **Scalability**: Clear horizontal scaling configuration
✅ **Migration**: Non-breaking path for existing agents
✅ **Tooling**: Standard tools (AJV, js-yaml) just work
✅ **Ecosystem**: Compatible with Kubernetes, Docker, cloud platforms

### Negative

❌ **Learning Curve**: Teams need to learn 12-Factor principles
   - *Mitigation*: Comprehensive documentation and examples
   - *Mitigation*: Gradual adoption without breaking changes

❌ **Initial Effort**: Creating manifests for existing agents
   - *Mitigation*: Auto-generation tools
   - *Mitigation*: Optional adoption (manifests not required)

❌ **File Complexity**: Full manifests can be large
   - *Mitigation*: Start minimal, add factors as needed
   - *Mitigation*: Templates for common patterns

### Neutral

○ **YAML vs JSON**: Some prefer JSON
   - *Response*: YAML converts to JSON for validation
   - *Response*: JSON manifests also supported if preferred

○ **12 Factors**: Not all factors needed for simple agents
   - *Response*: Minimal manifests require only 4 fields
   - *Response*: Add factors incrementally as needed

## Implementation

### Phase 1: Foundation (✅ Complete - Quick Win #1)

- [x] JSON Schema specification (schemas/agent-manifest-v1.json)
- [x] Annotated example (schemas/agent-manifest-v1.example.yaml)
- [x] Comprehensive documentation (docs/12fa/agent-yaml-specification.md)
- [x] Real-world example (examples/12fa/researcher-agent.yaml)
- [x] Validation tooling (examples/12fa/validate-example.js)
- [x] Migration guide
- [x] Best practices documentation

### Phase 2: Integration (Next)

- [ ] Claude Flow CLI integration (`claude-flow agent validate`)
- [ ] Auto-generation from existing agents
- [ ] Interactive manifest builder
- [ ] CI/CD validation hooks
- [ ] Runtime manifest loading
- [ ] Agent registry integration

### Phase 3: Adoption (Future)

- [ ] Migrate core agents (researcher, coder, reviewer, tester, planner)
- [ ] Migrate specialist agents (backend-dev, ml-developer, etc.)
- [ ] Migrate coordination agents (swarm coordinators)
- [ ] Update agent templates
- [ ] Agent marketplace integration
- [ ] Community agent submissions

### Phase 4: Enhancement (Long-term)

- [ ] Manifest-based agent deployment
- [ ] Dynamic agent loading from manifests
- [ ] Manifest versioning and upgrades
- [ ] Agent composition from multiple manifests
- [ ] Manifest inheritance and templates
- [ ] Advanced validation rules

## Architecture Decisions

### 1. Codebase Types (Factor 1)

**Decision**: Support 4 codebase types
- `git`: Production agents in repositories (recommended)
- `npm`: Published NPM packages (for distribution)
- `local`: Local filesystem paths (development only)
- `inline`: Prompts directly in manifest (simple agents)

**Rationale**: Covers all use cases from simple inline agents to production Git-based agents.

### 2. Dependency Management (Factor 2)

**Decision**: 5 dependency categories
- `npm_packages`: JavaScript dependencies
- `system_packages`: OS-level binaries
- `mcp_servers`: MCP server dependencies
- `other_agents`: Agent-to-agent dependencies
- `api_dependencies`: External API services

**Rationale**: Comprehensive dependency tracking enables proper validation and deployment.

### 3. Configuration (Factor 3)

**Decision**: JSON Schema + Environment Variables
- Config structure validated with JSON Schema
- All config via environment variables
- Required vs optional env vars
- Secret marking for sensitive values
- Strict mode for validation

**Rationale**: Industry standard approach (Kubernetes ConfigMaps, Docker env vars).

### 4. Backing Services (Factor 4)

**Decision**: 5 service categories
- `databases`: PostgreSQL, MySQL, MongoDB, Redis, SQLite, AgentDB
- `caches`: Redis, Memcached, in-memory
- `message_queues`: RabbitMQ, Kafka, Redis Queue, SQS
- `storage`: S3, GCS, Azure Blob, local filesystem
- `external_services`: Custom REST APIs, webhooks

**Rationale**: Covers all common backing service needs for AI agents.

### 5. Port Binding (Factor 7)

**Decision**: 4 protocol types
- `http`: REST APIs (most common)
- `grpc`: High-performance RPC
- `websocket`: Real-time bidirectional
- `mcp`: Model Context Protocol (agent-specific)

**Rationale**: MCP is key differentiator for AI agents vs traditional apps.

### 6. Observability (Extended Factor)

**Decision**: 3 observability pillars
- `metrics`: Prometheus, StatsD, JSON formats
- `tracing`: Jaeger, Zipkin, OpenTelemetry
- `health_checks`: Liveness and readiness probes

**Rationale**: Essential for production operations and debugging.

### 7. Backward Compatibility

**Decision**: Manifests are optional and additive
- Existing agents work without manifests
- Manifests enable enhanced features
- No breaking changes to current system
- Gradual migration supported

**Rationale**: Enables adoption without disrupting existing deployments.

## Validation Rules

### Strict Validation

1. **Name**: Must be kebab-case, 2-64 characters
2. **Version**: Must follow semver 2.0.0 exactly
3. **Agent Type**: Must be from predefined enum (17 types)
4. **Codebase Type**: Determines required fields (conditional validation)
5. **Environment Variables**: Pattern matching for names (UPPER_SNAKE_CASE)

### Permissive Validation

1. **Optional Factors**: Most factors are optional
2. **Extended Features**: All extended features optional
3. **Custom Metrics**: Free-form metric definitions
4. **Admin Tasks**: No required admin processes

**Rationale**: Balance between strictness (ensuring correctness) and flexibility (enabling diverse use cases).

## Security Considerations

### Secrets Management

**Decision**: Mark secrets explicitly
```yaml
config:
  env_vars_required:
    - name: API_KEY
      secret: true  # Never log or display
```

**Rationale**: Prevents accidental secret exposure in logs.

### Sandboxing

**Decision**: Security constraints in manifest
```yaml
constraints:
  security:
    sandbox: true
    network_access: restricted
    file_system_access: workspace-only
```

**Rationale**: Explicit security boundaries for agent execution.

### Log Redaction

**Decision**: Automatic sensitive data redaction
```yaml
logs:
  sensitive_data:
    redaction: true
    patterns:
      - "(?i)(api[_-]?key|token)..."
```

**Rationale**: Defense-in-depth against secret leakage.

## Scalability Considerations

### Horizontal Scaling

**Decision**: First-class scaling support
```yaml
concurrency:
  horizontal_scaling:
    enabled: true
    min_instances: 1
    max_instances: 10
    scaling_metric: queue-depth
```

**Rationale**: Cloud-native apps scale horizontally, not vertically.

### Stateless Design

**Decision**: Enforce stateless processes (Factor 6)
```yaml
processes:
  stateless: true
  session_management:
    type: external-store
```

**Rationale**: Stateless processes enable horizontal scaling.

### Resource Limits

**Decision**: Explicit resource constraints
```yaml
constraints:
  resource_limits:
    max_memory: 512Mi
    max_cpu: "1.0"
```

**Rationale**: Prevents resource exhaustion and enables bin packing.

## Comparison with Alternatives

### vs Docker Compose

| Feature | Agent Manifest | Docker Compose |
|---------|---------------|----------------|
| **Use Case** | AI agents | Containerized apps |
| **12-Factor** | ✅ Native | ⚠️ Partial |
| **Validation** | ✅ JSON Schema | ❌ No schema |
| **Observability** | ✅ Built-in | ⚠️ External tools |
| **Agent-specific** | ✅ MCP, capabilities | ❌ N/A |
| **Format** | YAML | YAML |

**Decision**: Inspired by Docker Compose but tailored for AI agents.

### vs Kubernetes Pod Spec

| Feature | Agent Manifest | Kubernetes Pod |
|---------|---------------|----------------|
| **Complexity** | 🟢 Low | 🔴 High |
| **12-Factor** | ✅ Native | ✅ Native |
| **Agent-specific** | ✅ MCP, skills | ❌ N/A |
| **Portability** | ✅ Platform-agnostic | ⚠️ K8s-specific |
| **Learning Curve** | 🟢 Easy | 🔴 Steep |

**Decision**: Simpler than K8s but can generate K8s manifests.

### vs package.json

| Feature | Agent Manifest | package.json |
|---------|---------------|--------------|
| **Scope** | 🟢 Full agent spec | 🔴 Just dependencies |
| **12-Factor** | ✅ Native | ❌ No |
| **Config** | ✅ Built-in | ❌ External |
| **Observability** | ✅ Built-in | ❌ No |
| **Compatibility** | ✅ Works together | N/A |

**Decision**: Complements package.json, doesn't replace it.

## Future Enhancements

### Version 1.1 (Planned)

- **Manifest Inheritance**: Base manifests with overrides
- **Composition**: Combine multiple manifests
- **Variables**: Manifest-level variable substitution
- **Profiles**: Environment-specific profiles (dev, staging, prod)

### Version 2.0 (Considered)

- **Distributed Coordination**: Multi-agent coordination specs
- **Learning Specifications**: Training and fine-tuning config
- **Cost Management**: Budget and cost tracking
- **Compliance**: Governance and compliance rules

## Alternatives Considered

### 1. Single Monolithic Config File

**Rejected**:
- ❌ Too large and complex
- ❌ Hard to maintain
- ❌ No separation of concerns

### 2. Multiple Small Config Files

**Rejected**:
- ❌ File sprawl
- ❌ Hard to find configs
- ❌ No single source of truth

### 3. Environment Variables Only

**Rejected**:
- ❌ No validation
- ❌ No documentation
- ❌ Not portable

### 4. Code-based Configuration

**Rejected**:
- ❌ Language-specific
- ❌ Requires code changes
- ❌ Not declarative

## References

1. [12-Factor App Methodology](https://12factor.net/)
2. [JSON Schema Specification](https://json-schema.org/)
3. [Semantic Versioning](https://semver.org/)
4. [Kubernetes Configuration Best Practices](https://kubernetes.io/docs/concepts/configuration/)
5. [Docker Compose Specification](https://docs.docker.com/compose/compose-file/)
6. [OpenAPI Specification](https://swagger.io/specification/)

## Related Documents

- `schemas/agent-manifest-v1.json` - JSON Schema definition
- `docs/12fa/agent-yaml-specification.md` - Complete specification
- `examples/12fa/researcher-agent.yaml` - Production example
- `docs/12fa/quick-win-1-summary.md` - Implementation summary

## Decision History

| Version | Date | Decision | Rationale |
|---------|------|----------|-----------|
| 1.0.0 | 2025-11-01 | Initial specification | Foundation for 12-Factor agents |
| - | Future | Manifest inheritance | Enable reuse and composition |
| - | Future | Multi-manifest composition | Support complex agent systems |

---

**Status**: ✅ Accepted and Implemented
**Decision Date**: 2025-11-01
**Review Date**: 2025-Q2 (quarterly review)
**Next Steps**: Phase 2 Integration (CLI tooling)
