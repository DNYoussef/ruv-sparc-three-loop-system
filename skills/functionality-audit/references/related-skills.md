# Functionality Audit Related Skills

## Overview

This document maps the functionality-audit skill to complementary skills in the validation and quality assurance ecosystem. Understanding when to use each skill and how they interact enables comprehensive code quality workflows.

---

## Theater Detection Audit

### Skill Description

Theater-detection-audit validates code authenticity through multi-agent Byzantine consensus verification. It answers the question "Is this real, working code or theatrical appearance of correctness?" by coordinating multiple independent agents to examine code through different lenses and reach consensus on its authenticity.

### Complementary Relationship

**Orthogonal Validation Dimensions**: Functionality-audit validates that code works correctly through execution testing. Theater-detection-audit validates that code is genuinely implemented rather than simulated or faked. These are independent dimensions of quality that both matter.

**Sequential Application**: Apply theater-detection first to verify code is real, then functionality-audit to verify it works correctly. This sequence prevents wasting execution testing effort on code that is fundamentally theatrical. However, functionality failures can also reveal theater that visual inspection missed.

**Consensus vs Execution**: Theater-detection uses multi-agent consensus to detect patterns of theatrical code. Functionality-audit uses direct execution to verify behavior. Both approaches surface quality issues but through different mechanisms.

### Decision Matrix: When to Use Each

| Scenario | Functionality Audit | Theater Detection | Both | Neither |
|----------|--------------------|--------------------|------|---------|
| AI-generated code with uncertain reliability | ⚠️ | ✅ | Ideal | ❌ |
| Legacy code with unknown correctness | ✅ | ⚠️ | Recommended | ❌ |
| High-stakes code requiring maximum confidence | ✅ | ✅ | Required | ❌ |
| Rapid prototyping with code quality uncertainty | ⚠️ | ✅ | Optional | ⚠️ |
| Production-ready code pre-deployment | ✅ | ⚠️ | Recommended | ❌ |
| Code review before merge | ⚠️ | ✅ | Ideal | ❌ |
| Simple utility functions | ✅ | ❌ | Optional | ⚠️ |
| Quick validation during development | ❌ | ❌ | ❌ | ✅ Use quick-quality-check |

**Legend**: ✅ Highly Applicable | ⚠️ Conditionally Useful | ❌ Not Recommended

### Integration Pattern

**Pipeline Integration**: Configure CI/CD to run theater-detection on code changes first, then functionality-audit only if theater-detection passes. This fail-fast approach saves compute resources and provides rapid feedback on fundamental quality issues.

**Complementary Evidence**: Use results from both skills together. Code that passes theater-detection but fails functionality-audit has genuine implementation bugs. Code that fails both has more fundamental problems. Code that passes functionality but fails theater-detection may indicate subtle theatrical patterns worth investigating.

---

## Quick Quality Check

### Skill Description

Quick-quality-check provides fast parallel validation through linting, security scanning, and basic test execution. It trades comprehensive validation for speed, delivering instant feedback on obvious issues without the overhead of full sandbox testing.

### Complementary Relationship

**Pre-Screening Function**: Quick-quality-check serves as a pre-screening filter before functionality-audit. Run quick checks first to catch obvious issues like syntax errors, linting violations, and security vulnerabilities. Only invoke full functionality auditing after quick checks pass.

**Speed vs Depth Trade-off**: Quick checks complete in seconds, functionality audits take minutes to hours. The speed differential makes quick checks appropriate for rapid iteration cycles, while functionality audits provide deep validation less frequently.

**Complementary Coverage**: Quick checks validate static properties (code style, security patterns, basic compilation). Functionality audits validate dynamic behavior through execution. Both coverage types are necessary for comprehensive quality assurance.

### Decision Matrix: When to Use Each

| Scenario | Quick Quality Check | Functionality Audit | Both | Neither |
|----------|--------------------|--------------------|------|---------|
| Pre-commit validation | ✅ | ❌ | Optional | ❌ |
| Continuous feedback during development | ✅ | ⚠️ | Sequential | ❌ |
| Pre-deployment validation | ⚠️ | ✅ | Required | ❌ |
| Code review preparation | ✅ | ⚠️ | Recommended | ❌ |
| Legacy code assessment | ❌ | ✅ | Sequential | ⚠️ |
| High-frequency iteration cycles | ✅ | ❌ | Periodic audit | ⚠️ |
| Security-critical code | ✅ | ✅ | Required | ❌ |
| Rapid prototyping | ✅ | ❌ | Final audit only | ⚠️ |

### Integration Pattern

**Pre-Commit Hooks**: Configure pre-commit hooks to run quick-quality-check automatically. Block commits that fail basic quality checks. This provides immediate feedback and prevents obvious issues from entering the codebase.

**CI/CD Pipeline Stages**: Structure pipelines with quick checks in early stages and functionality audits in later stages. Quick checks gate advancement to expensive audit stages, reducing wasted compute on fundamentally broken code.

**Feedback Latency Optimization**: Use quick checks for rapid iteration feedback during development. Schedule functionality audits periodically (e.g., nightly) or trigger them for significant changes. This balances feedback speed with validation depth.

---

## Testing Quality

### Skill Description

Testing-quality focuses on TDD (Test-Driven Development) methodology, test suite design, and test infrastructure quality. It validates and improves the quality of tests themselves rather than executing tests to validate production code.

### Complementary Relationship

**Test Infrastructure Building**: Testing-quality creates high-quality test suites. Functionality-audit executes those test suites in sandboxes to validate code. The skills work in sequence: testing-quality improves test design, functionality-audit runs tests effectively.

**Meta-Testing**: Testing-quality evaluates whether tests are well-designed, comprehensive, and maintainable. Functionality-audit evaluates whether code passes tests. Both perspectives on test quality matter for reliable validation.

**TDD Workflow Support**: Testing-quality guides TDD practice of writing tests before implementation. Functionality-audit validates implementations against those tests through sandbox execution. Together they support rigorous TDD methodology.

### Decision Matrix: When to Use Each

| Scenario | Testing Quality | Functionality Audit | Both | Neither |
|----------|----------------|--------------------|----|--------|
| Designing new test suite | ✅ | ❌ | Sequential | ⚠️ |
| Improving existing tests | ✅ | ⚠️ | Recommended | ❌ |
| Validating implementation | ⚠️ | ✅ | Optional | ❌ |
| TDD workflow | ✅ | ✅ | Required | ❌ |
| Test suite refactoring | ✅ | ✅ | Recommended | ❌ |
| Production deployment | ❌ | ✅ | Optional | ⚠️ |
| Test coverage analysis | ✅ | ⚠️ | Complementary | ❌ |
| Code without tests | ✅ | ⚠️ | Sequential | ❌ |

### Integration Pattern

**TDD Cycle Integration**: In TDD workflow, use testing-quality to design test cases, functionality-audit to validate implementation against tests, and iterate. The red-green-refactor cycle benefits from both test design guidance and execution validation.

**Test Suite Maintenance**: Periodically apply testing-quality to evaluate test suite health, identify gaps, and recommend improvements. Use functionality-audit to verify that test suite improvements actually improve validation effectiveness.

---

## Smart Bug Fix

### Skill Description

Smart-bug-fix provides intelligent automated debugging and fix generation. It analyzes bugs identified through testing and attempts to generate appropriate repairs using AI-assisted code modification.

### Complementary Relationship

**Debugging Handoff**: Functionality-audit identifies bugs through systematic testing. Smart-bug-fix receives those bug reports and attempts automated repair. The skills form a find-fix pipeline where audit discovers issues and smart-bug-fix addresses them.

**Validation Feedback Loop**: After smart-bug-fix generates a repair, functionality-audit validates the fix through re-execution of tests. This feedback loop ensures fixes actually work and don't introduce regressions.

**Manual vs Automated Debugging**: Functionality-audit includes systematic manual debugging methodology. Smart-bug-fix provides automated repair attempts. Use automated fixing when patterns are clear, fall back to manual debugging for complex issues.

### Decision Matrix: When to Use Each

| Scenario | Smart Bug Fix | Functionality Audit | Both | Neither |
|----------|--------------|--------------------|----|--------|
| Known bug requiring fix | ✅ | ⚠️ | Validation after fix | ❌ |
| Bug discovery | ❌ | ✅ | Sequential | ⚠️ |
| Regression fixing | ✅ | ✅ | Required | ❌ |
| Pattern-based bug repair | ✅ | ⚠️ | Fix validation | ❌ |
| Complex debugging requiring analysis | ⚠️ | ✅ | Automated attempt first | ❌ |
| Automated fix verification | ❌ | ✅ | Required | ⚠️ |
| High-volume similar bugs | ✅ | ⚠️ | Batch repair | ❌ |
| Novel or unusual bugs | ⚠️ | ✅ | Manual debugging | ❌ |

### Integration Pattern

**Automated Repair Pipeline**: Configure CI/CD to run functionality-audit, attempt smart-bug-fix on failures, then re-run functionality-audit to validate repairs. This automation can resolve straightforward bugs without human intervention.

**Human-in-Loop for Complex Cases**: Use smart-bug-fix for initial repair attempts but require human review before applying fixes. Functionality-audit validates both automated and human-reviewed fixes equally.

---

## Production Readiness

### Skill Description

Production-readiness provides comprehensive pre-deployment validation including infrastructure checks, monitoring setup, operational procedures, deployment automation, and end-to-end system validation.

### Complementary Relationship

**Subset Relationship**: Functionality-audit validates code correctness, which is one component of production readiness. Production-readiness encompasses functionality plus infrastructure, operations, monitoring, security, performance, and deployment concerns.

**Sequential Workflow**: Code must pass functionality-audit before production-readiness makes sense. No point validating deployment infrastructure for non-functional code. Apply functionality-audit first, then production-readiness if code works.

**Scope Differentiation**: Functionality-audit focuses narrowly on code execution correctness. Production-readiness takes a holistic view of deployment readiness including non-code concerns. Both scopes are necessary but address different questions.

### Decision Matrix: When to Use Each

| Scenario | Production Readiness | Functionality Audit | Both | Neither |
|----------|--------------------|--------------------|------|---------|
| Pre-deployment validation | ✅ | ⚠️ | Required | ❌ |
| Code correctness verification | ❌ | ✅ | Optional | ⚠️ |
| Infrastructure setup validation | ✅ | ❌ | Optional | ⚠️ |
| New feature development | ⚠️ | ✅ | Final validation | ❌ |
| Deployment automation testing | ✅ | ❌ | Complementary | ⚠️ |
| Monitoring and alerting setup | ✅ | ❌ | Independent | ⚠️ |
| End-to-end system validation | ✅ | ⚠️ | Complementary | ❌ |
| Bug fixing and regression testing | ❌ | ✅ | Optional | ⚠️ |

### Integration Pattern

**Deployment Pipeline Stages**: Structure deployment pipelines with functionality-audit in early stages validating code correctness, then production-readiness in later stages validating deployment infrastructure and operational readiness.

**Gate Criteria**: Use functionality-audit passage as a gate criterion before proceeding to production-readiness checks. This prevents wasting infrastructure validation effort on non-functional code.

---

## SOP Dogfooding Quality Detection

### Skill Description

SOP-dogfooding-quality-detection (Phase 1 of dogfooding system) runs Connascence analysis to detect code quality violations including God Objects, Parameter Bombs, cyclomatic complexity issues, deep nesting, long functions, and magic literals. Results are stored in Memory-MCP with WHO/WHEN/PROJECT/WHY metadata.

### Complementary Relationship

**Static vs Dynamic Analysis**: Quality detection performs static analysis identifying structural and coupling issues. Functionality-audit performs dynamic analysis through execution testing. Both perspectives catch different bug categories.

**Phase 1 Integration**: Quality detection is Phase 1 of a three-phase dogfooding cycle. It identifies violations that functionality-audit can validate through execution, and pattern-retrieval (Phase 2) can address through historical fixes.

**Connascence Focus**: Quality detection specializes in connascence and coupling analysis using NASA safety standards. Functionality-audit validates runtime behavior. Coupling issues often cause runtime failures that both skills can detect.

### Decision Matrix: When to Use Each

| Scenario | Quality Detection | Functionality Audit | Both | Neither |
|----------|------------------|--------------------|----|--------|
| Code quality assessment | ✅ | ⚠️ | Recommended | ❌ |
| Coupling analysis | ✅ | ❌ | Optional | ⚠️ |
| Runtime behavior validation | ❌ | ✅ | Optional | ⚠️ |
| NASA safety compliance | ✅ | ⚠️ | Complementary | ❌ |
| Pre-commit validation | ✅ | ⚠️ | Sequential | ❌ |
| Production deployment | ⚠️ | ✅ | Both recommended | ❌ |
| Legacy code audit | ✅ | ✅ | Required | ❌ |
| Refactoring planning | ✅ | ⚠️ | Quality guides focus | ❌ |

### Integration Pattern

**Dogfooding Cycle Entry Point**: Quality detection identifies issues, pattern-retrieval finds similar historical fixes, continuous-improvement orchestrates resolution, and functionality-audit validates fixes. This creates a complete self-improvement cycle.

**Memory-MCP Integration**: Both skills use Memory-MCP with tagging protocol (WHO/WHEN/PROJECT/WHY). Quality detection stores violation data, functionality-audit stores test results, enabling cross-skill analysis and learning.

---

## Comparison Summary Table

| Skill | Primary Focus | Validation Method | Speed | Depth | Use Case |
|-------|---------------|-------------------|-------|-------|----------|
| **functionality-audit** | Runtime correctness | Sandbox execution | Medium | High | Verify code works correctly |
| **theater-detection-audit** | Code authenticity | Multi-agent consensus | Medium | High | Detect theatrical fake code |
| **quick-quality-check** | Basic quality gates | Parallel static checks | Fast | Low | Rapid feedback during development |
| **testing-quality** | Test suite quality | Test design analysis | Medium | Medium | Improve test infrastructure |
| **smart-bug-fix** | Automated repair | AI-assisted fixing | Fast | Medium | Generate bug fixes automatically |
| **production-readiness** | Deployment readiness | Comprehensive audit | Slow | Very High | Pre-deployment validation |
| **quality-detection** | Code quality metrics | Connascence analysis | Fast | Medium | Detect coupling and complexity issues |

---

## Skill Combination Patterns

### Comprehensive Validation Pipeline

**Sequential Pattern**:
1. **quick-quality-check**: Fast static validation
2. **quality-detection**: Connascence and coupling analysis
3. **theater-detection-audit**: Authenticity verification
4. **functionality-audit**: Execution testing and debugging
5. **production-readiness**: Deployment validation

This pipeline provides defense-in-depth through multiple validation layers catching different issue categories.

### Rapid Iteration Cycle

**Iterative Pattern**:
1. **quick-quality-check**: Continuous during development
2. **functionality-audit**: Periodic (e.g., hourly) validation
3. **production-readiness**: Only for release candidates

This pattern balances feedback speed with validation thoroughness for rapid development.

### TDD Workflow

**TDD Pattern**:
1. **testing-quality**: Design test cases
2. **functionality-audit**: Validate implementation
3. **smart-bug-fix**: Repair failures
4. **functionality-audit**: Verify fixes

This pattern supports rigorous test-driven development methodology.

### Dogfooding Self-Improvement

**Dogfooding Pattern**:
1. **quality-detection**: Identify violations (Phase 1)
2. **pattern-retrieval**: Find historical fixes (Phase 2)
3. **smart-bug-fix**: Apply patterns (Phase 3)
4. **functionality-audit**: Validate repairs
5. **continuous-improvement**: Orchestrate cycle

This pattern enables continuous self-improvement through systematic quality enhancement.

---

## Conclusion

Functionality-audit operates as part of a comprehensive quality ecosystem. Understanding relationships between complementary skills enables practitioners to construct validation workflows appropriate for their context. No single skill provides complete quality assurance; combining skills systematically creates robust quality gates that catch different issue categories through diverse validation approaches.
