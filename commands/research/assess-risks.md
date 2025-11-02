---
name: assess-risks
description: Comprehensive risk assessment for AI/ML research with ethical, safety, and reproducibility analysis
---

# 🛡️ Assess Research Risks (Pipeline F: Safety & Ethics)

Performs comprehensive risk assessment covering ethical concerns, safety implications, bias risks, privacy issues, dual-use potential, and reproducibility threats. Mandatory for all quality gates.

## Purpose

Part of **Pipeline F: Safety & Ethics**, this command conducts systematic risk analysis across 6 domains to identify and document potential harms, mitigation strategies, and ethical considerations. Required for Quality Gates 1, 2, and 3.

## Usage

```bash
npx claude-flow@alpha command assess-risks --component "<what>" [options]
```

## Required Arguments

- `--component <string>` - What to assess: `dataset`, `model`, `method`, `deployment`, `all`

## Optional Arguments

- `--output-dir <path>` - Output directory (default: `./risk-assessments/`)
- `--severity-threshold <level>` - Minimum severity to report: `low`, `medium` (default), `high`, `critical`
- `--domains <list>` - Risk domains to assess (default: `all`)
  - `ethical` - Fairness, bias, discrimination
  - `safety` - Harmful outputs, adversarial robustness
  - `privacy` - Data leakage, re-identification
  - `dual-use` - Misuse potential, weaponization
  - `reproducibility` - Replication risks, opacity
  - `environmental` - Energy consumption, carbon footprint
- `--framework <name>` - Assessment framework: `IEEE7010` (default), `NIST-AI-RMF`, `ISO42001`, `DPIA`
- `--interactive` - Interactive risk identification mode
- `--auto-mitigate` - Suggest mitigation strategies automatically
- `--store-memory` - Store assessment in Memory MCP

## What This Command Does

1. **Ethical Risk Assessment** (IEEE 7010 Framework):
   - **Fairness Analysis**: Disparate impact across protected groups
   - **Bias Detection**: Dataset bias, label bias, measurement bias
   - **Discrimination Risk**: Direct/indirect discrimination potential
   - **Transparency**: Explainability, interpretability assessment
   - **Accountability**: Clear responsibility assignment
   - **Human Rights**: Impact on fundamental rights (privacy, expression, etc.)

2. **Safety Risk Assessment** (NIST AI RMF):
   - **Harmful Outputs**: Toxic, offensive, dangerous content generation
   - **Adversarial Robustness**: Susceptibility to attacks (evasion, poisoning)
   - **Failure Modes**: Graceful degradation, edge case handling
   - **Uncertainty Quantification**: Confidence calibration, OOD detection
   - **Cascading Failures**: Downstream impact of errors
   - **Safety Guardrails**: Existing controls and gaps

3. **Privacy Risk Assessment** (GDPR DPIA):
   - **Data Minimization**: Excessive data collection risks
   - **Re-identification**: De-anonymization potential
   - **Membership Inference**: Training data leakage
   - **Model Inversion**: Reconstruction attacks
   - **Linkage Attacks**: Cross-dataset correlation
   - **Consent & Control**: Data subject rights compliance

4. **Dual-Use Risk Assessment**:
   - **Misuse Potential**: Weaponization, surveillance, manipulation
   - **Accessibility**: Who can access? What safeguards exist?
   - **Offensive Applications**: Disinformation, deepfakes, automated attacks
   - **Proliferation**: Spread of harmful capabilities
   - **Mitigation Strategies**: Access controls, rate limits, watermarking

5. **Reproducibility Risk Assessment**:
   - **Opacity**: Undocumented choices, hidden parameters
   - **Randomness**: Insufficient seed control, non-deterministic ops
   - **Environment**: Dependency versioning, hardware dependencies
   - **Data Provenance**: Dataset versioning, splits documentation
   - **Code Availability**: Missing code, incomplete instructions

6. **Environmental Risk Assessment**:
   - **Energy Consumption**: Training/inference power usage
   - **Carbon Footprint**: CO2 emissions estimate
   - **Resource Efficiency**: Computational waste, over-parameterization
   - **Sustainability**: Long-term environmental impact

## Output Files

- `risk-assessments/<component>-risk-assessment.md` - Comprehensive risk report
- `risk-assessments/<component>-risk-matrix.csv` - Severity x likelihood matrix
- `risk-assessments/<component>-mitigation-plan.md` - Proposed mitigations
- `risk-assessments/<component>-ethics-checklist.md` - Ethics review checklist
- `risk-assessments/<component>-dpia.md` - Data Protection Impact Assessment (if applicable)

## Example

```bash
# Assess all risks for a dataset
npx claude-flow@alpha command assess-risks \
  --component dataset \
  --domains "ethical,privacy,bias" \
  --framework IEEE7010 \
  --auto-mitigate \
  --store-memory

# Safety assessment for model deployment
npx claude-flow@alpha command assess-risks \
  --component deployment \
  --domains safety \
  --severity-threshold high \
  --framework NIST-AI-RMF

# Comprehensive assessment (all domains, all components)
npx claude-flow@alpha command assess-risks \
  --component all \
  --interactive \
  --auto-mitigate
```

## Risk Matrix Example

The command generates a structured risk matrix:

| Risk ID | Domain | Description | Likelihood | Severity | Impact | Mitigation |
|---------|--------|-------------|------------|----------|--------|------------|
| RISK-001 | Ethical | Gender bias in training data | High | High | Critical | Bias audit, rebalancing |
| RISK-002 | Safety | Toxic output generation | Medium | High | High | Content filtering, human review |
| RISK-003 | Privacy | Membership inference attack | Medium | Medium | Medium | Differential privacy, aggregation |
| RISK-004 | Dual-use | Deepfake generation | Low | Critical | High | Access controls, watermarking |
| RISK-005 | Repro | Missing random seeds | High | Low | Low | Seed documentation, deterministic ops |

**Severity Levels**:
- **Critical**: Fundamental rights violations, serious harm
- **High**: Significant harm, regulatory violations
- **Medium**: Moderate harm, reputational risk
- **Low**: Minor issues, edge cases

**Likelihood Levels**:
- **High**: Very likely to occur (>50%)
- **Medium**: May occur (10-50%)
- **Low**: Unlikely (<10%)

## IEEE 7010 Risk Domains

The command implements IEEE 7010 (Well-being Impact Assessment):

1. **Physical**: Bodily harm, safety risks
2. **Psychological**: Mental health, stress, manipulation
3. **Social**: Relationships, community impact, isolation
4. **Economic**: Financial harm, job displacement
5. **Political**: Discrimination, surveillance, freedom
6. **Environmental**: Ecological impact, sustainability

## Mitigation Strategies

Auto-generated mitigations by risk type:

**Bias Mitigation**:
- Data rebalancing and augmentation
- Fairness constraints during training
- Post-processing calibration
- Regular bias audits

**Safety Mitigation**:
- Content filtering and safety classifiers
- Human-in-the-loop review
- Rate limiting and access controls
- Adversarial training

**Privacy Mitigation**:
- Differential privacy (ε, δ parameters)
- Federated learning
- Secure multi-party computation
- Data anonymization techniques

**Dual-Use Mitigation**:
- Access authentication and authorization
- Usage monitoring and logging
- Watermarking and provenance tracking
- Ethical use agreements

## Integration with Quality Gates

**Gate 1 Requirements**:
- ✅ Ethics review initiated (Form F-F1)
- ✅ Dataset risks assessed
- ✅ High/critical risks documented

**Gate 2 Requirements**:
- ✅ Model safety assessment complete
- ✅ Mitigation strategies implemented
- ✅ Fairness metrics computed

**Gate 3 Requirements**:
- ✅ Deployment risk assessment
- ✅ All critical risks mitigated
- ✅ Monitoring plan established

## Integration with Agents

- **ethics-agent**: Primary owner, conducts ethical review
- **evaluator**: Validates risk assessment completeness
- **data-steward**: Provides dataset risk information
- **archivist**: Archives risk assessments with versions
- **reliability-agent**: Implements safety mitigations

## Memory MCP Storage

When `--store-memory` is enabled:

```json
{
  "key": "sop/risk-assessments/<component>",
  "value": {
    "component": "dataset",
    "assessment_path": "./risk-assessments/dataset-risk-assessment.md",
    "total_risks": 12,
    "critical_risks": 2,
    "high_risks": 4,
    "medium_risks": 4,
    "low_risks": 2,
    "mitigated_risks": 8,
    "unmitigated_risks": 4,
    "gate_1_ready": true,
    "gate_2_ready": false,
    "gate_3_ready": false,
    "created_at": "2025-11-01T12:00:00Z",
    "last_updated": "2025-11-01T14:30:00Z"
  },
  "tags": ["SOP", "Pipeline-F", "risk-assessment", "ethics-agent", "safety"]
}
```

## Risk Assessment Template

```markdown
# Risk Assessment: [Component Name]

## Executive Summary
- Total Risks Identified: 12
- Critical: 2 | High: 4 | Medium: 4 | Low: 2
- Mitigation Status: 8/12 (67%)

## 1. Ethical Risks
### RISK-001: Gender Bias in Training Data
- **Domain**: Ethical / Fairness
- **Likelihood**: High (estimated 70%)
- **Severity**: High
- **Impact**: Discriminatory predictions for protected groups
- **Mitigation**: Bias audit, rebalancing, fairness constraints
- **Status**: In Progress

[... detailed risk entries ...]

## 2. Safety Risks
[...]

## 3. Privacy Risks
[...]

## Quality Gate Readiness
- Gate 1: ✅ Ready
- Gate 2: ⏳ In Progress
- Gate 3: ❌ Not Ready
```

## Related Commands

- `/ethics-review` - Initiate formal ethics review (Form F-F1)
- `/bias-audit` - Detailed bias analysis for datasets/models
- `/safety-eval` - Safety evaluation with adversarial testing
- `/privacy-audit` - Privacy risk analysis with attack simulation
- `/validate-gate-1` - Check Gate 1 requirements including risk assessment
- `/mitigate-risk` - Implement specific risk mitigation

## Compliance Frameworks

The command supports multiple compliance frameworks:

- **IEEE 7010**: Well-being Impact Assessment
- **NIST AI RMF**: AI Risk Management Framework
- **ISO 42001**: AI Management System
- **GDPR DPIA**: Data Protection Impact Assessment
- **FDA AI/ML**: Software as Medical Device guidance
- **EU AI Act**: High-risk AI system requirements

## References

- IEEE 7010-2020: Well-being Metrics for Ethical AI and AS
- NIST AI Risk Management Framework (2023)
- ISO/IEC 42001:2023: AI Management System
- Deep Research SOP Pipeline F: Safety & Ethics
- Quality Gate Requirements (SOP Section 4.3)

## Next Steps

After risk assessment:
1. Review high/critical risks with ethics committee
2. Implement mitigation strategies: `/mitigate-risk --risk-id <ID>`
3. Document in ethics review: `/ethics-review --assessment <file>`
4. Schedule regular reassessment (quarterly recommended)
5. Validate gate readiness: `/validate-gate-<N> --pipeline F`
