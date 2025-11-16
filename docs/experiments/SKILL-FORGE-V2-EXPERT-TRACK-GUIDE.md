# skill-forge v2.0 Expert Track Guide

**Audience**: Power users, security-critical use cases (20% of skill creators)
**Time**: 60-90 minutes per skill
**Goal**: Create thoroughly validated, vulnerability-free skills for critical systems

---

## What is Expert Track?

**Expert Track** is the methodology-driven workflow for maximum thoroughness.

**Key Differences from Quick Track**:

| Feature | Quick Track | Expert Track |
|---------|-------------|--------------|
| **Time** | 20-30 min | 60-90 min |
| **Quality Ceiling** | 97% | 97% + vulnerability detection |
| **Phases** | 1 (Intake) + 5 (Instructions) | 0-8 (all phases) |
| **Techniques** | Templates + Validation | Templates + CoV + Adversarial + Metrics |
| **Best For** | Simple/Medium skills | Complex/Security-critical skills |
| **Value Prop** | Speed + Quality | Thoroughness + Safety |

**When to Use Expert Track**:
- ✅ Security-critical skills (authentication, encryption, access control)
- ✅ Complex workflows with many edge cases
- ✅ External dependencies (APIs, databases, third-party tools)
- ✅ Production systems with high impact
- ✅ Novel domains (low confidence in implementation)

**Validated Results (Pilot 3)**:
- Time: 115 min (v1.0) → 88 min (v2.0) = **-23% savings**
- Quality: 54% → 96.75% = **+79% improvement**
- **8 CRITICAL vulnerabilities found** (would have missed with Quick Track)
- All Expert Track techniques ROI ≥2.0

---

## The 9-Phase Expert Track Process

### Phase 0: Schema Definition (7-10 min) [Expert Track Only]

**Goal**: Define I/O contracts, error conditions, and dependencies BEFORE writing prose

**Why Important**: Prevents structural ambiguity, makes skills testable

**Use Template**: `templates/skill-schema.json`

**Required Sections**:
```json
{
  "skill_metadata": {
    "name": "skill-name",
    "version": "1.0.0",
    "complexity": "complex",
    "estimated_time": "30-60 min"
  },
  "input_contract": {
    "required": ["param1", "param2"],
    "optional": ["param3"],
    "constraints": ["param1 must be string", "param2 > 0"]
  },
  "output_contract": {
    "format": "json|yaml|text",
    "schema": {...},
    "success_conditions": ["condition1", "condition2"]
  },
  "error_conditions": {
    "error_types": [
      {"type": "invalid_input", "recovery": "..."},
      {"type": "network_failure", "recovery": "..."}
    ]
  },
  "dependencies": {
    "external_tools": ["tool1", "tool2"],
    "libraries": [...],
    "installation_check": "command to verify"
  },
  "performance_contract": {
    "expected_time": "30-60 seconds",
    "max_memory": "500MB",
    "scalability": "O(n) for n files"
  },
  "testing_contract": {
    "test_cases": ["nominal", "edge", "error"],
    "mock_inputs": [...],
    "expected_outputs": [...]
  }
}
```

**Validation**: `npm run validate:schema -- schema.json`
- Target: All required sections complete ✓
- Target: ≥2 output examples (nominal + edge) ✓

**ROI**: 5.0 (high completeness gains / time invested)

**Example**: Pilot 3 (security-audit-workflow)
- Defined 6 vulnerability types with severity levels
- Specified JSON output format for scan results
- Documented success conditions (zero critical vulns)
- Result: Prevented ambiguity in workflow execution

---

### Phase 1: Intent Archaeology (10-12 min) [Same as Quick Track]

**Goal**: Understand WHAT the skill does

**Use Template**: `templates/intake-template.yaml` (same as Quick Track)

**Required Fields**: See Quick Start Guide

**Validation**: `npm run validate:intake`
- Target: 15/15 checks pass ✅

---

### Phase 1b: Chain-of-Verification (CoV) on Intent (9-10 min) [Expert Track Only]

**Goal**: Self-critique the intake to catch errors BEFORE implementation

**Why Important**: Pilot 3 found 3 ambiguous requirements that CoV caught

**Use Protocol**: `templates/cov-protocol.md`

**5-Step CoV Process**:

**Step 1: Generate Initial Output** (Already done in Phase 1)

**Step 2: Self-Critique**
- What assumptions did I make?
- What did I miss or overlook?
- Where might I have misunderstood?

**Questions to Ask**:
```
1. Are trigger keywords specific enough? (avoid false positives)
2. Are success criteria measurable? (objective vs subjective)
3. Are constraints realistic? (not too restrictive or permissive)
4. Did I cover all user types? (who else might use this?)
5. Are example scenarios realistic? (not toy examples)
```

**Step 3: Evidence Check**
For each claim/decision in intake.yaml:
```
Claim: [statement]
Evidence FOR: [supporting facts, prior experience, research]
Evidence AGAINST: [contradicting facts, edge cases, assumptions]
Confidence: High/Medium/Low
```

**Step 4: Revise Based on Critique**
- Update intake.yaml with clarifications
- Add missing trigger keywords
- Refine vague success criteria
- Document assumptions

**Step 5: Confidence Rating**
Rate confidence per section:
```yaml
metadata:
  phase_1b_cov:
    skill_name_confidence: "High"
    problem_definition_confidence: "High"
    outcome_clarity_confidence: "Medium"  # Revised from initial
    trigger_keywords_confidence: "High"
    constraints_confidence: "Medium"  # Added 2 more after CoV
    overall_confidence: "High"
```

**Re-Validation**: `npm run validate:intake` (should still pass 15/15)

**ROI**: 3.5 (error reduction / time invested)

**Example**: Pilot 3 found:
- "Comprehensive scan" was vague → Specified CVE databases to check
- Improved factual accuracy from 70% → 85%+

---

### Phase 2-4: [Same as Quick Track - Not detailed here]

*Phases 2 (Use Case Mapping), 3 (Structural Scaffolding), 4 (Metadata Enrichment) follow standard workflow*

---

### Phase 5: Instruction Crafting (16-18 min) [Same as Quick Track]

**Goal**: Define HOW the skill works

**Use Template**: `templates/instruction-template.md`

**Pattern per Step**: See Quick Start Guide

**Validation**: `npm run validate:instructions`
- Target: Actionability ≥80% ✅

---

### Phase 5b: Chain-of-Verification (CoV) on Instructions (9-10 min) [Expert Track Only]

**Goal**: Self-critique instructions to catch vague/ambiguous steps

**Why Important**: Pilot 3 found 4 vague instructions ("check for vulnerabilities" → specific exploit tests)

**Use Protocol**: `templates/cov-protocol.md`

**Apply CoV to Each Step**:

**Step 2: Self-Critique Questions**:
```
1. Is the action verb specific? (not "handle", "process", "deal with")
2. Are success criteria measurable? (objective test conditions)
3. Is error handling complete? (all failure modes covered)
4. Are assumptions documented? (what does user need to know?)
5. Is implementation realistic? (actually works, not pseudocode)
```

**Step 3: Evidence Check Example**:
```markdown
**Step 3: Scan for SQL Injection**

Claim: "Run sqlmap to detect SQL injection vulnerabilities"
Evidence FOR: sqlmap is industry-standard tool, widely used
Evidence AGAINST: sqlmap requires target URL, may have false positives
Confidence: Medium (need to specify sqlmap configuration)

**Revised Step 3**:
**Action**: Run sqlmap with specific configuration to scan login endpoint

**Implementation**:
```bash
sqlmap -u "http://target/login" \
  --batch \
  --level=3 \
  --risk=2 \
  --random-agent \
  --output-dir=./sqlmap-results
```

**Success Criteria**:
- ✓ sqlmap completes scan (exit code 0)
- ✓ Results saved to ./sqlmap-results/
- ✓ No critical SQL injection detected (severity >= HIGH)

**Error Handling**:
- If target unreachable → Check network, retry with --timeout=10
- If false positives → Manual verification required for HIGH severity
```
```

**Step 4: Revise Vague Instructions**
- Replace vague verbs with specific actions
- Add explicit success criteria
- Document all failure modes
- Include concrete examples

**Step 5: Actionability Rating**
```yaml
metadata:
  phase_5b_cov:
    step_1_actionability: "High"  # Clear action + criteria
    step_2_actionability: "High"  # Revised from Medium
    step_3_actionability: "High"  # Added explicit criteria
    step_4_actionability: "Medium"  # Still some ambiguity
    overall_actionability: "90%"  # 9/10 steps with criteria
```

**Re-Validation**: `npm run validate:instructions`
- Target: Actionability ≥90% (Expert Track is stricter) ✅

**ROI**: 7.2 (highest ROI technique! Actionability gain / time)

**Example**: Pilot 3 improvements:
- "Check for vulnerabilities" → "Run specific exploit tests (SQL injection, XSS, CSRF)"
- Actionability: 25% → 90%+

---

### Phase 6: Resource Indexing [Standard workflow - Not detailed]

---

### Phase 7: Validation Gates [Enhanced for Expert Track]

**Standard Validation** (5 min):
```bash
npm run validate:skill -- my-skill/
```

**Plus: Phase 7a - Adversarial Testing** (24-30 min) [Expert Track Only]

**Goal**: Systematically discover vulnerabilities via red-team thinking

**Why Important**: Pilot 3 found **8 CRITICAL vulnerabilities** in security workflow

**Use Protocol**: `templates/adversarial-testing-protocol.md`

**4-Step Adversarial Process**:

**Step 1: Brainstorm Failure Modes** (10 min)
- Think like an attacker: "How could this skill fail catastrophically?"
- Consider: Security, performance, correctness, edge cases

**Categories**:
1. **Input Manipulation**: Malformed inputs, injection attacks
2. **State Corruption**: Race conditions, partial failures
3. **Dependency Failures**: External services down, rate limits
4. **Authentication/Authorization**: Privilege escalation, token theft
5. **Data Leakage**: Logs expose secrets, error messages leak info
6. **Resource Exhaustion**: Memory/CPU DoS, infinite loops

**Example**: Pilot 3 brainstormed 15 failure modes for security-audit-workflow

---

**Step 2: Risk Scoring Matrix** (5 min)

For each failure mode:
```
Risk Score = Likelihood × Impact

Likelihood (1-5):
1 = Rare (requires exotic conditions)
2 = Unlikely (non-obvious attack)
3 = Possible (moderate effort)
4 = Likely (easy to exploit)
5 = Very Likely (trivial exploit)

Impact (1-5):
1 = Trivial (cosmetic issue)
2 = Low (minor inconvenience)
3 = Medium (workflow degradation)
4 = High (data corruption, service disruption)
5 = Critical (security breach, complete failure)

Priority:
Score ≥12: CRITICAL - Must fix before deployment
Score 8-11: HIGH - Should fix
Score 4-7: MEDIUM - Fix if time permits
Score 1-3: LOW - Document limitation
```

**Example**: Pilot 3 found 8 scores ≥12 (all CRITICAL)
```
1. Scanner output not signed → tampering risk
   Likelihood: 4, Impact: 5 = Score: 20 (CRITICAL)

2. Credentials in logs → exposure risk
   Likelihood: 5, Impact: 4 = Score: 20 (CRITICAL)

3. No timeout → infinite scan DoS
   Likelihood: 3, Impact: 4 = Score: 12 (CRITICAL)
```

---

**Step 3: Fix Vulnerabilities** (7-10 min)

For each CRITICAL/HIGH risk:
1. Design mitigation
2. Update instructions/code
3. Add test case to verify fix
4. Document residual risk (if any)

**Example Fixes** (Pilot 3):
```markdown
**Vulnerability**: Scanner output tampering (score: 20)

**Fix**: Cryptographically sign scan results
```bash
# After scan completes
openssl dgst -sha256 -sign private.key \
  -out scan_results.sig \
  scan_results.json

# Verification step
openssl dgst -sha256 -verify public.key \
  -signature scan_results.sig \
  scan_results.json
```

**Test Case**: Modify scan_results.json manually, verify verification fails

**Residual Risk**: If attacker has private key, can forge results (LOW)
```

---

**Step 4: Reattack (Red Team Again)** (2-5 min)

After fixes:
1. Re-brainstorm: "Can I still exploit this?"
2. Re-score: Did risk decrease to acceptable level?
3. Iterate until all CRITICAL/HIGH risks mitigated

**Target**: All risks ≤7 (MEDIUM or below)

**ROI**: 2.0 (vulnerabilities found / time invested)
- Worth 3x time cost for security-critical skills
- Quick Track would have missed all 8 vulnerabilities

---

### Phase 8: Metrics Tracking (7-10 min) [Expert Track Only]

**Goal**: Measure V0→V1→V2 improvement for technique effectiveness database

**Why Important**: Long-term value - builds institutional knowledge of which techniques work

**Use Template**: `templates/skill-metrics.yaml`

**Track These Metrics**:

**Baseline V0** (before v2.0):
```yaml
baseline_v0:
  time_to_create: 115  # minutes
  metrics:
    factual_accuracy: 70  # % claims correct
    completeness: 40      # % required elements present
    precision: 82         # % sentences relevant
    actionability: 25     # % instructions with success criteria
  aggregate_score: 54     # weighted average
```

**Revision V1** (after v2.0):
```yaml
revision_v1:
  time_to_create: 88  # minutes
  revision_techniques_applied:
    - "Phase 0: Schema Definition"
    - "Phase 1b: CoV on Intent"
    - "Phase 5b: CoV on Instructions"
    - "Phase 7a: Adversarial Testing"
  metrics:
    factual_accuracy: 93
    completeness: 100
    precision: 94
    actionability: 100
  aggregate_score: 96.75
  time_savings: -23  # %
  quality_gain: +79  # %
```

**Technique Effectiveness**:
```yaml
technique_effectiveness:
  techniques_tested:
    - name: "Phase 0: Schema Definition"
      time_cost: 8  # minutes
      completeness_gain: 15  # percentage points
      overall_gain: 15
      roi_score: 5.0  # gain / (time / 3)

    - name: "Phase 1b: CoV on Intent"
      time_cost: 9
      factual_accuracy_gain: 15
      overall_gain: 15
      roi_score: 3.5

    - name: "Phase 5b: CoV on Instructions"
      time_cost: 9
      actionability_gain: 65
      overall_gain: 65
      roi_score: 7.2  # highest ROI!

    - name: "Phase 7a: Adversarial Testing"
      time_cost: 24
      vulnerabilities_found: 8
      critical_vulnerabilities: 8
      roi_score: 2.0  # vulnerabilities / (time / 12)
```

**Future V2** (optional - track continuous improvement):
```yaml
revision_v2:
  techniques_applied: ["Phase 7a: Reattack after 3 months"]
  new_vulnerabilities_found: 2
  ...
```

**ROI**: N/A (long-term benefit - builds technique database over time)

---

## Complete Expert Track Workflow

### Time Breakdown (88 min total for Pilot 3):

| Phase | Time | Description | ROI |
|-------|------|-------------|-----|
| **Phase 0** | 8 min | Schema Definition | 5.0 |
| **Phase 1** | 10 min | Intake (with template) | 9.0 |
| **Phase 1b** | 9 min | CoV on Intent | 3.5 |
| **Phase 2-4** | 10 min | Use Cases, Structure, Metadata | N/A |
| **Phase 5** | 16 min | Instructions (with template) | 7.8 |
| **Phase 5b** | 9 min | CoV on Instructions | 7.2 |
| **Phase 6** | 4 min | Resource Indexing | N/A |
| **Phase 7** | 3 min | Standard Validation | N/A |
| **Phase 7a** | 24 min | Adversarial Testing | 2.0 |
| **Phase 8** | 7 min | Metrics Tracking | N/A |
| **Total** | **88 min** | Expert Track Complete | **3.0 avg** |

**Compare to Quick Track**: 20-30 min (3x faster)
**Value**: Expert Track catches vulnerabilities Quick Track misses

---

## Decision Matrix: Quick vs Expert Track

| Factor | Quick Track | Expert Track |
|--------|-------------|--------------|
| **Time** | 20-30 min ✅ | 60-90 min ⚠️ |
| **Complexity** | Simple/Medium ✅ | Complex ✅ |
| **Security-Critical** | Low risk ✅ | High risk ✅ |
| **Dependencies** | Few/none ✅ | Many/external ✅ |
| **Quality Ceiling** | 97% ✅ | 97% ✅ |
| **Vulnerability Detection** | No ❌ | Yes (8 found) ✅ |
| **Best For** | 80% of skills | 20% of skills |

**Rule of Thumb**:
- If skill handles money, authentication, or encryption → **Expert Track**
- If skill interacts with external APIs or databases → **Expert Track**
- If skill will be used in production by non-technical users → **Expert Track**
- Otherwise → **Quick Track** (can always upgrade later)

---

## Expert Track ROI Analysis

### Investment: 88 minutes (vs 20-30 min Quick Track)
### Extra Time: 58-68 minutes

### Return:
1. **8 Critical Vulnerabilities Found** (Pilot 3)
   - Would have caused production incidents
   - Security workflow itself was vulnerable (ironic!)
   - Quick Track would have missed all 8

2. **Higher Confidence**
   - CoV catches errors before implementation
   - Adversarial testing provides safety assurance
   - Schema prevents structural ambiguity

3. **Long-Term Value**
   - Technique database improves over time
   - Institutional knowledge of what works
   - Future skills benefit from past metrics

**Breakeven**: 1-2 security incidents prevented = justified investment

---

## Common Questions

### Q: Can I mix Quick and Expert Track?
**A**: Yes! Use Quick Track as baseline, add Expert Track phases selectively:
- Add Phase 0 (Schema) if I/O contracts complex
- Add Phase 1b CoV if domain novel/uncertain
- Add Phase 5b CoV if instructions have many steps
- Add Phase 7a Adversarial if security-critical
- Add Phase 8 Metrics for first time using technique

---

### Q: Is Expert Track always better?
**A**: No. Expert Track is for thoroughness, not quality ceiling.
- Both tracks reach ~97% quality
- Expert Track catches vulnerabilities Quick Track misses
- But 3x longer (only worth it for critical systems)

**Quick Track sufficient for 80% of skills**

---

### Q: How do I know if adversarial testing found enough vulnerabilities?
**A**: No magic number, but aim for:
- 10-15 failure modes brainstormed
- All CRITICAL/HIGH risks (≥8) mitigated
- Residual risks documented and accepted
- Reattack finds no new CRITICAL/HIGH risks

**Pilot 3**: 15 brainstormed → 8 CRITICAL → all fixed → 0 new in reattack

---

### Q: Do I need to re-do Expert Track for updates?
**A**: Depends on update scope:
- Minor fixes (typos, clarifications) → No
- New features/steps → Redo Phase 7a Adversarial on new code
- Major refactor → Full Expert Track recommended

---

### Q: Can I skip metrics tracking (Phase 8)?
**A**: For your first skill, **yes** (saves 7-10 min). But recommended to track at least once per technique to validate ROI for your use cases.

**Long-term value**: Technique database helps prioritize future improvements.

---

## Template & Protocol Locations

**Expert Track Templates**:
- `templates/skill-schema.json` - Phase 0 schema
- `templates/skill-metrics.yaml` - Phase 8 metrics
- `templates/cov-protocol.md` - Phases 1b & 5b CoV
- `templates/adversarial-testing-protocol.md` - Phase 7a adversarial

**Also Needed (from Quick Track)**:
- `templates/intake-template.yaml` - Phase 1 intake
- `templates/instruction-template.md` - Phase 5 instructions

---

## Validation Scripts

```bash
npm run validate:schema -- schema.json        # Phase 0
npm run validate:intake -- intake.yaml        # Phase 1
npm run validate:instructions -- INSTRUCTIONS.md  # Phase 5
npm run validate:skill -- ./                  # Overall
```

---

## Expert Track Success Metrics

Track these to validate Expert Track value:

**Time**:
- V1.0 time: _____ min (estimate)
- V2.0 Expert Track time: _____ min (target: 60-90 min)
- Savings: _____ % (target: -23%)

**Quality**:
- V0 aggregate: _____ % (estimate)
- V1 aggregate: _____ % (target: 97%)
- Improvement: _____ % (target: +79%)

**Vulnerabilities**:
- Failure modes brainstormed: _____ (target: 10-15)
- CRITICAL/HIGH risks found: _____ (target: 5-10)
- Risks mitigated: _____ (target: 100%)
- Residual risks accepted: _____ (target: <3)

**Technique ROI**:
- Phase 0 ROI: _____ (target: 5.0)
- Phase 1b ROI: _____ (target: 3.5)
- Phase 5b ROI: _____ (target: 7.2)
- Phase 7a ROI: _____ (target: 2.0)

---

## Next Steps

1. **Try Expert Track on Security-Critical Skill**:
   - Use full 9-phase workflow
   - Track time and quality metrics
   - Count vulnerabilities found

2. **Compare to Quick Track**:
   - Did Expert Track find issues Quick Track would miss?
   - Was 3x time investment justified?
   - Which techniques provided highest ROI?

3. **Build Your Technique Database**:
   - Track Phase 8 metrics for 3-5 skills
   - Identify highest ROI techniques for your domain
   - Customize Expert Track workflow based on findings

4. **Share Feedback**:
   - Which Expert Track phases most valuable?
   - Which could be optional?
   - How to streamline without sacrificing thoroughness?

---

**Remember**: Expert Track is **NOT** about perfection - it's about **systematically catching issues Quick Track misses**. Use it when thoroughness > speed, and the cost of vulnerabilities > 60 minutes of extra work.

---

**Status**: Expert Track Guide Complete
**Last Updated**: 2025-11-06
**Version**: 1.0.0
