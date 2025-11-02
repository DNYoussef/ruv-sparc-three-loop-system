# Research Synthesis Techniques

## Overview

Techniques for combining research findings from multiple sources into coherent, actionable recommendations for software engineering decisions.

## Synthesis Process Overview

```
Raw Research → Evaluation → Triangulation → Synthesis → Recommendation
```

**Goal:** Transform scattered information into structured decision support

## Synthesis Technique 1: Decision Matrix

### When to Use
- Comparing 3-7 options with quantifiable criteria
- Need objective scoring methodology
- Multiple stakeholders with different priorities
- Trade-offs across multiple dimensions

### Process

**Step 1: Define Options (Rows)**
List all viable alternatives being considered

**Step 2: Define Criteria (Columns)**
Identify evaluation dimensions (performance, cost, complexity, etc.)

**Step 3: Assign Weights**
Prioritize criteria based on project requirements
- Sum of weights = 1.0
- Typically 3-6 criteria for clarity

**Step 4: Score Each Option**
Rate each option per criterion (1-10 scale)
- 10 = Excellent
- 7-9 = Good
- 4-6 = Acceptable
- 1-3 = Poor

**Step 5: Calculate Weighted Scores**
```
Weighted Score = Σ (Score × Weight)
```

**Step 6: Analyze Results**
- Identify highest-scoring option
- Examine close scores (within 10%)
- Investigate outliers

### Example: API Authentication Method Selection

**Context:** Selecting authentication method for customer-facing REST API

**Options:**
1. API Keys
2. JWT Tokens
3. OAuth 2.0
4. Session Cookies

**Criteria & Weights:**
| Criterion | Weight | Justification |
|-----------|--------|---------------|
| Security | 0.35 | Critical for customer data |
| Scalability | 0.25 | Expect 100K+ users |
| Implementation Complexity | 0.20 | Team bandwidth limited |
| User Experience | 0.15 | Mobile app integration |
| Cost | 0.05 | Budget not primary concern |

**Scoring Matrix:**

| Option | Security (×0.35) | Scalability (×0.25) | Complexity (×0.20) | UX (×0.15) | Cost (×0.05) | **Total** |
|--------|------------------|---------------------|---------------------|------------|--------------|-----------|
| API Keys | 4 (1.4) | 9 (2.25) | 10 (2.0) | 6 (0.9) | 10 (0.5) | **7.05** |
| JWT Tokens | 8 (2.8) | 9 (2.25) | 7 (1.4) | 8 (1.2) | 9 (0.45) | **8.10** ✅ |
| OAuth 2.0 | 10 (3.5) | 9 (2.25) | 4 (0.8) | 9 (1.35) | 7 (0.35) | **8.25** ✅ |
| Session Cookies | 6 (2.1) | 4 (1.0) | 8 (1.6) | 7 (1.05) | 10 (0.5) | **6.25** |

**Analysis:**
- **Top Choice:** OAuth 2.0 (8.25) - Highest security, good UX
- **Close Second:** JWT (8.10) - Simpler implementation
- **Decision:** OAuth 2.0 with JWT access tokens (hybrid approach)
  - Combines OAuth 2.0 security with JWT scalability
  - Complexity mitigated by using managed provider (Auth0)

**Sensitivity Analysis:**
If implementation complexity weight increases to 0.30 (team bandwidth critical):
- JWT: 8.80
- OAuth 2.0: 7.85
→ JWT becomes preferred (complexity matters more)

### Decision Matrix Template

```markdown
## Decision Matrix: [Topic]

### Options
1. [Option A]
2. [Option B]
3. [Option C]

### Criteria & Weights
- [Criterion 1]: 0.XX - [Justification]
- [Criterion 2]: 0.XX - [Justification]
- [Criterion 3]: 0.XX - [Justification]

### Scoring Matrix
| Option | Crit1 (×W) | Crit2 (×W) | Crit3 (×W) | Total |
|--------|------------|------------|------------|-------|
| A      | X (Y)      | X (Y)      | X (Y)      | Z.ZZ  |
| B      | X (Y)      | X (Y)      | X (Y)      | Z.ZZ  |
| C      | X (Y)      | X (Y)      | X (Y)      | Z.ZZ  |

### Analysis
- **Winner:** [Option X] (Score)
- **Runner-up:** [Option Y] (Score)
- **Decision:** [Choice with rationale]
- **Sensitivity:** [How decision changes with weight adjustments]
```

## Synthesis Technique 2: Phased Strategy

### When to Use
- High uncertainty or learning curve
- Large implementation effort (months)
- Need to validate assumptions before full commitment
- Risk of costly mistakes
- Team needs to build expertise

### Process

**Step 1: Define Phases**
Typically 2-4 phases:
- Phase 1: Proof of concept / pilot (weeks)
- Phase 2: Limited rollout (1-3 months)
- Phase 3: Full deployment (3-6 months)
- Phase 4: Optimization (ongoing)

**Step 2: Define Phase Goals**
Each phase should have:
- Clear objective
- Success metrics
- Duration estimate
- Resource requirements

**Step 3: Define Transition Criteria**
When to proceed to next phase:
- Quantitative metrics (performance, errors, adoption)
- Qualitative assessment (team confidence, user feedback)
- Risk assessment (acceptable vs unacceptable risks)

**Step 4: Plan Risk Mitigation**
- Rollback procedures
- Fallback options
- Monitoring and alerting
- Communication plan

**Step 5: Define Learning Goals**
What will we learn in each phase?
- Technical feasibility
- Performance characteristics
- Operational complexity
- User acceptance

### Example: Database Migration Strategy

**Context:** Migrating from PostgreSQL to Cassandra for 10K writes/sec requirement

**Phase 1: Proof of Concept (Weeks 1-4)**
**Goal:** Validate Cassandra can meet performance requirements

**Activities:**
- Set up 3-node Cassandra cluster (dev environment)
- Migrate core data model (users, events tables)
- Implement write and read patterns
- Run synthetic load tests (10K writes/sec)

**Success Metrics:**
- ✅ Achieve 10K+ writes/sec with p99 latency <20ms
- ✅ Read latency <10ms
- ✅ Data consistency verified
- ✅ Team can perform basic operations

**Transition Criteria:**
- All success metrics met
- No major technical blockers identified
- Team comfortable with Cassandra basics

**Rollback:** Abandon migration if performance targets not met

**Phase 2: Parallel Run (Weeks 5-12)**
**Goal:** Validate in production with real traffic

**Activities:**
- Deploy Cassandra cluster (production)
- Implement dual-write (PostgreSQL + Cassandra)
- Route 10% read traffic to Cassandra
- Monitor consistency, performance, errors

**Success Metrics:**
- ✅ Error rate <0.5% for Cassandra reads
- ✅ Latency within 10% of PostgreSQL
- ✅ Zero data consistency issues
- ✅ Ops team comfortable with monitoring

**Transition Criteria:**
- Run parallel for 2+ weeks without issues
- All success metrics met consistently
- Runbook documented

**Rollback:** Route reads back to PostgreSQL if error rate >2%

**Phase 3: Traffic Migration (Weeks 13-20)**
**Goal:** Gradually shift all traffic to Cassandra

**Activities:**
- Week 13-14: 25% reads to Cassandra
- Week 15-16: 50% reads to Cassandra
- Week 17-18: 100% reads to Cassandra
- Week 19-20: Migrate writes to Cassandra

**Success Metrics:**
- ✅ Error rate remains <0.5%
- ✅ Performance meets targets
- ✅ User satisfaction stable

**Transition Criteria:**
- Complete traffic migration
- PostgreSQL becomes backup only
- Cassandra handles 100% production load

**Rollback:** Instant rollback available via feature flag

**Phase 4: Optimization & Decommission (Weeks 21-24)**
**Goal:** Optimize Cassandra, decommission PostgreSQL

**Activities:**
- Tune Cassandra configuration
- Optimize data model based on production patterns
- Remove dual-write code
- Archive PostgreSQL data
- Decommission PostgreSQL cluster

**Success Metrics:**
- ✅ Cost savings achieved (40% reduction)
- ✅ Performance optimized
- ✅ Technical debt cleaned up

**Risk Assessment:**

| Phase | Risk Level | Key Risks | Mitigation |
|-------|------------|-----------|------------|
| 1 | Low | Performance insufficient | POC validates early |
| 2 | Medium | Data inconsistency | Parallel run detects issues |
| 3 | Medium-High | Production outage | Gradual rollout, instant rollback |
| 4 | Low | Optimization regression | Monitor closely |

**Learning Timeline:**

```
Week 0-4:  Cassandra basics, performance characteristics
Week 5-12: Production behavior, consistency model, operational procedures
Week 13-20: Scale behavior, edge cases, optimization opportunities
Week 21-24: Production tuning, cost optimization
```

### Phased Strategy Template

```markdown
## Phased Strategy: [Project]

### Phase 1: [Name] (Duration)
**Goal:** [Primary objective]

**Activities:**
- [Activity 1]
- [Activity 2]
- [Activity 3]

**Success Metrics:**
- ✅ [Metric 1]
- ✅ [Metric 2]
- ✅ [Metric 3]

**Transition Criteria:**
- [Quantitative criteria]
- [Qualitative criteria]

**Rollback:** [Rollback procedure]

### Phase 2: [Name] (Duration)
[Same structure]

### Risk Assessment
| Phase | Risk | Mitigation |
|-------|------|------------|
| ...   | ...  | ...        |

### Learning Goals
- Phase 1: [What we'll learn]
- Phase 2: [What we'll learn]
- Phase 3: [What we'll learn]
```

## Synthesis Technique 3: Trade-off Analysis

### When to Use
- No clear winner (all options have significant pros/cons)
- Need to communicate trade-offs to stakeholders
- Decision depends on context or priorities
- Want to document rationale for future reference

### Process

**Step 1: List All Options**
Typically 2-4 options

**Step 2: Identify Pros and Cons**
For each option:
- List 3-7 pros
- List 3-7 cons
- Categorize by impact (HIGH/MEDIUM/LOW)

**Step 3: Identify Deal-Breakers**
Mark any pros/cons that are absolute requirements or blockers

**Step 4: Analyze Context Sensitivity**
How do pros/cons change with different contexts?
- Team expertise
- Time constraints
- Budget limitations
- Scale requirements

**Step 5: Make Recommendation**
Choose option with justification:
- Best overall fit
- Conditional recommendation ("If X, choose A; if Y, choose B")
- Hybrid approach ("Combine elements of A and B")

### Example: Frontend Framework Selection

**Context:** Building internal admin dashboard for 5-person team

**Option A: React**

**Pros:**
- ✅ Team has strong React experience (HIGH)
- ✅ Huge ecosystem and library support (HIGH)
- ✅ Excellent developer tools (MEDIUM)
- ✅ Easy to hire React developers (MEDIUM)
- ✅ Component reusability (MEDIUM)

**Cons:**
- ❌ Requires additional libraries for routing, state (MEDIUM)
- ❌ Bundle size larger than competitors (LOW)
- ❌ JSX syntax has learning curve for non-React devs (LOW)

**Context Sensitivity:**
- If team is experienced: EXCELLENT choice
- If need rapid development: EXCELLENT choice
- If bundle size critical: Consider alternatives

**Option B: Vue**

**Pros:**
- ✅ Simpler learning curve (MEDIUM)
- ✅ All-in-one framework (routing, state included) (HIGH)
- ✅ Smaller bundle size (MEDIUM)
- ✅ Good documentation (MEDIUM)

**Cons:**
- ❌ Team has zero Vue experience (HIGH - DEAL-BREAKER)
- ❌ Smaller ecosystem than React (MEDIUM)
- ❌ Less common for enterprise (MEDIUM)
- ❌ Learning curve = 2-3 weeks (HIGH)

**Context Sensitivity:**
- If team is new: Better choice (simpler)
- If need to start immediately: Poor choice (learning curve)
- If internal tool with small scope: Good choice

**Option C: Vanilla JavaScript + Web Components**

**Pros:**
- ✅ No framework lock-in (HIGH)
- ✅ Minimal bundle size (HIGH)
- ✅ No build step required (MEDIUM)
- ✅ Standards-based (MEDIUM)

**Cons:**
- ❌ Much slower development (HIGH - DEAL-BREAKER)
- ❌ Need to build common patterns from scratch (HIGH)
- ❌ Limited component library (HIGH)
- ❌ Team unfamiliar with Web Components (MEDIUM)

**Context Sensitivity:**
- If long-term maintenance critical: Interesting option
- If rapid development needed: Poor choice
- If simple requirements: Could work

**Trade-off Matrix:**

| Factor | React | Vue | Vanilla JS |
|--------|-------|-----|------------|
| Development Speed | ✅✅✅ | ✅✅ | ❌ |
| Team Expertise | ✅✅✅ | ❌❌ | ❌ |
| Ecosystem | ✅✅✅ | ✅✅ | ❌ |
| Bundle Size | ✅ | ✅✅ | ✅✅✅ |
| Learning Curve | ✅✅ | ✅✅✅ | ❌❌ |
| Long-term Maintenance | ✅✅ | ✅✅ | ✅✅✅ |

**Decision:**
**Recommend: React**

**Rationale:**
1. Team expertise is HIGH impact factor (immediate productivity)
2. Rapid development needed (internal tool, 2-week timeline)
3. Bundle size LOW impact (internal admin dashboard, fast network)
4. Ecosystem matters (need charts, tables, date pickers)
5. Vue learning curve = 2-3 weeks (unacceptable delay)
6. Vanilla JS development time = 3-4x React (unacceptable)

**Conditional Recommendations:**
- **If team had no framework experience:** Choose Vue (simpler learning curve)
- **If bundle size critical:** Choose Vue (smaller)
- **If avoiding framework lock-in critical:** Consider Vanilla JS (but slow development)

**Hybrid Option:**
Not applicable (can't meaningfully combine frameworks)

### Trade-off Analysis Template

```markdown
## Trade-off Analysis: [Topic]

### Option A: [Name]
**Pros:**
- ✅ [Pro 1] (HIGH/MEDIUM/LOW)
- ✅ [Pro 2] (IMPACT)

**Cons:**
- ❌ [Con 1] (HIGH/MEDIUM/LOW)
- ❌ [Con 2] (IMPACT)

**Context Sensitivity:**
- If [context X]: [Impact on suitability]
- If [context Y]: [Impact on suitability]

### Option B: [Name]
[Same structure]

### Trade-off Matrix
| Factor | Option A | Option B | Option C |
|--------|----------|----------|----------|
| ...    | ✅✅✅   | ✅✅     | ✅       |

### Decision
**Recommend:** [Option X]

**Rationale:**
1. [Reason 1]
2. [Reason 2]
3. [Reason 3]

**Conditional Recommendations:**
- If [context A]: Choose [Option Y]
- If [context B]: Choose [Option Z]

**Hybrid Option:**
[If applicable, describe how to combine strengths]
```

## Synthesis Technique 4: Consensus Mapping

### When to Use
- Many sources with varying claims
- Need to identify expert consensus
- Conflicting information requires reconciliation
- Building confidence through agreement

### Process

**Step 1: Extract Key Claims**
From each source, extract specific factual claims

**Step 2: Categorize Claims**
Group similar claims together

**Step 3: Count Agreement**
For each claim category, count how many sources agree

**Step 4: Weight by Credibility**
Give more weight to high-credibility sources

**Step 5: Identify Consensus**
- Strong consensus: 80%+ of sources agree
- Moderate consensus: 60-79% agree
- No consensus: <60% agree

**Step 6: Investigate Conflicts**
For no-consensus claims:
- Why do sources disagree?
- Is it context-dependent?
- Are definitions different?

### Example: WebAssembly Performance Consensus

**Claim: "WebAssembly is faster than JavaScript"**

**Source Breakdown (8 sources):**

| Source | Claim | Credibility | Context |
|--------|-------|-------------|---------|
| Academic Paper (2019) | 2-4x faster | 95% | CPU-intensive tasks |
| Google V8 Blog (2023) | 1.2-1.5x faster | 95% | Optimized JavaScript |
| Mozilla Benchmarks (2024) | 4-6x faster | 90% | SIMD operations |
| Figma Case Study (2020) | 3x faster | 95% | Document parsing |
| Vendor Blog (ScyllaDB) | 10x faster | 60% | Specific benchmark |
| Medium Article | 1.5x faster | 70% | General tasks |
| Stack Overflow | "Much faster" | 60% | Anecdotal |
| Reddit Discussion | "Not always faster" | 55% | String operations |

**Consensus Analysis:**

**Strong Consensus (85% agreement):**
- ✅ WASM faster for CPU-intensive tasks (7/8 sources)
- ✅ WASM faster with SIMD support (6/7 relevant sources)
- ✅ WASM slower for string operations (5/6 sources mentioning)
- ✅ WASM slower for DOM manipulation (6/6 sources mentioning)

**Moderate Consensus (65% agreement):**
- ⚠️ Speedup ranges from 1.5x to 6x (depends on task type)
- ⚠️ Modern JavaScript JIT narrows gap (newer claims)

**No Consensus (<50%):**
- ❌ "10x faster" claim (only 1 source, low credibility)
- ❌ Whether WASM worth complexity (subjective, context-dependent)

**Conflict Resolution:**

**Conflict: "How much faster?"**
- Range: 1.2x to 10x
- Resolution: **Context-dependent**
  - CPU-intensive + SIMD: 4-6x (strong evidence)
  - CPU-intensive without SIMD: 2-4x (strong evidence)
  - Optimized JavaScript: 1.2-1.5x (strong evidence)
  - String operations: 0.5x (WASM slower) (strong evidence)

**Synthesis:**
"WebAssembly provides 4-6x speedup for CPU-intensive tasks with SIMD support (image processing, cryptography, compression). Speedup reduces to 1.2-1.5x when compared to highly-optimized JavaScript. WASM is slower for string operations and DOM manipulation."

### Consensus Mapping Template

```markdown
## Consensus Map: [Topic]

### Claim: "[Claim]"

### Source Breakdown
| Source | Claim | Credibility | Context |
|--------|-------|-------------|---------|
| ...    | ...   | ...         | ...     |

### Consensus Analysis

**Strong Consensus (>80%):**
- ✅ [Claim 1] (X/Y sources)
- ✅ [Claim 2] (X/Y sources)

**Moderate Consensus (60-79%):**
- ⚠️ [Claim 3] (X/Y sources)

**No Consensus (<60%):**
- ❌ [Claim 4] (X/Y sources)

### Conflict Resolution
**Conflict:** [Description]
- **Sources Disagree:** [Why]
- **Resolution:** [How resolved]

### Synthesis Statement
"[Final synthesized claim with nuance and context]"
```

## Combining Synthesis Techniques

Often, multiple techniques work together:

**Example Workflow:**
1. **Consensus Mapping** → Establish facts
2. **Decision Matrix** → Score options objectively
3. **Trade-off Analysis** → Understand pros/cons
4. **Phased Strategy** → Plan implementation

**Example: Technology Selection**
1. Consensus: "React has largest ecosystem" (7/8 sources agree)
2. Decision Matrix: React scores 8.5/10 (highest)
3. Trade-off Analysis: React pros (ecosystem, team expertise) outweigh cons (bundle size)
4. Phased Strategy: Phase 1 prototype (2 weeks) → Phase 2 full implementation (8 weeks)

## Memory-MCP Storage for Synthesis

```bash
# Store synthesis results for future reference
npx claude-flow@alpha memory store \
  --key "research/synthesis/[topic]" \
  --value "[synthesis result]" \
  --metadata '{
    "method": "decision-matrix|phased-strategy|trade-off-analysis|consensus-mapping",
    "options": ["A", "B", "C"],
    "recommendation": "Option X",
    "confidence": "high|medium|low",
    "sources": 8,
    "consensus": 85
  }'
```

## Next Steps

After synthesis:
1. Document synthesis using appropriate template
2. Store in Memory-MCP with metadata
3. Create implementation plan
4. Define success metrics
5. Schedule decision review (validate synthesis accuracy)
