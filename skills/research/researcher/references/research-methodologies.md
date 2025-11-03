# Research Methodologies for Software Engineering

## Overview

This document outlines systematic research methodologies for software engineering decision-making, from quick fact-finding to comprehensive technical analysis.

## Research Methodology Spectrum

### Level 1: Quick Research (30-60 minutes)
**When to Use:** Straightforward questions with established best practices

**Process:**
1. Formulate specific question
2. Search authoritative sources (2-3)
3. Validate with official documentation
4. Synthesize findings
5. Document decision

**Sources:**
- Official documentation
- Well-known technical blogs (Google, Meta, Netflix)
- Stack Overflow (for consensus)
- MDN Web Docs, W3C specs

**Quality Bar:**
- 2+ authoritative sources
- 85%+ credibility
- Clear consensus

### Level 2: Multi-Source Research (2-4 hours)
**When to Use:** Decisions with trade-offs requiring balanced evaluation

**Process:**
1. Define research scope and sub-questions
2. Gather quantitative data (benchmarks, costs)
3. Gather qualitative data (case studies, experiences)
4. Triangulate across 5-7 sources
5. Evaluate credibility
6. Synthesize recommendations
7. Store in Memory-MCP

**Sources:**
- Academic benchmarks (YCSB, TPC)
- Industry case studies
- Official documentation
- Cost calculators
- Community surveys

**Quality Bar:**
- 5-7 independent sources
- 90%+ average credibility
- Triangulation validates consensus
- Quantitative + qualitative data

### Level 3: Technical Deep Dive (1-2 days)
**When to Use:** Complex technical decisions with significant implementation cost

**Process:**
1. Comprehensive question formulation
2. Literature review (academic papers, technical reports)
3. Real-world case study analysis
4. Experimental validation (reproduce key benchmarks)
5. Toolchain comparison
6. Pitfall identification
7. Phased implementation strategy
8. Risk mitigation planning

**Sources:**
- Peer-reviewed academic papers
- Browser vendor documentation
- Production case studies (Figma, Uber, Discord)
- Experimental validation (E2B sandboxes)
- Community benchmarks (reproducible)

**Quality Bar:**
- 8-12 authoritative sources
- 90%+ average credibility
- Experimental validation confirms research
- Real-world production evidence
- Phased implementation plan

## Research Question Formulation

### SMART Research Questions

**Specific:** Not "Is WASM faster?" but "What is the performance difference between WASM and JavaScript for CPU-bound image processing?"

**Measurable:** Include quantifiable metrics (latency, throughput, cost)

**Achievable:** Answerable within time/resource constraints

**Relevant:** Directly impacts decision at hand

**Time-bound:** Consider recency of sources (prefer 2023+)

### Question Decomposition

Break complex questions into sub-questions:

**Example: "Should we migrate to microservices?"**
1. What are current monolith pain points? (performance, scalability, development velocity)
2. What are microservice benefits for our specific use case?
3. What are microservice challenges? (operational complexity, latency, consistency)
4. What is our team's expertise level with distributed systems?
5. What is the migration cost? (time, effort, risk)
6. What are similar companies' experiences?

## Source Credibility Evaluation

### Source Authority Matrix

| Source Type | Authority | Typical Credibility | When to Use |
|-------------|-----------|---------------------|-------------|
| Peer-reviewed papers | ★★★★★ | 95-98% | Technical deep dives |
| Browser vendor docs | ★★★★★ | 90-95% | Web technology decisions |
| Production case studies | ★★★★★ | 90-95% | Real-world validation |
| Official documentation | ★★★★☆ | 85-90% | API usage, features |
| Industry surveys | ★★★★☆ | 80-90% | Market trends, practices |
| Technical blogs (reputable) | ★★★☆☆ | 75-85% | Tutorials, opinions |
| Vendor benchmarks | ★★☆☆☆ | 60-75% | Initial exploration (verify independently) |
| Anonymous forums | ★★☆☆☆ | 50-70% | Problem-solving, not decisions |

### Credibility Assessment Checklist

**Author:**
- [ ] Identifiable author with relevant expertise
- [ ] Author works for reputable organization
- [ ] Author has track record in domain

**Methodology:**
- [ ] Transparent methodology
- [ ] Reproducible results
- [ ] Data sources cited
- [ ] Limitations acknowledged

**Recency:**
- [ ] Published/updated within 2 years
- [ ] Information still relevant (tech changes fast)

**Bias:**
- [ ] Potential conflicts of interest disclosed
- [ ] Balanced presentation (not purely promotional)
- [ ] Acknowledges trade-offs

**Validation:**
- [ ] Confirmed by other independent sources
- [ ] Aligns with expert consensus
- [ ] Real-world production evidence

## Triangulation Techniques

### Consensus Triangulation
**Goal:** Find agreement across independent sources

**Method:**
1. Gather 5-7 independent sources
2. Extract key claims from each
3. Identify consensus claims (4+ sources agree)
4. Identify conflicts (sources disagree)
5. Investigate conflicts (why do they disagree?)
6. Weight consensus by source credibility

**Example:**
- 7 sources researched on database selection
- 6/7 agree: Cassandra excels at write throughput
- 6/7 agree: PostgreSQL easier to operate
- 3/7 say MongoDB, 3/7 say Cassandra for analytics
  → Conflict requires deeper investigation

### Cross-Domain Triangulation
**Goal:** Validate findings across different types of evidence

**Method:**
1. Quantitative data (benchmarks, metrics)
2. Qualitative data (case studies, experiences)
3. Theoretical analysis (academic papers)
4. Practical validation (experiments, proof-of-concept)

**Example:**
- Benchmark: WASM 5x faster than JavaScript
- Case study: Figma achieved 3x faster with WASM
- Academic paper: WASM 67-95% of native performance
- Experiment: Self-conducted test achieved 5.1x speedup
  → All evidence aligns, high confidence

### Temporal Triangulation
**Goal:** Validate findings over time

**Method:**
1. Original research (2019)
2. Follow-up research (2022)
3. Recent research (2024)
4. Trend analysis

**Example:**
- 2019: WASM 2-3x faster than JavaScript
- 2022: WASM 3-4x faster (JIT improvements)
- 2024: WASM 4-5x faster (SIMD support)
  → Trend: Performance improving over time

## Synthesis Techniques

### Decision Matrix Synthesis
**When:** Comparing multiple options with quantifiable criteria

**Method:**
1. List all options (rows)
2. List all criteria (columns)
3. Score each option per criterion (1-10)
4. Weight criteria by importance
5. Calculate weighted scores
6. Identify top option(s)

**Example: Database Selection**
| Database | Performance | Cost | Ops Complexity | Team Expertise | Weighted Score |
|----------|-------------|------|----------------|----------------|----------------|
| PostgreSQL | 6 (×0.3) | 9 (×0.2) | 9 (×0.2) | 10 (×0.3) | 8.2 |
| MongoDB | 7 (×0.3) | 7 (×0.2) | 7 (×0.2) | 5 (×0.3) | 6.5 |
| Cassandra | 9 (×0.3) | 8 (×0.2) | 3 (×0.2) | 2 (×0.3) | 5.7 |

**Result:** PostgreSQL wins due to team expertise weight

### Phased Strategy Synthesis
**When:** Decisions with high uncertainty or learning curve

**Method:**
1. Identify phases (short-term → long-term)
2. Define goals for each phase
3. Specify conditions for phase transitions
4. Plan risk mitigation
5. Define success metrics

**Example: Technology Migration**
- **Phase 1 (1-3 mo):** Proof of concept with 10% traffic
- **Phase 2 (3-6 mo):** Expand to 50% if <5% error rate
- **Phase 3 (6-12 mo):** Full migration if user satisfaction +20%
- **Rollback Plan:** Instant rollback available if errors >10%

### Trade-off Analysis Synthesis
**When:** No clear winner, all options have pros/cons

**Method:**
1. List all pros and cons for each option
2. Categorize by impact (high/medium/low)
3. Quantify when possible
4. Identify deal-breakers
5. Make recommendation with caveats

**Example:**
```
Option A: PostgreSQL
✅ Team expertise (HIGH impact)
✅ Low operational complexity (MEDIUM)
❌ Write performance insufficient (HIGH)
❌ Horizontal scaling limited (MEDIUM)

Option B: Cassandra
✅ Excellent write performance (HIGH)
✅ Linear scalability (HIGH)
❌ Team has zero experience (HIGH)
❌ Very high operational complexity (HIGH)

Synthesis: Start with PostgreSQL (leverage expertise),
migrate to Cassandra if performance becomes bottleneck
after optimization (6-12 months)
```

## Documentation Best Practices

### Research Report Structure

```markdown
# Research Topic

## Executive Summary
- Research question
- Key findings (3-5 bullets)
- Recommendation
- Expected impact

## Research Methodology
- Question formulation
- Source selection criteria
- Number of sources consulted
- Average credibility score

## Findings
### Finding 1: [Topic]
- Source: [URL]
- Credibility: [Score]
- Key insight: [Summary]

### Finding 2: [Topic]
...

## Source Evaluation
- Credibility matrix
- Consensus analysis
- Conflicts identified

## Synthesis
- Decision matrix / phased strategy / trade-off analysis
- Recommendation with justification
- Risk mitigation
- Success metrics

## Implementation Plan
- Phased approach
- Timeline
- Resource requirements
- Success criteria

## Memory Storage
```bash
npx claude-flow@alpha memory store \
  --key "research/[topic]" \
  --value "[summary]" \
  --metadata '{"sources":N,"credibility":XX%,"decision":"[choice]"}'
```
```

### Memory-MCP Storage Strategy

**Short-term Research (24h retention):**
- Quick research questions
- Temporary exploration
- Prototyping decisions

**Mid-term Research (7d retention):**
- Feature implementation research
- Technical decision-making
- Comparative analysis

**Long-term Research (30d+ retention):**
- Architectural decisions
- Technology stack choices
- Major migrations
- Strategic research

**Tagging Protocol:**
```javascript
{
  who: "researcher",
  when: "2025-11-02T10:30:00Z",
  project: "database-selection",
  intent: "research",
  sources: 8,
  credibility: 90.6,
  decision: "timescaledb-phase1",
  keywords: ["database", "scalability", "performance"]
}
```

## Common Research Pitfalls

### Pitfall 1: Confirmation Bias
**Problem:** Only seeking sources that confirm initial hypothesis

**Solution:**
- Actively seek conflicting viewpoints
- Include sources with different conclusions
- Document conflicts explicitly

### Pitfall 2: Recency Bias
**Problem:** Overweighting newest information, ignoring fundamentals

**Solution:**
- Balance recent trends with established principles
- Check if "new" approach has proven track record
- Validate with production case studies (not just benchmarks)

### Pitfall 3: Authority Worship
**Problem:** Trusting big names without verification

**Solution:**
- Verify even authoritative sources
- Cross-check with independent sources
- Distinguish vendor marketing from technical truth

### Pitfall 4: Analysis Paralysis
**Problem:** Endless research, never making decision

**Solution:**
- Set time box (Level 1: 1h, Level 2: 4h, Level 3: 2d)
- Define "good enough" threshold (e.g., 85% confidence)
- Use phased approach (decide Phase 1, defer Phase 2)

### Pitfall 5: Ignoring Context
**Problem:** Applying research findings without considering specific context

**Solution:**
- Evaluate team expertise
- Consider organizational constraints
- Assess risk tolerance
- Factor in time/budget limitations

## Research Tool Recommendations

### Web Research
- **Gemini Search**: Grounded web search with source citations
- **Google Scholar**: Academic papers, citations
- **GitHub**: Code examples, issue discussions
- **HackerNews**: Community discussions (take with grain of salt)

### Benchmarking
- **YCSB**: Database benchmarks
- **TechEmpower**: Web framework benchmarks
- **Phoronix**: Hardware/software performance testing

### Cost Analysis
- **Cloud pricing calculators**: AWS, GCP, Azure
- **TCO calculators**: Gartner, Forrester reports

### Technical Documentation
- **MDN Web Docs**: Web technologies
- **Official docs**: Language/framework documentation
- **RFCs**: Internet standards
- **W3C specs**: Web standards

### Memory & Context
- **Memory-MCP**: Persistent research storage
- **Claude-Flow Memory**: Cross-session context

## Next Steps

After research completion:
1. Document findings in structured report
2. Store in Memory-MCP with proper tagging
3. Share with team for feedback
4. Create implementation plan
5. Define success metrics
6. Schedule follow-up review (validate research accuracy)
