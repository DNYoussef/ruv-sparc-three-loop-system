# Source Credibility Evaluation Framework

## Overview

Systematic framework for evaluating the credibility, reliability, and relevance of information sources in software engineering research.

## Credibility Scoring System

### Overall Credibility Score Formula

```
Credibility Score = (Authority × 0.30) +
                   (Recency × 0.20) +
                   (Methodology × 0.25) +
                   (Bias Assessment × 0.15) +
                   (Validation × 0.10)
```

**Scoring Scale:** 0-100%
- 90-100%: Highly credible (authoritative, use with high confidence)
- 75-89%: Credible (reliable, cross-check key claims)
- 60-74%: Moderate credibility (useful for initial exploration, verify independently)
- Below 60%: Low credibility (use with extreme caution, not for decisions)

## Dimension 1: Authority (30% weight)

### Authority Scoring Rubric

**Level 5 (90-100 points): Authoritative Expert**
- Peer-reviewed academic publications
- Browser vendor technical documentation (Google, Mozilla, Apple)
- Official standards bodies (W3C, IETF, ISO)
- Production engineering blogs from FAANG companies
- Authors with 10+ years relevant experience

**Examples:**
- ACM/IEEE papers
- Mozilla Developer Network (MDN)
- IETF RFCs
- Google V8 team blog posts
- Martin Fowler articles

**Level 4 (70-89 points): Recognized Practitioner**
- Technical blogs from established companies
- Authors with 5-10 years experience
- Well-known conference talks
- Industry survey reports (Gartner, Forrester)

**Examples:**
- Netflix Engineering Blog
- Thoughtworks Technology Radar
- AWS Architecture Blog
- Stack Overflow official documentation

**Level 3 (50-69 points): Experienced Developer**
- Personal technical blogs with good reputation
- Authors with 3-5 years experience
- Tutorial websites with quality content
- Medium articles from verified authors

**Examples:**
- CSS-Tricks
- Smashing Magazine
- Dev.to verified authors

**Level 2 (30-49 points): Community Contributor**
- Anonymous blog posts
- Stack Overflow answers (not accepted)
- Reddit discussions
- Unverified authors

**Level 1 (0-29 points): Unverified Source**
- Random blogs with no credentials
- No author attribution
- AI-generated content without verification
- Marketing materials disguised as technical content

### Authority Verification Checklist

- [ ] Author identified by name
- [ ] Author's background/credentials stated
- [ ] Author works for reputable organization
- [ ] Author has published other credible work
- [ ] Content reviewed or fact-checked
- [ ] Organization has domain expertise
- [ ] Content cited by other authoritative sources

## Dimension 2: Recency (20% weight)

### Recency Scoring Rubric

**Context Matters:** Technology domain affects recency requirements

**Fast-Moving Technologies (JavaScript frameworks, cloud services, browser APIs):**
- 0-6 months: 100 points (current)
- 6-12 months: 80 points (recent)
- 1-2 years: 60 points (acceptable with verification)
- 2-3 years: 40 points (outdated, verify current status)
- 3+ years: 20 points (likely obsolete)

**Stable Technologies (algorithms, design patterns, databases):**
- 0-1 year: 100 points
- 1-3 years: 90 points
- 3-5 years: 75 points
- 5-7 years: 60 points
- 7+ years: Check if principles still apply

**Fundamental Concepts (complexity analysis, computer science theory):**
- Recency less critical (10+ year old papers may still be authoritative)
- Focus on whether concepts remain applicable

### Last Updated Indicators

**Explicit Update Dates:**
- [ ] Last updated date clearly stated
- [ ] Update history available
- [ ] Regular maintenance evident

**Implicit Update Indicators:**
- [ ] References current versions (e.g., React 18, not React 16)
- [ ] Mentions recent events (conferences, releases)
- [ ] Links to current documentation
- [ ] Code examples use modern syntax

### Staleness Warning Signs

- ❌ References deprecated technologies
- ❌ Broken links to external resources
- ❌ Code examples no longer work
- ❌ Contradicts current official documentation
- ❌ No update date provided

## Dimension 3: Methodology (25% weight)

### Methodology Scoring Rubric

**Level 5 (90-100 points): Rigorous Scientific Method**
- Peer-reviewed research with reproducible methodology
- Controlled experiments with statistical analysis
- Open-source benchmarks with documented setup
- Detailed methodology section
- Limitations explicitly acknowledged

**Indicators:**
- [ ] Hypothesis clearly stated
- [ ] Experimental design detailed
- [ ] Variables controlled
- [ ] Sample size specified
- [ ] Statistical significance reported
- [ ] Reproducibility instructions provided
- [ ] Raw data available
- [ ] Limitations discussed

**Level 4 (70-89 points): Systematic Testing**
- Structured benchmarks with clear setup
- Multiple test scenarios
- Consistent measurement approach
- Results presented with context

**Indicators:**
- [ ] Test environment described
- [ ] Test cases documented
- [ ] Measurement tools specified
- [ ] Multiple runs averaged
- [ ] Outliers explained

**Level 3 (50-69 points): Anecdotal Evidence**
- Real-world production experience
- Case studies with outcomes
- Personal experience with lessons learned

**Indicators:**
- [ ] Context provided (scale, constraints)
- [ ] Results documented
- [ ] Trade-offs discussed
- [ ] Not just success stories

**Level 2 (30-49 points): Opinion/Recommendation**
- Expert opinion based on experience
- Best practices guidance
- No empirical testing

**Indicators:**
- [ ] Reasoning explained
- [ ] Alternative views acknowledged
- [ ] Context limitations stated

**Level 1 (0-29 points): Unsubstantiated Claims**
- No methodology
- No evidence provided
- Vague or unverifiable claims

**Red Flags:**
- ❌ "Everyone knows..."
- ❌ "Obviously the best..."
- ❌ No supporting data
- ❌ Absolute statements without qualification

## Dimension 4: Bias Assessment (15% weight)

### Bias Scoring Rubric

**Level 5 (90-100 points): Minimal Bias**
- No commercial interest
- Multiple viewpoints presented
- Trade-offs clearly discussed
- Limitations acknowledged
- Competing solutions fairly compared

**Level 4 (70-89 points): Low Bias**
- Potential conflicts disclosed
- Balanced presentation
- Not purely promotional
- Acknowledges alternatives

**Level 3 (50-69 points): Moderate Bias**
- Vendor content with technical value
- Clear commercial interest but informative
- One-sided but transparent

**Level 2 (30-49 points): High Bias**
- Marketing disguised as technical content
- Cherry-picked data
- Ignores competing solutions
- Overstates benefits

**Level 1 (0-29 points): Extreme Bias**
- Purely promotional
- Misleading comparisons
- No acknowledgment of limitations
- Attacks competitors unfairly

### Bias Indicators Checklist

**Commercial Bias:**
- [ ] Author works for vendor of discussed product
- [ ] Sponsored content
- [ ] Affiliate links present
- [ ] Comparison only includes vendor's product favorably

**Confirmation Bias:**
- [ ] Only presents evidence supporting one viewpoint
- [ ] Ignores contradictory evidence
- [ ] Dismisses alternatives without fair evaluation

**Recency Bias:**
- [ ] "New = better" mentality
- [ ] Dismisses proven solutions as "legacy"
- [ ] Overemphasis on latest trends

**Survivorship Bias:**
- [ ] Only success stories, no failures
- [ ] No mention of companies that tried and failed
- [ ] Ignores total cost of ownership

### Vendor Content Evaluation

**When evaluating vendor-provided content:**

**Acceptable Vendor Content:**
- ✅ Technical deep-dives into product internals
- ✅ Performance characteristics with methodology
- ✅ Architecture explanations
- ✅ Integration guides
- ✅ Transparent about limitations

**Questionable Vendor Content:**
- ⚠️ Comparison benchmarks (verify independently)
- ⚠️ Case studies (may be cherry-picked)
- ⚠️ "Best practices" that favor their product
- ⚠️ Total cost of ownership claims

**Untrustworthy Vendor Content:**
- ❌ "X is always better than Y" claims
- ❌ Benchmarks with opaque methodology
- ❌ Unfair comparison setups
- ❌ FUD (Fear, Uncertainty, Doubt) about competitors

## Dimension 5: Validation (10% weight)

### Validation Scoring Rubric

**Level 5 (90-100 points): Extensively Validated**
- Confirmed by 5+ independent authoritative sources
- Real-world production evidence
- Community consensus
- Reproducible by third parties

**Level 4 (70-89 points): Well-Validated**
- Confirmed by 3-4 independent sources
- Multiple case studies
- Independently reproduced

**Level 3 (50-69 points): Partially Validated**
- Confirmed by 1-2 independent sources
- Anecdotal evidence
- Plausible but not proven

**Level 2 (30-49 points): Unvalidated**
- Single source
- No independent confirmation
- Theoretical only

**Level 1 (0-29 points): Contradicted**
- Conflicts with authoritative sources
- Proven incorrect
- Refuted by evidence

### Validation Techniques

**Cross-Reference Validation:**
1. Find 3-5 independent sources on same topic
2. Extract key claims from each
3. Identify consensus claims (4+ agree)
4. Investigate conflicts
5. Weight by source credibility

**Experimental Validation:**
1. Reproduce benchmark in controlled environment (E2B sandbox)
2. Compare results with source's claims
3. Validate within ±20% margin (accounting for hardware differences)

**Production Validation:**
1. Find real-world case studies
2. Verify claimed benefits match actual outcomes
3. Check if companies still use solution (or abandoned it)

**Community Validation:**
1. Check discussions on HackerNews, Reddit, Stack Overflow
2. Look for consensus or significant disagreement
3. Identify experienced practitioners' opinions

## Practical Application Examples

### Example 1: Evaluating Database Performance Benchmark

**Source:** ScyllaDB Blog - "ScyllaDB 3x faster than Cassandra"

**Authority (30%):**
- Vendor blog: Level 2 (40 points) → 40 × 0.30 = 12

**Recency (20%):**
- Published 2024: 100 points → 100 × 0.20 = 20

**Methodology (25%):**
- Methodology documented but vendor-controlled setup: Level 3 (60 points) → 60 × 0.25 = 15

**Bias (15%):**
- Vendor promoting own product: Level 2 (30 points) → 30 × 0.15 = 4.5

**Validation (10%):**
- Some independent confirmation but not extensive: Level 3 (50 points) → 50 × 0.10 = 5

**Total Credibility Score:** 12 + 20 + 15 + 4.5 + 5 = **56.5% (Moderate)**

**Verdict:** Useful for initial exploration, but verify independently before making decisions. High bias risk (vendor content).

### Example 2: Evaluating Academic Paper

**Source:** "Not So Fast: Analyzing the Performance of WebAssembly vs. Native Code" (ASPLOS 2019)

**Authority (30%):**
- Peer-reviewed conference paper (UMass, Stanford): Level 5 (95 points) → 95 × 0.30 = 28.5

**Recency (20%):**
- Published 2019 (6 years ago) but fundamental analysis: 70 points → 70 × 0.20 = 14

**Methodology (25%):**
- Rigorous scientific method, reproducible: Level 5 (98 points) → 98 × 0.25 = 24.5

**Bias (15%):**
- Academic research, no commercial interest: Level 5 (95 points) → 95 × 0.15 = 14.25

**Validation (10%):**
- Widely cited, independently reproduced: Level 5 (95 points) → 95 × 0.10 = 9.5

**Total Credibility Score:** 28.5 + 14 + 24.5 + 14.25 + 9.5 = **90.75% (Highly Credible)**

**Verdict:** Highly credible source. Use with confidence. Cross-check recency (WASM evolved since 2019).

### Example 3: Evaluating Production Case Study

**Source:** Discord Engineering Blog - "How Discord Stores Billions of Messages"

**Authority (30%):**
- Production engineering from established company: Level 4 (85 points) → 85 × 0.30 = 25.5

**Recency (20%):**
- Published 2023: 90 points → 90 × 0.20 = 18

**Methodology (25%):**
- Real production experience with metrics: Level 3 (75 points) → 75 × 0.25 = 18.75

**Bias (15%):**
- Transparent about challenges, not promotional: Level 4 (85 points) → 85 × 0.15 = 12.75

**Validation (10%):**
- Confirmed by other large-scale case studies: Level 4 (80 points) → 80 × 0.10 = 8

**Total Credibility Score:** 25.5 + 18 + 18.75 + 12.75 + 8 = **83% (Credible)**

**Verdict:** Credible real-world evidence. Context-specific (Discord's scale), adapt to your needs.

## Red Flags Checklist

### Immediate Rejection Criteria

- ❌ No author attribution
- ❌ Claims too good to be true ("10x faster with zero effort")
- ❌ Contradicts fundamental principles
- ❌ No evidence or methodology provided
- ❌ Exclusively promotional language
- ❌ Attacks competitors with FUD
- ❌ Presents opinion as fact
- ❌ "Secret technique experts don't want you to know"

### Caution Indicators

- ⚠️ Single source, no corroboration
- ⚠️ Vendor benchmark without independent validation
- ⚠️ Anecdotal evidence without data
- ⚠️ Cherry-picked success stories
- ⚠️ Outdated information (>3 years for fast-moving tech)
- ⚠️ Methodology unclear or missing
- ⚠️ Conflicts of interest not disclosed
- ⚠️ Absolute statements ("always", "never", "best")

## Source Evaluation Workflow

### Quick Evaluation (2-3 minutes per source)

1. **Identify Author:**
   - Who wrote it?
   - What are their credentials?
   - Do they have expertise?

2. **Check Date:**
   - When published/updated?
   - Is it current for the technology?

3. **Scan for Bias:**
   - Vendor content?
   - Balanced presentation?
   - Conflicts disclosed?

4. **Quick Credibility Score:**
   - Mental calculation using rubric
   - Accept (>75%), Caution (60-74%), Reject (<60%)

### Detailed Evaluation (10-15 minutes per source)

1. **Full Credibility Scoring:**
   - Calculate all 5 dimensions
   - Weighted total score
   - Document reasoning

2. **Methodology Review:**
   - Read methodology section
   - Assess reproducibility
   - Check for limitations

3. **Cross-Reference:**
   - Find 2-3 other sources on same topic
   - Compare claims
   - Identify consensus or conflicts

4. **Validation Check:**
   - Real-world evidence?
   - Independently reproduced?
   - Community consensus?

## Documentation Template

```markdown
## Source: [Title]
- **URL:** [link]
- **Author:** [name/organization]
- **Date:** [publication date]
- **Type:** [blog/paper/case study/documentation]

### Credibility Assessment
- **Authority:** X/100 (Level Y)
- **Recency:** X/100
- **Methodology:** X/100 (Level Y)
- **Bias:** X/100 (Level Y)
- **Validation:** X/100 (Level Y)
- **Total Score:** XX% ([Highly Credible|Credible|Moderate|Low])

### Key Findings
- [Finding 1]
- [Finding 2]
- [Finding 3]

### Limitations
- [Limitation 1]
- [Limitation 2]

### Validation Status
- [ ] Cross-referenced with 2+ independent sources
- [ ] Methodology reviewed
- [ ] Conflicts of interest assessed
- [ ] Recency verified

### Usage Recommendation
[How to use this source: high confidence / cross-check / initial exploration only / reject]
```

## Next Steps

After source evaluation:
1. Document credibility scores
2. Cross-reference high-credibility sources
3. Investigate conflicts between credible sources
4. Reject low-credibility sources (<60%)
5. Proceed to synthesis phase
