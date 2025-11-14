# Source Credibility Analyzer - SOP v2 (Optimized)

**Version**: 2.0 (Production-Ready)
**Type**: Standalone Tool
**Agent**: analyst
**Duration**: 5-15 minutes per source
**Purpose**: Automate credibility/bias/priority scoring using program-of-thought rubrics

---

## Overview

Standalone tool that evaluates academic/research sources with transparent scoring. Outputs structured JSON with credibility (1-5), bias (1-5), and priority (1-5) scores plus explanations. Can be used independently OR within general-research-workflow Step 3.

**Key Innovation**: Program-of-thought rubrics make scoring auditable and reproducible.

---

## When to Use This Tool

**✅ USE FOR**:
- Evaluating research sources for academic projects
- Automating Step 3 of general-research-workflow
- Scoring large batches of sources consistently
- Getting second opinion on source quality

**❌ DO NOT USE FOR**:
- Entertainment content (movies, novels) - not designed for this
- Already know source is high/low quality - skip straight to reading/skipping
- Source is unique/irreplaceable (e.g., only source on obscure topic) - read anyway

**Decision Tree**: If source evaluation takes >10 min manually → use this tool

---

## Input Requirements

### ✅ Required Metadata
- `title`: Source title
- `author`: Author name(s)
- `year`: Publication year (1500-2025)
- `venue`: Publication venue (journal, publisher, website)
- `type`: Source type (see Edge Cases below)

### ⚠️ Optional Metadata (Improves Accuracy)
- `citations`: Citation count (Google Scholar, etc.)
- `doi`: DOI or persistent identifier
- `url`: Web URL
- `abstract`: Abstract or summary text
- `institution`: Author's affiliation
- `credentials`: Author's credentials (PhD, etc.)

### 💡 Tip
More optional metadata = more accurate scoring (especially citations and credentials)

---

## Agent Workflow

### Agent: analyst

**Sequential Steps**:

| Step | Objective | Deliverable | Duration | Quality Gate |
|------|-----------|-------------|----------|--------------|
| 0 | Validate inputs | Confirmed metadata | 30 sec | ✅ Required fields present |
| 0.5 | Classify source type | Source category | 1 min | ✅ Type assigned (academic/institutional/general) |
| 1 | Calculate credibility | Score 1-5 + explanation | 2-5 min | ✅ Score justified with rules |
| 2 | Calculate bias | Score 1-5 + explanation | 2-5 min | ✅ Score justified with rules |
| 3 | Calculate priority | Score 1-5 + explanation | 1-3 min | ✅ Score justified with rules |
| 4 | Resolve conflicts | Final recommendation | 1 min | ✅ Logic applied correctly |
| 5 | Generate output | JSON + storage | 1 min | ✅ All scores + stored |

---

## Step 0: Validate Input Metadata

**Objective**: Ensure required metadata is present and complete

**Procedure**:
1. Check for ✅ required fields: `title`, `author`, `year`, `venue`, `type`
2. If missing critical field → Return error with field name
3. If ⚠️ optional fields present → Note for enhanced scoring
4. Validate data types:
   - Year: Integer 1500-2025
   - Type: One of allowed types (see Step 0.5)

**Deliverable**: Validated metadata object

**Quality Gate 0**:
- **GO**: All required fields present, year valid
- **NO-GO**: Missing required field → Return error to user

---

## Step 0.5: Classify Source Type (Edge Case Handling)

**Objective**: Assign source to appropriate category for rubric selection

**Edge Case Decision Tree**:

```
Is source peer-reviewed?
├─ ✅ Yes → ACADEMIC (journals, academic books, conference proceedings)
│
├─ ❌ No → Is source from recognized institution?
    ├─ ✅ Yes → INSTITUTIONAL (government reports, think tank papers, university press)
    │
    └─ ❌ No → Is source verifiable/documented?
        ├─ ✅ Yes → GENERAL (Wikipedia, reputable news, expert blogs)
        └─ ❌ No → UNVERIFIED (personal blogs, social media, unknown sites)
```

**Source Categories and Rubric Adjustments**:

| Category | Examples | Credibility Baseline | Notes |
|----------|----------|---------------------|-------|
| **ACADEMIC** | Peer-reviewed journals, academic books, conference papers | Start at 4 | Standard rubric applies |
| **INSTITUTIONAL** | Government reports, white papers, university press (non-peer-reviewed) | Start at 3 | Check funding source for bias |
| **GENERAL** | Wikipedia, reputable news (NYT, BBC), expert blogs (Gwern, Scott Alexander) | Start at 3 | Verify against other sources |
| **PREPRINTS** | arXiv, bioRxiv, SSRN | Start at 3 | High credibility but unverified, verify claims |
| **UNVERIFIED** | Personal blogs, social media, unknown sites | Start at 2 | Use with extreme caution |

**Special Cases**:

1. **Wikipedia**: Credibility 3 (verifiable, crowd-sourced), Bias 4 (NPOV policy), Priority 2 (background only, not citable)
2. **Preprints**: Credibility 3 (not peer-reviewed yet), Bias 4 (assume good faith), Priority = depends on recency and relevance
3. **Gray Literature** (government reports, NGO papers): Credibility 3-4, Bias = check funding source, Priority = depends on topic
4. **Expert Blogs** (e.g., Gwern, LessWrong): Credibility 3 (no formal review), Bias 3-4, Priority = depends on expertise match

**💡 Tip**: When unsure, classify as GENERAL and note uncertainty in output

**Deliverable**: Source category (ACADEMIC | INSTITUTIONAL | GENERAL | PREPRINTS | UNVERIFIED)

**Quality Gate 0.5**:
- **GO**: Category assigned based on decision tree
- **NO-GO**: Ambiguous category → Default to GENERAL and flag uncertainty

---

## Step 1: Calculate Credibility Score (1-5)

**Objective**: Assess source trustworthiness using program-of-thought rubric

**Program-of-Thought Rubric**:

**Baseline Score** (from Step 0.5):
- ACADEMIC: Start at 4
- INSTITUTIONAL / GENERAL / PREPRINTS: Start at 3
- UNVERIFIED: Start at 2

**Add +1 for EACH** (max +3):
- ✅ Peer-reviewed publication (academic journal, university press)
- ✅ Author has PhD or recognized expertise in field
- ✅ Cites primary sources and provides extensive references (≥20 citations)
- ✅ Published by top-tier institution (Ivy League, top publisher)
- ✅ High citation count (≥100 for papers, ≥500 for books)
- ✅ Has DOI or persistent identifier

**Subtract -1 for EACH** (max -3):
- ❌ Self-published or vanity press
- ❌ No author credentials listed or anonymous
- ❌ No citations or references provided
- ❌ Known conflicts of interest (industry-funded study on own product)
- ❌ Published on unmoderated platform (personal blog, social media)
- ❌ Retracted or corrected after publication

**Borderline Score Policy**:
- If final score = 2.5 or 3.5 → **Round DOWN** (conservative for credibility)
- Example: 2.5 → 2, 3.5 → 3

**Final Credibility Score**: [1-5, capped at boundaries]

**Procedure**:
1. Determine baseline from Step 0.5
2. Apply each applicable rule (+1 or -1)
3. Sum adjustments
4. Apply borderline rounding if needed
5. Cap at 1 (minimum) and 5 (maximum)
6. Generate explanation listing: Baseline + Rules Applied + Final Score

**Example Calculation**:
```
Source: "Machine Learning: A Probabilistic Perspective" by Kevin Murphy (MIT Press, 2012)
Category: ACADEMIC

Baseline: 4 (Academic source)
+1 (Published by MIT Press - top-tier university press)
+1 (Author: PhD, Google Research - recognized expertise)
+1 (15,000 citations - highly influential)
= 7 → capped at 5

Final Credibility: 5/5
Explanation: "Academic textbook baseline 4, +1 MIT Press, +1 PhD author with expertise, +1 15k citations = 7 capped at 5. Authoritative source."
```

**Deliverable**: Credibility score (1-5) + explanation showing baseline + rules + final

**Quality Gate 1**:
- **GO**: Score 1-5, explanation lists baseline and applied rules
- **NO-GO**: Score outside range or missing explanation → Recalculate

---

## Step 2: Calculate Bias Score (1-5)

**Objective**: Assess source objectivity and potential bias

**Program-of-Thought Rubric**:

**Baseline Score** (all categories):
- Start at 3 (Neutral)

**Special Baseline Adjustments**:
- **Primary sources** (historical documents, datasets): Start at 5 (factual records, minimal interpretation)
- **Opinion pieces** (editorials, op-eds): Start at 2 (explicitly opinionated)

**Add +1 for EACH** (max +3):
- ✅ Academic/scholarly source with peer review
- ✅ Presents multiple perspectives or counterarguments
- ✅ Clearly distinguishes facts from opinions/interpretations
- ✅ Transparent about methodology and limitations
- ✅ No financial conflicts of interest disclosed
- ✅ Published by neutral/academic institution (not advocacy group)

**Subtract -1 for EACH** (max -3):
- ❌ Advocacy organization or political think tank
- ❌ Funded by interested party with conflicts (Exxon-funded climate paper)
- ❌ One-sided presentation (no counterarguments or limitations discussed)
- ❌ Opinion piece or editorial without clear labeling
- ❌ Sensationalist language or clickbait title
- ❌ Known partisan publication (Breitbart, Jacobin)

**Borderline Score Policy**:
- If final score = 2.5 or 3.5 → **Round UP** (benefit of doubt for bias)
- Example: 2.5 → 3, 3.5 → 4

**Final Bias Score**: [1-5, where 5 = least biased, 1 = most biased]

**Procedure**:
1. Determine baseline (3, or adjusted for primary/opinion sources)
2. Apply each applicable rule (+1 or -1)
3. Sum adjustments
4. Apply borderline rounding if needed
5. Cap at 1 (minimum) and 5 (maximum)
6. Generate explanation

**Example Calculation**:
```
Source: "Climate Change Impacts Study" by Heartland Institute (2021)
Category: INSTITUTIONAL (think tank)

Baseline: 3
-1 (Published by Heartland Institute - known climate denial advocacy group)
-1 (Funded by fossil fuel industry - conflicts of interest)
-1 (One-sided presentation - dismisses scientific consensus without balanced counterarguments)
= 0 → capped at 1

Final Bias: 1/5 (highly biased)
Explanation: "Institutional baseline 3, -1 advocacy org, -1 industry funding, -1 one-sided = 0 capped at 1. Strong bias, verify all claims against independent sources."
```

**Deliverable**: Bias score (1-5) + explanation showing baseline + rules + final

**Quality Gate 2**:
- **GO**: Score 1-5, explanation lists baseline and applied rules
- **NO-GO**: Score outside range or missing explanation → Recalculate

---

## Step 3: Calculate Priority Score (1-5)

**Objective**: Assess reading priority for research project

**Program-of-Thought Rubric**:

**Baseline Score**:
- Start at 3 (Neutral)

**Add +1 for EACH** (max +3):
- ✅ **Recent** publication (≤5 years for empirical fields, ≤10 years for historical)
- ✅ **Directly relevant** to research question (addresses core topic explicitly)
- ✅ **Highly cited** (≥50 citations for papers, ≥100 for books)
- ✅ **Primary source** or seminal work (foundational to field, e.g., Shannon 1948 for information theory)
- ✅ **Recommended** by expert or cited in multiple key papers
- ✅ **Comprehensive** coverage (textbook, review article, meta-analysis)

**Subtract -1 for EACH** (max -3):
- ❌ **Outdated** (>20 years old for empirical fields, unless seminal work)
- ❌ **Tangentially relevant** (only mentions topic in passing, not core focus)
- ❌ **Low credibility** (<3 from Step 1)
- ❌ **High bias** (<3 from Step 2)
- ❌ **Redundant** (covers same ground as already-read higher-priority source)
- ❌ **Too narrow** (hyper-specialized subsection, not useful for overview)

**Special Considerations**:
- **Classic works**: Even if old (>50 years), retain high priority if seminal (Darwin's *Origin of Species* for evolution research)
- **Breadth vs depth**: Introductory overviews = lower priority than specialized deep dives (unless building foundation)

**Borderline Score Policy**:
- If final score = 2.5 or 3.5 → **Round UP** (favor reading when uncertain)
- Example: 2.5 → 3, 3.5 → 4

**Final Priority Score**: [1-5, where 5 = read first, 1 = skip]

**Procedure**:
1. Start at 3
2. Apply each applicable rule (+1 or -1)
3. **Check credibility/bias from Steps 1-2**: Auto-subtract if credibility <3 or bias <3
4. Sum adjustments
5. Apply borderline rounding if needed
6. Cap at 1 (minimum) and 5 (maximum)
7. Generate explanation

**Example Calculation**:
```
Source: "Byzantium and the Renaissance" by N.G. Wilson (Cambridge, 1992)
Category: ACADEMIC
For Renaissance research project

Baseline: 3
-1 (Published 33 years ago - outdated for empirical work BUT historical analysis)
+1 (Directly addresses research question - Byzantine influence on Renaissance)
+1 (Cited 250+ times - highly influential in field)
+1 (Primary scholarship on topic - seminal work)
= 5

Final Priority: 5/5 (read first)
Explanation: "Baseline 3, -1 age (offset by seminal status), +1 directly relevant, +1 highly cited, +1 primary scholarship = 5. Despite age, essential reading for Byzantine-Renaissance research."
```

**Deliverable**: Priority score (1-5) + explanation showing baseline + rules + credibility/bias check + final

**Quality Gate 3**:
- **GO**: Score 1-5, explanation references credibility/bias from Steps 1-2
- **NO-GO**: Score ignores credibility/bias → Recalculate with penalty

---

## Step 4: Resolve Conflicting Scores

**Objective**: Handle edge cases where credibility/bias/priority conflict

**Conflict Resolution Logic**:

### Conflict Type 1: High Credibility + High Bias
**Example**: Peer-reviewed paper in advocacy journal (Credibility 4, Bias 2)

**Resolution**:
- If Credibility ≥4 AND Bias ≤2 → Recommendation: **VERIFY_CLAIMS**
- Reasoning: "Credible source but biased presentation. Read critically and verify claims against independent sources."

### Conflict Type 2: Low Credibility + Low Bias
**Example**: Anonymous blog with balanced presentation (Credibility 2, Bias 4)

**Resolution**:
- If Credibility ≤2 AND Bias ≥4 → Recommendation: **SKIP** (unless unique source)
- Reasoning: "Unbiased but not credible enough to rely on. Find authoritative alternative."

### Conflict Type 3: High Priority + Low Credibility
**Example**: Only source on obscure topic (Priority 5, Credibility 2)

**Resolution**:
- If Priority ≥4 AND Credibility ≤2 → Recommendation: **READ_LATER** with warning
- Reasoning: "Highly relevant but low credibility. Read as last resort, verify everything."

### Conflict Type 4: High Credibility + Low Priority
**Example**: Tangentially related textbook (Credibility 5, Priority 2)

**Resolution**:
- If Credibility ≥4 AND Priority ≤2 → Recommendation: **READ_LATER** (background material)
- Reasoning: "Authoritative but not directly relevant. Read for context if time permits."

**Default Recommendation Matrix**:

| Credibility | Bias | Priority | Recommendation |
|-------------|------|----------|----------------|
| ≥4 | ≥3 | ≥4 | READ_FIRST (ideal source) |
| ≥3 | ≥3 | ≥3 | READ_LATER (solid source) |
| ≥4 | ≤2 | ANY | VERIFY_CLAIMS (credible but biased) |
| ≤2 | ANY | ANY | SKIP (not credible) |
| ANY | ≤2 | ≥4 | VERIFY_CLAIMS (needed but biased) |

**Deliverable**: Final recommendation (READ_FIRST | READ_LATER | VERIFY_CLAIMS | SKIP) + reasoning

**Quality Gate 4**:
- **GO**: Recommendation matches matrix logic, conflicts explained
- **NO-GO**: Recommendation doesn't match scores → Reapply logic

---

## Step 5: Generate Structured Output

**Objective**: Create complete scoring report in standardized JSON format

**Output Format**:

```json
{
  "source": {
    "title": "[Title]",
    "author": "[Author]",
    "year": [YYYY],
    "venue": "[Venue]",
    "type": "[Type]",
    "category": "[ACADEMIC | INSTITUTIONAL | GENERAL | PREPRINTS | UNVERIFIED]",
    "doi": "[DOI if available]",
    "url": "[URL if available]"
  },
  "scores": {
    "credibility": {
      "score": [1-5],
      "explanation": "[Baseline + rules applied + final score]"
    },
    "bias": {
      "score": [1-5],
      "explanation": "[Baseline + rules applied + final score]"
    },
    "priority": {
      "score": [1-5],
      "explanation": "[Baseline + rules applied + credibility/bias check + final score]"
    }
  },
  "recommendation": {
    "action": "[READ_FIRST | READ_LATER | VERIFY_CLAIMS | SKIP]",
    "reason": "[1-2 sentences justifying recommendation based on scores]",
    "conflicts": "[If any conflicts, explain resolution logic]"
  },
  "metadata": {
    "analyzed_by": "source-credibility-analyzer",
    "timestamp": "[ISO8601]",
    "version": "2.0"
  }
}
```

**Storage**:
```bash
npx claude-flow@alpha memory store \
  --key "source-analysis/[project]/[source-id]" \
  --value "[JSON output]" \
  --tags "WHO=analyst,WHEN=[timestamp],PROJECT=[topic],WHY=source-scoring,CREDIBILITY=[score],BIAS=[score],PRIORITY=[score],RECOMMENDATION=[action]"
```

**Deliverable**: Complete JSON + Memory MCP storage confirmation

**Quality Gate 5**:
- **GO**: All scores present, explanations complete, stored successfully
- **NO-GO**: Missing field or storage failed → Regenerate and retry

---

## Success Metrics

### Quantitative
- ✅ All 3 scores calculated (credibility, bias, priority)
- ✅ All scores between 1-5 (no invalid values)
- ✅ Explanations provided showing calculations
- ✅ Recommendation matches decision matrix
- ✅ Execution time 5-15 minutes per source
- ✅ Output stored in Memory MCP with tags

### Qualitative
- ✅ Scores match manual scoring within ±1 point (self-consistency)
- ✅ Explanations clearly justify scores with explicit rules
- ✅ Recommendation is actionable (READ_FIRST vs SKIP)
- ✅ Tool handles edge cases (Wikipedia, preprints, gray literature)
- ✅ Conflicts resolved with transparent logic

---

## Error Handling

| Failure Mode | Gate | Resolution |
|--------------|------|------------|
| **Missing required metadata** | 0 | Return error with field name |
| **Invalid year (e.g., 3000)** | 0 | Reject as invalid, request correction |
| **Score outside 1-5** | 1-3 | Recalculate with correct capping |
| **No explanation provided** | 1-3 | Regenerate with explicit rule listing |
| **Borderline score uncertainty** | 1-3 | Apply rounding policy (down for credibility, up for bias/priority) |
| **Ambiguous source type** | 0.5 | Default to GENERAL, flag uncertainty in output |
| **Conflicting scores** | 4 | Apply conflict resolution matrix |
| **Recommendation mismatch** | 4 | Reapply matrix logic, document conflict |
| **Memory MCP storage fails** | 5 | Retry 3x, fallback to local JSON export |

---

## Integration

### Standalone Usage
```bash
# Direct invocation
Skill("source-credibility-analyzer") + {
  "title": "...",
  "author": "...",
  "year": 2020,
  "venue": "...",
  "type": "journal article"
}
```

### Within general-research-workflow
- Called during **Step 3** (Source Classification)
- Automates manual scoring from program-of-thought rubrics
- Reduces Step 3 time: 30-60 min → 5-15 min per source
- Outputs feed directly into Step 4 (Reading Plan)

### Output Feeds To
- **general-research-workflow Step 4**: Prioritizes sources for reading
- **academic-reading-workflow**: Decides which sources to annotate deeply
- **Memory MCP**: Persistent scoring for future reference

---

## Complete Examples

See `examples/scoring-examples.md` for 5 complete end-to-end examples:
1. ✅ Academic paper (Credibility 5, Bias 5, Priority 4) → READ_FIRST
2. ✅ Think tank report (Credibility 3, Bias 1, Priority 3) → VERIFY_CLAIMS
3. ⚠️ Preprint (Credibility 3, Bias 4, Priority 5) → READ_FIRST (verify claims)
4. ⚠️ Wikipedia article (Credibility 3, Bias 4, Priority 2) → READ_LATER (background)
5. ❌ Blog post (Credibility 2, Bias 2, Priority 1) → SKIP

Each example shows: Input → Step-by-step calculations → Output JSON → Explanation

---

## Design Notes

**Why This is an SOP**:
- Sequential agent workflow (Steps 0-5)
- Clear deliverables and handoffs
- Quality Gates enforce correctness
- Procedural guidance, not technical implementation

**Why Single Agent**:
- Scoring is analytical, not creative
- Program-of-thought rubrics are deterministic
- No need for multi-agent coordination overhead

**Why Program-of-Thought**:
- Explicit calculations make scores auditable
- Agents reproduce scoring consistently
- Users can verify logic and adjust rubrics
- Reduces subjective judgment variability

**Changes from v1**:
- ✅ Added Step 0.5 (edge case classification)
- ✅ Added 5 complete examples (separate file)
- ✅ Added conflict resolution logic (Step 4)
- ✅ Added visual markers (✅/⚠️/💡/🚨)
- ✅ Added borderline score rounding policy
- ✅ Clarified baseline scores by category

---

**Next**: Build complete skill with skill-forge (SKILL.md, process diagram, examples, references, README)
