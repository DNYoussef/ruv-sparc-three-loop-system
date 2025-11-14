# Source Credibility Analyzer - SOP v1 (Draft)

**Version**: 1.0 (Draft)
**Type**: Standalone Tool
**Agent**: analyst
**Duration**: 5-15 minutes per source
**Purpose**: Automate credibility/bias/priority scoring from general-research-workflow Step 3

---

## Overview

Standalone tool that evaluates academic/research sources using program-of-thought rubrics. Outputs structured scores (1-5) for credibility, bias, and priority with explanations. Can be used independently OR within general-research-workflow Step 3.

**Design Principle**: Extract the scoring logic from general-research-workflow Step 3 into a reusable tool that any agent can invoke.

---

## Input Requirements

**Required Metadata**:
- `title`: Source title
- `author`: Author name(s)
- `year`: Publication year
- `venue`: Publication venue (journal, publisher, website)
- `type`: Source type (peer-reviewed, book, article, website, video, etc.)

**Optional Metadata** (improves scoring accuracy):
- `citations`: Citation count (Google Scholar, etc.)
- `doi`: DOI or other persistent identifier
- `url`: Web URL
- `abstract`: Abstract or summary text
- `institution`: Author's affiliation
- `credentials`: Author's credentials (PhD, etc.)

---

## Agent Workflow

### Agent: analyst

**Sequential Steps**:

| Step | Objective | Deliverable | Duration | Quality Gate |
|------|-----------|-------------|----------|--------------|
| 0 | Validate inputs | Confirmed metadata completeness | 30 sec | Gate 0: Required fields present |
| 1 | Calculate credibility score | Credibility (1-5) + explanation | 2-5 min | Gate 1: Score justified |
| 2 | Calculate bias score | Bias (1-5) + explanation | 2-5 min | Gate 2: Score justified |
| 3 | Calculate priority score | Priority (1-5) + explanation | 1-3 min | Gate 3: Score justified |
| 4 | Generate structured output | Complete scoring report | 1 min | Gate 4: All scores + explanations |

---

## Step 0: Validate Input Metadata

**Objective**: Ensure required metadata is present and complete

**Procedure**:
1. Check for required fields: `title`, `author`, `year`, `venue`, `type`
2. If missing critical field → Prompt user OR return error
3. If optional fields present → Note for enhanced scoring
4. Validate data types and ranges (year 1500-2025, etc.)

**Deliverable**: Validated metadata object

**Quality Gate 0**:
- **GO**: All required fields present, data valid
- **NO-GO**: Missing required field → Request from user or abort

---

## Step 1: Calculate Credibility Score (1-5)

**Objective**: Assess source trustworthiness using program-of-thought rubric

**Program-of-Thought Rubric**:

**Start at Score 3 (Neutral)**

**Add +1 for EACH** (max +3):
- ✅ Peer-reviewed publication (academic journal, university press)
- ✅ Author has PhD or recognized expertise in field
- ✅ Cites primary sources and provides references
- ✅ Published by reputable institution (top university, established publisher)
- ✅ High citation count (≥50 citations for papers, ≥100 for books)
- ✅ Has DOI or persistent identifier

**Subtract -1 for EACH** (max -3):
- ❌ Self-published or vanity press
- ❌ No author credentials listed
- ❌ No citations or references provided
- ❌ Known conflicts of interest (funded by industry for biased topic)
- ❌ Published on personal blog or unmoderated platform
- ❌ Retracted or corrected after publication

**Final Credibility Score**: [1-5, capped at boundaries]

**Procedure**:
1. Start at 3
2. Apply each applicable rule (+1 or -1)
3. Sum adjustments
4. Cap at 1 (minimum) and 5 (maximum)
5. Generate explanation listing which rules applied

**Example Calculation**:
```
Source: "Machine Learning Fundamentals" by Prof. Jane Smith (MIT Press, 2020)
- Start: 3
- +1 (Published by MIT Press - university press)
- +1 (Author: Professor at Stanford - PhD + expertise)
- +1 (Book cites 200+ references)
- +1 (150 citations on Google Scholar)
= 7 → capped at 5
Final Credibility: 5/5
```

**Deliverable**: Credibility score (1-5) + explanation showing calculation

**Quality Gate 1**:
- **GO**: Score between 1-5, explanation lists applied rules
- **NO-GO**: Score outside range or no explanation → Recalculate

---

## Step 2: Calculate Bias Score (1-5)

**Objective**: Assess source objectivity and potential bias

**Program-of-Thought Rubric**:

**Start at Score 3 (Neutral)**

**Add +1 for EACH** (max +3):
- ✅ Academic/scholarly source with peer review
- ✅ Presents multiple perspectives or counterarguments
- ✅ Clearly distinguishes facts from opinions
- ✅ Transparent about methodology and limitations
- ✅ No financial conflicts of interest
- ✅ Published by neutral/academic institution

**Subtract -1 for EACH** (max -3):
- ❌ Advocacy organization or political think tank
- ❌ Funded by interested party (industry-funded study on own product)
- ❌ One-sided presentation (no counterarguments)
- ❌ Opinion piece or editorial (not research)
- ❌ Sensationalist language or clickbait title
- ❌ Known partisan publication

**Special Cases**:
- **Primary sources** (historical documents, data): Default 4-5 (low bias as factual records)
- **Opinion pieces**: Default 1-2 (high bias by nature)
- **Government reports**: Start at 3, adjust based on independence

**Final Bias Score**: [1-5, where 5 = least biased, 1 = most biased]

**Procedure**:
1. Start at 3
2. Apply each applicable rule (+1 or -1)
3. Sum adjustments
4. Cap at 1 (minimum) and 5 (maximum)
5. Generate explanation listing which rules applied

**Example Calculation**:
```
Source: "Climate Change Impacts" - Think Tank Report (2021)
- Start: 3
- -1 (Published by advocacy organization)
- -1 (Funded by oil industry)
- -1 (One-sided presentation, no counterarguments)
= 0 → capped at 1
Final Bias: 1/5 (highly biased)
```

**Deliverable**: Bias score (1-5) + explanation showing calculation

**Quality Gate 2**:
- **GO**: Score between 1-5, explanation lists applied rules
- **NO-GO**: Score outside range or no explanation → Recalculate

---

## Step 3: Calculate Priority Score (1-5)

**Objective**: Assess reading priority for research project

**Program-of-Thought Rubric**:

**Start at Score 3 (Neutral)**

**Add +1 for EACH** (max +3):
- ✅ **Recent** publication (within 5 years for fast-moving fields, 10 years for historical)
- ✅ **Directly relevant** to research question (addresses core topic)
- ✅ **Highly cited** (indicates influential work)
- ✅ **Primary source** or seminal work (foundational to field)
- ✅ **Recommended** by expert or cited in key papers

**Subtract -1 for EACH** (max -3):
- ❌ **Outdated** (>20 years old for empirical fields, unless historical analysis)
- ❌ **Tangentially relevant** (only mentions topic in passing)
- ❌ **Low credibility** (<3 from Step 1)
- ❌ **High bias** (<3 from Step 2)
- ❌ **Redundant** (covers same ground as already-read higher-priority source)

**Special Considerations**:
- **Classic works**: Even if old, retain high priority if seminal (e.g., Darwin's Origin of Species for evolution research)
- **Breadth vs depth**: Introductory overviews = lower priority than specialized deep dives (unless building foundation)

**Final Priority Score**: [1-5, where 5 = read first, 1 = read last or skip]

**Procedure**:
1. Start at 3
2. Apply each applicable rule (+1 or -1)
3. Sum adjustments
4. Cap at 1 (minimum) and 5 (maximum)
5. Generate explanation listing which rules applied

**Example Calculation**:
```
Source: "Byzantine Trade Routes" by Wilson (1995) - for Renaissance research
- Start: 3
- -1 (Published 30 years ago - somewhat outdated)
- +1 (Directly addresses research question)
- +1 (Cited 250+ times - highly influential)
- +1 (Primary scholarship on topic)
= 5
Final Priority: 5/5 (read first)
```

**Deliverable**: Priority score (1-5) + explanation showing calculation

**Quality Gate 3**:
- **GO**: Score between 1-5, explanation lists applied rules, considers credibility/bias from Steps 1-2
- **NO-GO**: Score outside range or ignores prior scores → Recalculate

---

## Step 4: Generate Structured Output

**Objective**: Create complete scoring report in standardized format

**Output Format** (JSON):

```json
{
  "source": {
    "title": "[Title]",
    "author": "[Author]",
    "year": [YYYY],
    "venue": "[Venue]",
    "type": "[Type]",
    "doi": "[DOI if available]",
    "url": "[URL if available]"
  },
  "scores": {
    "credibility": {
      "score": [1-5],
      "explanation": "[Calculation showing rules applied]"
    },
    "bias": {
      "score": [1-5],
      "explanation": "[Calculation showing rules applied]"
    },
    "priority": {
      "score": [1-5],
      "explanation": "[Calculation showing rules applied]"
    }
  },
  "recommendation": {
    "action": "[READ_FIRST | READ_LATER | SKIP | VERIFY_CLAIMS]",
    "reason": "[1-2 sentences justifying recommendation]"
  },
  "metadata": {
    "analyzed_by": "source-credibility-analyzer",
    "timestamp": "[ISO8601]",
    "version": "1.0"
  }
}
```

**Recommendation Logic**:
- **READ_FIRST**: Priority ≥4 AND Credibility ≥3 AND Bias ≥3
- **READ_LATER**: Priority ≥3 AND (Credibility ≥3 OR Bias ≥3)
- **VERIFY_CLAIMS**: Credibility <3 OR Bias <3 (useful but verify against other sources)
- **SKIP**: Priority <3 AND Credibility <3

**Storage**:
```bash
npx claude-flow@alpha memory store \
  --key "source-analysis/[project]/[source-id]" \
  --value "[JSON output]" \
  --tags "WHO=analyst,WHEN=[timestamp],PROJECT=[topic],WHY=source-scoring,CREDIBILITY=[score],BIAS=[score],PRIORITY=[score]"
```

**Deliverable**: Complete JSON scoring report + Memory MCP storage confirmation

**Quality Gate 4**:
- **GO**: All scores present (1-5), explanations provided, recommendation logic correct, stored in Memory MCP
- **NO-GO**: Missing score/explanation or incorrect recommendation → Review and regenerate

---

## Success Metrics

### Quantitative
- ✅ All 3 scores calculated (credibility, bias, priority)
- ✅ All scores between 1-5 (no invalid values)
- ✅ Explanations provided for each score
- ✅ Recommendation action matches scoring logic
- ✅ Execution time 5-15 minutes per source
- ✅ Output stored in Memory MCP with correct tags

### Qualitative
- ✅ Scores match manual scoring (within ±1 point)
- ✅ Explanations clearly justify scores
- ✅ Recommendation is actionable
- ✅ Tool is reusable across different research projects

---

## Error Handling

| Failure Mode | Gate | Resolution |
|--------------|------|------------|
| **Missing required metadata** | 0 | Prompt user for field OR return error |
| **Score outside 1-5 range** | 1-3 | Recalculate with correct capping |
| **No explanation provided** | 1-3 | Regenerate with explicit rule listing |
| **Recommendation doesn't match scores** | 4 | Reapply recommendation logic |
| **Memory MCP storage fails** | 4 | Retry storage or export JSON locally |
| **Ambiguous source type** | 1 | Default to lower credibility, document uncertainty |

---

## Integration

**Standalone Usage**:
```bash
# Invoke tool directly with source metadata
Skill("source-credibility-analyzer") + metadata
```

**Within general-research-workflow**:
- Called during Step 3 (Source Classification)
- Automates manual scoring process
- Reduces Step 3 time from 30-60 min → 5-15 min per source

**Output Feeds To**:
- general-research-workflow Step 4 (Reading Plan)
- academic-reading-workflow (prioritizes which sources to read deeply)

---

## Example Execution

**Input**:
```json
{
  "title": "Machine Learning: A Probabilistic Perspective",
  "author": "Kevin Murphy",
  "year": 2012,
  "venue": "MIT Press",
  "type": "textbook",
  "citations": 15000,
  "institution": "Google Research"
}
```

**Execution**:
- Step 0: Validate (2 sec) → ✅ All required fields present
- Step 1: Credibility (3 min) → 5/5 (MIT Press + PhD author + 15k citations + comprehensive references)
- Step 2: Bias (2 min) → 5/5 (Academic textbook + neutral presentation + peer-reviewed)
- Step 3: Priority (2 min) → 4/5 (Foundational work + highly cited, -1 for age 13 years)
- Step 4: Generate output (1 min) → JSON + Memory storage

**Output**:
```json
{
  "scores": {
    "credibility": {"score": 5, "explanation": "Start 3, +1 MIT Press, +1 PhD author, +1 15k citations = 6 → capped at 5"},
    "bias": {"score": 5, "explanation": "Start 3, +1 academic textbook, +1 neutral presentation = 5"},
    "priority": {"score": 4, "explanation": "Start 3, +1 foundational work, +1 highly cited, -1 13 years old = 4"}
  },
  "recommendation": {
    "action": "READ_FIRST",
    "reason": "Foundational textbook with highest credibility and zero bias. Essential reading despite age."
  }
}
```

**Total Time**: 8 minutes

---

## Design Notes

**Why This is an SOP, Not a Script**:
- Sequential agent workflow with clear steps
- Quality Gates ensure correctness
- Deliverables and handoffs between steps
- Procedural guidance for analyst agent
- No technical implementation details

**Why Single Agent**:
- Scoring is analytical task, not creative
- Rubrics are deterministic (program-of-thought)
- No need for multiple perspectives (coordination overhead)

**Why Program-of-Thought**:
- Explicit calculations make scores auditable
- Agents can reproduce scoring consistently
- Users can verify logic and adjust rubrics
- Reduces subjective judgment

---

**Next**: Optimize this SOP with prompt-architect framework
