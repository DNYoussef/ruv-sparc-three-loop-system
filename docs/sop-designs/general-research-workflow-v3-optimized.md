# General Research Workflow - SOP (Standard Operating Procedure)

**Version**: 3.0 (Optimized with Prompt-Architect)
**Type**: Sequential Agent Workflow
**Duration**: 6-10 hours
**Agents Required**: researcher, analyst, coordinator

---

## Purpose

Systematic methodology for general-purpose research across history, mythology, literature, and non-ML domains. Implements Red's (OSP) 6-phase evidence-based research approach with rigorous source evaluation and synthesis.

---

## When to Invoke - Decision Tree

```
START: User requests research
│
├─ Is topic ML/academic paper research?
│  └─ YES → Use `literature-synthesis` or `deep-research-orchestrator`
│
├─ Is it quick fact-checking (<30 min)?
│  └─ YES → Use web search tools
│
├─ Is it historical/mythological/literary analysis?
│  └─ YES → ✅ USE THIS SOP
│
└─ Does it require primary/secondary source evaluation?
   └─ YES → ✅ USE THIS SOP
```

**Trigger Conditions** (AND logic):
- ✅ Topic outside academic ML research
- ✅ Requires source credibility evaluation
- ✅ Need evidence-based thesis with citations
- ✅ 6+ hours available for thorough research

**Do NOT Use For**:
- ❌ Academic ML research
- ❌ Quick fact-checking
- ❌ Literature review for academic papers

---

## Glossary

**Primary Source**: Original documents, eyewitness accounts, artifacts, contemporary records, or original research data created AT THE TIME of the events being studied.

**Secondary Source**: Analysis, interpretation, or evaluation of primary sources created AFTER the events. Includes textbooks, encyclopedias, review articles, biographies.

**Credibility Score (1-5)**: Reliability rating based on author expertise, publication venue, citations, and methodology.

**Bias Risk Score (1-5)**: Likelihood that source contains systematic distortion due to author's ideology, funding, or institutional affiliations.

**WorldCat**: Global library catalog (worldcat.org) for finding books in nearby libraries.

**Google Scholar**: Search engine for academic publications (scholar.google.com).

**Historiography**: The study of how history has been written and interpreted over time; scholarly debate on historical topics.

**Memory MCP Tagging**: All stored data must include WHO (agent), WHEN (timestamp), PROJECT (research topic), WHY (intent) metadata tags.

---

## Agent Sequence

| Step | Agent | Role | Input | Output | Duration |
|------|-------|------|-------|--------|----------|
| 0 | researcher | Pre-Flight Check | Research question | Verification (Wikipedia exists OR alt plan) | 5-10 min |
| 1 | researcher | Wikipedia Mining | Research question | Citation list (10+ refs) | 15-30 min |
| 2 | researcher | Source Discovery | Citation list | Source inventory (20+ sources) | 1-2 hours |
| 3 | analyst | Source Classification | Source inventory | Classified sources with scores | 30-60 min |
| 4 | researcher | Contextual Analysis | Classified sources | Context notes for top sources | 1-2 hours |
| 5 | researcher | Note-Taking | Contextualized sources | Comprehensive notes (50+ notes) | 2-3 hours |
| 6 | coordinator | Synthesis | All notes | Evidence-based thesis + report | 1-2 hours |

---

## Step-by-Step Procedure

### STEP 0: Pre-Flight Check (NEW - Gate 0)
**Agent**: `researcher`
**Objective**: Verify Wikipedia article exists and plan is viable

**Instructions to Agent**:
1. Search Wikipedia for exact research topic
2. IF Wikipedia article exists:
   - ✅ Proceed to Step 1
3. IF NO Wikipedia article found:
   - Search for related/broader topics
   - Try alternative spellings or terms
   - IF still no article:
     - **FALLBACK PLAN**: Start with Google Scholar search instead
     - Extract ≥10 citations from Google Scholar search results
     - Document limitation: "No Wikipedia article available, started with Google Scholar"
4. Check language accessibility:
   - IF topic requires non-English sources:
     - Flag for translation or assess if English sources sufficient
     - Document language limitation if proceeding without translations

**Deliverable**: Confirmation of viable starting point (Wikipedia OR alternative plan)

**Quality Gate 0**: STOP if no viable starting sources identified. Escalate to user for topic clarification.

**Handoff**: Pass to Step 1 (or adjusted Step 1 if using alternative sources)

---

### STEP 1: Wikipedia Mining (Gateway Phase)
**Agent**: `researcher`
**Objective**: Extract reference trail from Wikipedia to identify starting sources

**Instructions to Agent**:
1. Search Wikipedia for the research topic
2. Read the full article for overview and context
3. Navigate to "References" section
4. Extract ALL citations with complete metadata:
   - ✅ Author name(s) [REQUIRED]
   - ✅ Publication title [REQUIRED]
   - ✅ Year published [REQUIRED]
   - ⚠️ ISBN/DOI if available [OPTIONAL]
5. Navigate to "Further Reading" section and extract those citations
6. Categorize each citation by type:
   - Books (note if likely primary or secondary)
   - Academic papers
   - News articles/journalism
   - Websites/online sources
7. Store citation list with Memory MCP tags: WHO=researcher, WHEN=[timestamp], PROJECT=[research-topic], WHY=source-discovery

**Deliverable**: JSON array of 10+ citations with complete metadata and categories

**Example Output** (See Appendix A for full example):
```json
[
  {
    "title": "The Byzantine Empire",
    "authors": ["Charles Oman"],
    "year": 1892,
    "type": "Book - Secondary Source",
    "doi": null
  },
  ...
]
```

**Quality Gate 1**: STOP if <10 citations found. Agent must expand search to related Wikipedia articles.

**Handoff**: Pass citation list to Step 2 (researcher)

---

### STEP 2: Source Discovery (Expansion Phase)
**Agent**: `researcher`
**Objective**: Locate and collect actual sources, expanding beyond Wikipedia's initial references

**Instructions to Agent**:
1. Receive citation list from Step 1
2. For EACH citation:
   - Search library catalog (WorldCat: worldcat.org)
   - Search Google Books for preview/full text
   - Determine if primary or secondary source
   - Note accessibility: full text, preview only, physical copy, or unavailable
   - Note language: English, requires translation, or multilingual
3. Expand search using discovery methods:
   - Google Scholar "Cited by" for influential sources
   - "Related articles" suggestions
   - Author's other publications on topic
   - Library catalog "Related works"
4. For each source, record:
   - ✅ Title, author, year, type [REQUIRED]
   - ✅ Accessibility status [REQUIRED]
   - ✅ Language [REQUIRED]
   - ✅ Relevance note: "How does this address [research question]? Provides [primary evidence/expert analysis/contextual background]" [REQUIRED]
   - ✅ Preliminary relevance score (1-5) [REQUIRED]
5. Build source inventory with all metadata

**Deliverable**: Source inventory with 20+ sources, each with complete metadata and relevance scores

**Quality Gate 2**: STOP if <20 sources collected OR <50% accessible. Agent must continue discovery.

**Exception**: If topic has NO accessible sources (ancient/obscure topic), proceed with ≥10 highest-credibility sources and document limitation.

**Handoff**: Pass source inventory to Step 3 (analyst)

---

### STEP 3: Source Classification (Validation Phase)
**Agent**: `analyst`
**Supporting**: `researcher` (for domain context clarification)

**Objective**: Classify sources by type, evaluate credibility, and assign reading priority using systematic rubric

**Instructions to Agent**:
1. Receive source inventory from Step 2
2. For EACH source, perform classification using scoring rubrics below:

---

#### A. Primary vs Secondary Classification

✅ **Primary Source** if:
- Original documents from the time period (letters, diaries, official records)
- Eyewitness accounts of events
- Artifacts or physical evidence
- Contemporary records (newspapers from that era)
- Original research data

✅ **Secondary Source** if:
- Analysis of primary sources
- Textbooks or encyclopedias
- Biographies written after subject's death
- Review articles or meta-analyses
- Historical interpretations

**Mark**: PRIMARY or SECONDARY

---

#### B. Credibility Score (1-5) - Program-of-Thought Rubric

**Start at Score 3 (Neutral)**

**Add +1 for EACH**:
- ✅ Peer-reviewed publication (academic journal, university press)
- ✅ Author has PhD or recognized expertise in field
- ✅ Cites primary sources and provides references
- ✅ Published by reputable institution

**Subtract -1 for EACH**:
- ❌ Self-published or vanity press
- ❌ No author credentials listed
- ❌ No citations or references provided
- ❌ Known conflicts of interest

**Final Credibility Score**: [1-5, capped at boundaries]

**Example**:
```
Source: "Byzantine Trade Routes" by Dr. John Smith (2010), Cambridge University Press
Start: 3
+1 (university press)
+1 (author has PhD)
+1 (extensive citations)
= 6 → capped at 5
Final Credibility: 5/5
```

---

#### C. Bias Risk Score (1-5) - Program-of-Thought Rubric

**Start at Score 2 (Low Bias)**

**Add +1 for EACH**:
- ⚠️ Author affiliated with advocacy organization
- ⚠️ Funding from interested party (e.g., government, corporation)
- ⚠️ Strong ideological language detected
- ⚠️ Cherry-picked evidence or one-sided presentation

**Final Bias Risk Score**: [1-5]

**1 = Minimal bias** (objective, balanced)
**3 = Moderate bias** (identifiable lean but useful)
**5 = High bias** (propaganda, extreme partisanship)

---

#### D. Reading Priority (1-5) - Calculated Score

**Formula**:
```
Priority = (Relevance × 0.4) + (Credibility × 0.3) + (Primary = +2, Secondary = 0) + (Accessible = +1, Not Accessible = -1)

Normalized to 1-5 scale
```

**Priority Bands**:
- **5**: Read IMMEDIATELY (high-quality primary sources directly addressing question)
- **4**: Read soon (credible secondary sources or accessible primaries)
- **3**: Read if time permits (moderate quality or tangential)
- **2**: Defer to end (low relevance or credibility)
- **1**: Skip unless critical gap (inaccessible or very low quality)

---

3. Flag sources for specific handling:
   - 💡 Priority 4-5: Add to immediate reading queue (Step 5)
   - ⏸️ Priority 1-3: Defer to end of reading queue
   - ⚠️ Conflicting sources: Mark for cross-reference checking
   - 🚨 High bias (≥4): Mark for extra scrutiny in notes

**Deliverable**: Classified source inventory with all scores and flags (see Appendix B for example)

**Quality Gate 3**: STOP if <5 primary sources identified OR <80% of sources have credibility ≥3.

**Exception**: If topic has NO primary sources (e.g., ancient mythology with only later recordings), proceed with ≥10 high-credibility (≥4) secondary sources. Document this limitation.

**Handoff**: Pass classified inventory to Step 4 (researcher)

---

### STEP 4: Contextual Analysis (Deep Dive Phase)
**Agent**: `researcher`
**Supporting**: `analyst` (for verification of context claims)

**Objective**: Understand each source within its historical, cultural, and scholarly context

**Instructions to Agent**:
1. Receive classified sources from Step 3
2. Start with priority 4-5 sources first
3. For EACH high-priority source, research and document:

   **A. Temporal Context**:
   - ✅ When was this written/published? [REQUIRED]
   - ✅ What major events were happening at that time? [REQUIRED]
   - 💡 How might the time period influence the author's perspective? [ANALYSIS]

   **B. Cultural Context**:
   - ✅ What is the author's cultural/national background? [REQUIRED]
   - ✅ Who was the intended audience? [REQUIRED]
   - 💡 What cultural assumptions might be embedded? [ANALYSIS]

   **C. Historiographical Context**:
   - ✅ How does this fit into scholarly debate? [REQUIRED]
   - ⚠️ Mainstream or controversial interpretation? [IF APPLICABLE]
   - ⚠️ What school of thought does author represent? [IF APPLICABLE]

   **D. Translation/Provenance Issues** (if applicable):
   - ⚠️ If translated: Translation's reputation?
   - ⚠️ Known issues with this translation?
   - ⚠️ Multiple translations available for comparison?

4. Create context profile for each source (3-5 sentences per context type)

**Deliverable**: Context profiles for 10+ sources covering 3+ different temporal periods

**Quality Gate 4**: STOP if <10 sources contextualized OR <3 temporal periods covered. Agent must continue analysis.

**Handoff**: Pass contextualized sources to Step 5 (researcher)

---

### STEP 5: Comprehensive Note-Taking (Capture Phase)
**Agent**: `researcher`
**Objective**: Extract key claims, evidence, quotes, and page numbers from all sources

**Instructions to Agent**:
1. Receive contextualized sources from Step 4
2. Read sources in priority order (4-5 first, then 1-3)
3. For EACH source, create structured notes using template:

---

**NOTE TEMPLATE** (Simplified - Required vs Optional):

```markdown
## SOURCE: [Title] - [Author] ([Year])
TYPE: [Primary/Secondary] | CREDIBILITY: [Score] | BIAS RISK: [Score]

### ✅ KEY CLAIMS [REQUIRED - Minimum 2]
- Claim 1 (page X): "[exact quote or detailed paraphrase]"
- Claim 2 (page Y): "[exact quote or detailed paraphrase]"

### ✅ SUPPORTING EVIDENCE [REQUIRED - For each claim]
- For Claim 1 (pages X-Y): [How does author support this? Data, other sources, reasoning?]
- For Claim 2 (pages Y-Z): [How does author support this?]

### ✅ QUOTABLE PASSAGES [REQUIRED - Minimum 2 with page numbers]
- "Notable quote 1 that captures key argument" (page X)
- "Notable quote 2 with strong evidence" (page Y)

### ⚠️ CONTRADICTIONS WITH OTHER SOURCES [OPTIONAL - Only if detected]
- Conflicts with [Source B] on [specific point] (page X vs page Y in Source B)
- Explain nature of contradiction

### ⚠️ AUTHOR'S BIAS/AGENDA [OPTIONAL - Only if Step 3 bias score ≥3]
- Observable patterns: [Describe detectable bias, provide examples]

### ⚠️ CROSS-REFERENCES TO OTHER SOURCES [OPTIONAL - If relevant]
- Links to [Source C] on topic of [X]
- Supports/refutes claims in [Source D]
```

---

4. Tag notes with searchable keywords:
   - `#primary-source` or `#secondary-source`
   - `#key-claim`
   - `#needs-verification` (for claims requiring fact-checking)
   - `#high-confidence` or `#uncertain`
   - `#[topic-keywords]` (e.g., #byzantine-trade, #renaissance-art)

5. Create cross-reference links between related notes

**Deliverable**: 50+ structured notes, 20+ quotes with page numbers, 5+ cross-source links

**Example Note** (See Appendix C for full example)

**Quality Gate 5**: STOP if <50 notes OR <20 quotes with pages OR <5 cross-links. Agent must re-read sources.

**Handoff**: Pass all notes to Step 6 (coordinator)

---

### STEP 6: Synthesis & Dot Connecting (Integration Phase)
**Agent**: `coordinator`
**Supporting**: `researcher` (pattern recognition), `analyst` (validation)

**Objective**: Synthesize insights across sources into evidence-based thesis and final report

**Instructions to Agent**:

**Phase A - Pattern Recognition** (researcher performs):
1. Review ALL notes from Step 5
2. Identify recurring themes (appears in ≥3 sources)
3. Find areas of agreement (multiple sources making similar claims)
4. Find areas of disagreement or contradiction
5. Map evidence chains: "Source A claims X (p. 42), supported by Source B (p. 15) and Source C (p. 88)"
6. Identify gaps: claims without sufficient evidence

**Phase B - Thesis Formation** (researcher performs):
1. **CRITICAL**: Let thesis EMERGE from evidence (do NOT impose preconceived thesis)
2. Draft 1-2 sentence thesis that:
   - ✅ Makes clear argument or interpretation
   - ✅ Is directly supported by evidence from sources
   - ✅ Acknowledges scope/limitations
3. List supporting evidence (minimum 5 sources)
4. List counter-evidence or alternative interpretations
5. Identify limitations:
   - "Only 2 primary sources available"
   - "Sources all from Western European perspective"
   - "No sources from [time period/region]"

**SPECIAL CASE - Inconclusive Evidence**:
- IF evidence is too contradictory or insufficient:
  - ✅ State "INCONCLUSIVE" instead of forcing thesis
  - 💡 Explain WHY: "Sources present conflicting accounts of [X] without sufficient primary evidence to resolve"
  - 📋 Summarize competing interpretations
  - 🚨 Document what additional sources would resolve ambiguity

**Phase C - Validation** (analyst performs):
1. Check for logical fallacies:
   - ❌ Circular reasoning
   - ❌ Confirmation bias (cherry-picking)
   - ❌ Unsupported leaps in logic
2. Verify EVERY major claim has source citation WITH PAGE NUMBER
3. Flag any unsupported assertions → return to researcher for revision
4. Confirm ≥2 primary sources cited for key claims (if primaries available)
5. Assess argument strength:
   - ✅ STRONG: ≥5 sources, ≥2 primaries, logical consistency
   - ⚠️ MODERATE: 3-4 sources OR 1 primary, minor gaps
   - ❌ WEAK: <3 sources OR no primaries, significant gaps → return for revision

**Phase D - Final Report Creation** (coordinator performs):
1. Compile findings into structured report (see template below and Appendix D for full example):

---

**FINAL REPORT TEMPLATE**:

```markdown
# RESEARCH REPORT: [Topic]

## THESIS
[1-2 sentence evidence-based thesis statement]
OR
[INCONCLUSIVE: Explanation of why evidence insufficient]

## SUPPORTING EVIDENCE
1. [Claim 1]
   - **Source 1**: [Author, Year, pages X-Y] - [Quote or paraphrase]
   - **Source 2**: [Author, Year, pages Z-W] - [Quote or paraphrase]
   - **Source 3**: [Author, Year, page Q] - [Quote or paraphrase]

2. [Claim 2]
   - **Source A**: [Author, Year, page M] - [Quote or paraphrase]
   - **Source B**: [Author, Year, pages N-P] - [Quote or paraphrase]

[Continue for all supporting claims]

## LIMITATIONS & CAVEATS
- **Limitation 1**: [Description - e.g., "Only 2 primary sources accessible"]
- **Limitation 2**: [Description - e.g., "All sources from Western perspective"]
- **Limitation 3**: [Description - e.g., "Topic spans 200 years but sources concentrated in 1400-1450"]

## COUNTER-EVIDENCE OR ALTERNATIVE INTERPRETATIONS
- **Alternative View 1**: [Description]
  - **Source**: [Author, Year, page X] - [Brief explanation]
- **Alternative View 2**: [Description]
  - **Source**: [Author, Year, page Y] - [Brief explanation]

## PRIMARY SOURCES REFERENCED [N total]
1. [Source Title] - [Author], [Year] - [1-sentence description of significance]
2. [Source Title] - [Author], [Year] - [1-sentence description]
[Continue for all primary sources]

## SECONDARY SOURCES REFERENCED [N total]
1. [Source Title] - [Author], [Year] - [1-sentence description of contribution]
2. [Source Title] - [Author], [Year] - [1-sentence description]
[Continue for all secondary sources]

## RESEARCH METHODOLOGY NOTES
- **Total sources consulted**: [N]
- **Primary sources**: [N]
- **Secondary sources**: [N]
- **Time periods covered**: [List time ranges]
- **Geographic/cultural perspectives**: [List regions/cultures represented]
- **Languages**: [English + other languages if applicable]
- **Research duration**: [X hours]
- **Quality gates passed**: [0-6]

## RECOMMENDATIONS FOR FURTHER RESEARCH
- [What additional sources would strengthen this analysis?]
- [What questions remain unanswered?]
- [What archives or collections might have relevant materials?]
```

---

**Deliverable**: Final research report with all sections complete

**Quality Gate 6 (FINAL)**:
- **GO Criteria**:
  - ✅ Thesis supported by ≥5 sources OR marked "INCONCLUSIVE" with explanation
  - ✅ ≥2 primary sources cited (OR exception documented if none available)
  - ✅ NO unsupported claims
  - ✅ ≥1 limitation acknowledged
  - ✅ Analyst validation passed (no logical fallacies)

- **NO-GO Criteria** (return to Phase B for revision):
  - ❌ Unsupported claims exist
  - ❌ Logical fallacies detected
  - ❌ Insufficient evidence (<5 sources for thesis)
  - ❌ No limitations acknowledged (overconfident)

**Final Output**: Complete research report ready for use

---

## Red's Research Principles (Embedded in SOP)

| Principle | Implementation in SOP |
|-----------|----------------------|
| **"Trust No One"** | Step 3: Systematic credibility + bias scoring with explicit rubrics |
| **"Context is Everything"** | Step 4: Temporal, cultural, historiographical context required for all major sources |
| **"Thesis from Evidence"** | Step 6 Phase B: Thesis must EMERGE from evidence, not be imposed. "INCONCLUSIVE" option if evidence insufficient |
| **"Wikipedia is a Gateway"** | Step 1: Mine Wikipedia references as starting point, not final authority. Step 0 provides fallback if no Wikipedia article |
| **"Primary Sources Matter"** | Step 3: Classify all sources, require ≥2 primary sources in final report (or document exception) |
| **"Page Numbers Save Lives"** | Step 5: All quotes and claims MUST include page references for verification |

---

## Success Criteria

### Quantitative Metrics:
- ✅ ≥20 sources in inventory
- ✅ ≥5 primary sources (OR exception documented)
- ✅ ≥80% sources with credibility ≥3
- ✅ ≥50 notes captured
- ✅ ≥20 quotes with page numbers
- ✅ ≥5 cross-source links
- ✅ Thesis supported by ≥5 sources (OR "INCONCLUSIVE" documented)
- ✅ ≥2 primary sources cited (OR exception documented)
- ✅ Complete in 6-10 hours

### Qualitative Metrics:
- ✅ Can explain historical/cultural context of ≥10 sources
- ✅ Identified biases in ≥3 sources
- ✅ Thesis emerges from evidence (not imposed)
- ✅ All major claims have source citations + page numbers
- ✅ Identified ≥1 limitation or caveat
- ✅ Acknowledged alternative interpretations where they exist
- ✅ NO logical fallacies in final report

---

## Integration Points

### Before This SOP:
- Use `intent-analyzer` if research question vague or unclear

### During This SOP:
- Can run parallel `literature-synthesis` if topic has ML research component
- Use `source-credibility-analyzer` for Step 3 if available (automates scoring)

### After This SOP:
- Use `academic-reading-workflow` for deep reading of specific sources
- Use `research-publication` if producing academic paper from findings

---

## Error Handling & Failure Modes

| Failure Mode | Detection Point | Resolution |
|--------------|----------------|------------|
| **No Wikipedia article** | Gate 0 | Use Google Scholar fallback, start with ≥10 academic sources |
| **<10 citations** | Gate 1 | Expand to related Wikipedia articles, try alternative search terms |
| **<20 sources** | Gate 2 | Use different discovery methods, broaden search scope |
| **<50% accessible** | Gate 2 | Prioritize accessible sources, document limitation if unavoidable |
| **<5 primary sources** | Gate 3 | Continue discovery OR document "No primary sources available" exception |
| **<80% credibility ≥3** | Gate 3 | Return to Step 2 for higher-quality sources |
| **Non-English sources** | Step 0, Step 2 | Flag for translation OR proceed with English sources, document language limitation |
| **Contradictory evidence** | Step 6 Phase B | Use "INCONCLUSIVE" option, present competing interpretations |
| **Logical fallacies** | Step 6 Phase C | Return to Phase B for thesis revision |
| **Unsupported claims** | Step 6 Phase C | Add supporting sources OR remove claims |

---

## Agent Coordination Notes

- **researcher** performs: Steps 0, 1, 2, 4, 5, Phase A & B of Step 6
- **analyst** performs: Step 3, Phase C of Step 6 (validation)
- **coordinator** orchestrates: Phase D of Step 6 (final report compilation)

**All agents must**:
- Store deliverables with Memory MCP tags: WHO/WHEN/PROJECT/WHY
- Use cross-session persistence for multi-day research projects
- Follow 12fa agent coordination protocols
- Execute hooks for session tracking and metrics

---

## APPENDICES - Few-Shot Examples

### Appendix A: Sample Wikipedia Citation Extraction (Step 1 Output)

**Research Topic**: "Byzantine Empire's influence on Renaissance Italy"

**Step 1 Output**:
```json
[
  {
    "title": "The Byzantine Empire",
    "authors": ["Charles Oman"],
    "year": 1892,
    "type": "Book - Secondary Source",
    "isbn": null,
    "doi": null,
    "category": "General History"
  },
  {
    "title": "Byzantium and the Renaissance: Greek Scholars in Venice",
    "authors": ["N.G. Wilson"],
    "year": 1992,
    "type": "Book - Secondary Source",
    "isbn": "978-0521334518",
    "doi": null,
    "category": "Specialized Study"
  },
  {
    "title": "The Fall of Constantinople 1453",
    "authors": ["Steven Runciman"],
    "year": 1965,
    "type": "Book - Secondary Source (uses primary sources)",
    "isbn": null,
    "doi": null,
    "category": "Event Study"
  },
  {
    "title": "Plethon and the Notion of 'Paganism' in the Byzantine World",
    "authors": ["Niketas Siniossoglou"],
    "year": 2011,
    "type": "Academic Article - Secondary Source",
    "isbn": null,
    "doi": "10.1111/j.1748-0922.2011.00123.x",
    "category": "Scholarly Journal"
  }
]
```

**Total Citations**: 12 (showing 4 for brevity)
**Categories**: 3 (Books, Articles, Primary Documents)
**Quality Gate 1**: ✅ PASS (≥10 citations, ≥3 categories)

---

### Appendix B: Sample Source Classification (Step 3 Output)

**Source Classification Example**:

| Title | Author | Year | Type | Credibility | Bias | Priority | Flags |
|-------|--------|------|------|-------------|------|----------|-------|
| Letters of Cardinal Bessarion | Bessarion | 1460 | PRIMARY | 5 | 2 | 5 | 💡 Read immediately |
| Byzantium and Renaissance | N.G. Wilson | 1992 | SECONDARY | 5 | 1 | 5 | 💡 Read immediately |
| Byzantine Legacy | G. Huxley | 1975 | SECONDARY | 4 | 2 | 4 | 💡 Read soon |
| Venice and Byzantium | D. Howard | 2000 | SECONDARY | 4 | 2 | 3 | ⏸️ Defer |

**Credibility Calculation Example** (N.G. Wilson source):
```
Start: 3
+1 (Cambridge University Press - peer-reviewed)
+1 (Author: Professor Emeritus, Oxford - PhD + expertise)
+1 (Extensive primary source citations)
= 6 → capped at 5
Final: 5/5
```

**Priority Calculation Example** (N.G. Wilson source):
```
Relevance: 5 (directly addresses topic)
Credibility: 5
Type: Secondary (0 points, not +2)
Accessible: Full text (+1)

Priority = (5 × 0.4) + (5 × 0.3) + 0 + 1
         = 2.0 + 1.5 + 0 + 1
         = 4.5 → rounded to 5
Final: 5/5
```

---

### Appendix C: Sample Note (Step 5 Output)

```markdown
## SOURCE: Byzantium and the Renaissance - N.G. Wilson (1992)
TYPE: Secondary Source | CREDIBILITY: 5/5 | BIAS RISK: 1/5

### ✅ KEY CLAIMS
- Claim 1 (page 15): "The Greek scholars fleeing Constantinople after 1453 brought not just manuscripts but pedagogical methods that transformed Italian humanism"
- Claim 2 (page 42): "Cardinal Bessarion's donation of his manuscript collection to Venice (1468) created the foundation for the Marciana Library and preserved hundreds of classical texts"

### ✅ SUPPORTING EVIDENCE
- For Claim 1 (pages 15-22): Wilson documents employment records of Greek teachers in Venice, Florence, Rome 1440-1500. Shows curriculum changes from Latin classics to Greek originals. Cites letters from Italian humanists praising Greek grammatical instruction.
- For Claim 2 (pages 42-48): Inventory of Bessarion's 746 manuscripts, of which 482 survive in Venice. Catalogs rare texts otherwise lost. Traces provenance from Constantinople to Venice.

### ✅ QUOTABLE PASSAGES
- "Without the Byzantine émigrés, the Italian Renaissance would have remained a Latin phenomenon, cut off from the Greek roots of European civilization" (page 18)
- "Bessarion's library was not merely a collection but a deliberate act of cultural preservation in the face of Ottoman conquest" (page 44)

### ⚠️ CONTRADICTIONS WITH OTHER SOURCES
- Conflicts with Howard (2000) on timing: Wilson says significant Greek migration 1440s, Howard claims 1453 fall of Constantinople was trigger. (page 15 vs Howard page 88)

### ⚠️ CROSS-REFERENCES TO OTHER SOURCES
- Supports Runciman's (1965) account of Constantinople's fall creating diaspora
- Provides evidence for Siniossoglou's (2011) analysis of Plethon's influence
- Links to Bessarion's own letters (primary source) for manuscript donation details
```

---

### Appendix D: Sample Final Report (Step 6 Output - Abbreviated)

```markdown
# RESEARCH REPORT: Byzantine Empire's Influence on Renaissance Italy

## THESIS
The migration of Greek scholars from Constantinople to Italy (1440-1500), accelerated by the 1453 Ottoman conquest, fundamentally transformed the Italian Renaissance from a Latin-focused recovery of classical antiquity into a comprehensive revival encompassing both Greek and Latin intellectual traditions, thereby reconnecting Western Europe with its Hellenic philosophical and literary roots.

## SUPPORTING EVIDENCE
1. **Greek Pedagogical Methods Transformed Italian Humanism**
   - **Wilson, 1992, pages 15-22**: Documents employment of Greek teachers in Venice, Florence, Rome. Shows curriculum shift from Latin classics to Greek originals. Cites letters from Italian humanists praising Greek grammatical instruction.
   - **Bessarion Letters, 1460, folio 12r**: Cardinal writes, "The Greeks bring us not mere books but the keys to understanding Plato in his mother tongue."
   - **Siniossoglou, 2011, pages 88-92**: Analyzes how Gemistos Plethon's Platonism influenced Ficino's Florentine Academy.

2. **Manuscript Migration Preserved Classical Texts**
   - **Wilson, 1992, pages 42-48**: Bessarion's 746 manuscripts donated to Venice (1468), 482 survive. Catalogs rare texts otherwise lost.
   - **Runciman, 1965, page 193**: Fall of Constantinople (1453) prompted mass exodus of scholars carrying manuscripts.

[Continue for all claims...]

## LIMITATIONS & CAVEATS
- **Limitation 1**: Only 2 primary sources directly accessible (Bessarion's letters and one manuscript catalog). Most evidence from secondary historical analyses.
- **Limitation 2**: Sources predominantly Western European perspective. Limited access to Ottoman records from same period that might provide alternative view of cultural exchange.
- **Limitation 3**: Topic spans 60 years (1440-1500) but sources concentrate on 1450-1470 period. Earlier and later periods less documented.

## COUNTER-EVIDENCE OR ALTERNATIVE INTERPRETATIONS
- **Alternative View**: Some scholars (Howard, 2000) argue Greek influence was minor compared to Arabic transmission of classical knowledge via Spain and Sicily.
  - **Howard, 2000, page 103**: "The Greeks arrived late to a party the Arabs had been hosting for centuries."
  - **Counter**: Wilson demonstrates Greek émigrés brought original texts and linguistic expertise unavailable via Arabic translations.

## PRIMARY SOURCES REFERENCED [2 total]
1. **Letters of Cardinal Bessarion** - Bessarion, 1460 - Contemporary account of manuscript collection and Greek scholarly network in Italy
2. **Manuscript Catalog of San Marco Library** - Anonymous, 1468 - Inventory of Bessarion's donated collection

## SECONDARY SOURCES REFERENCED [8 total]
1. **Byzantium and the Renaissance** - N.G. Wilson, 1992 - Comprehensive study of Greek scholars in Italy with archival evidence
2. **The Fall of Constantinople 1453** - Steven Runciman, 1965 - Authoritative account of Constantinople's fall and diaspora
[Continue...]

## RESEARCH METHODOLOGY NOTES
- **Total sources consulted**: 23
- **Primary sources**: 2
- **Secondary sources**: 21
- **Time periods covered**: 1440-1500 (core), 1204-1600 (context)
- **Geographic perspectives**: Italian (Venice, Florence, Rome), Greek (Constantinople), broader European
- **Languages**: English (all sources), with English translations of Greek/Latin primary sources
- **Research duration**: 8 hours
- **Quality gates passed**: 7/7 (including Gate 0)

## RECOMMENDATIONS FOR FURTHER RESEARCH
- Access Vatican Archives for additional Bessarion correspondence
- Examine Ottoman records from 1453-1500 for alternative perspective on cultural exchange
- Compare with Arabic-Latin transmission in Spain for fuller picture of classical revival
- Investigate later period (1500-1600) for longer-term impact assessment
```

---

**END OF SOP - Version 3.0 Optimized**
