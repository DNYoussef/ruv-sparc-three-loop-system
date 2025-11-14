# General Research Workflow - SOP (Standard Operating Procedure)

**Version**: 2.0
**Type**: Sequential Agent Workflow
**Duration**: 6-10 hours
**Agents Required**: researcher, analyst, coordinator

---

## Purpose

Systematic methodology for general-purpose research across history, mythology, literature, and non-ML domains. Implements Red's (OSP) 6-phase evidence-based research approach with rigorous source evaluation and synthesis.

---

## When to Invoke

**Trigger Conditions**:
- User requests research on historical events, mythology, literary topics
- Need to evaluate primary vs secondary sources
- Building evidence-based arguments from diverse sources
- Topics outside academic ML research scope

**Do NOT Use For**:
- Academic ML research (use `literature-synthesis` instead)
- Quick fact-checking (use search tools)
- Literature review for papers (use `deep-research-orchestrator`)

---

## Agent Sequence

| Step | Agent | Role | Input | Output | Duration |
|------|-------|------|-------|--------|----------|
| 1 | researcher | Wikipedia Mining | Research question | Citation list (10+ refs) | 15-30 min |
| 2 | researcher | Source Discovery | Citation list | Source inventory (20+ sources) | 1-2 hours |
| 3 | analyst | Source Classification | Source inventory | Classified sources with scores | 30-60 min |
| 4 | researcher | Contextual Analysis | Classified sources | Context notes for top sources | 1-2 hours |
| 5 | researcher | Note-Taking | Contextualized sources | Comprehensive notes (50+ notes) | 2-3 hours |
| 6 | coordinator | Synthesis | All notes | Evidence-based thesis + report | 1-2 hours |

---

## Step-by-Step Procedure

### STEP 1: Wikipedia Mining (Gateway Phase)
**Agent**: `researcher`
**Objective**: Extract reference trail from Wikipedia to identify starting sources

**Instructions to Agent**:
1. Search Wikipedia for the research topic
2. Read the full article for overview and context
3. Navigate to "References" section
4. Extract ALL citations with complete metadata:
   - Author name(s)
   - Publication title
   - Year published
   - ISBN/DOI if available
5. Navigate to "Further Reading" section and extract those citations
6. Categorize each citation by type:
   - Books (note if primary or secondary source)
   - Academic papers
   - News articles/journalism
   - Websites/online sources
7. Store citation list with metadata tags

**Deliverable**: JSON array of 10+ citations with complete metadata and categories

**Quality Gate 1**: STOP if <10 citations found. Agent must expand search to related Wikipedia articles.

**Handoff**: Pass citation list to Step 2 (researcher)

---

### STEP 2: Source Discovery (Expansion Phase)
**Agent**: `researcher`
**Objective**: Locate and collect actual sources, expanding beyond Wikipedia's initial references

**Instructions to Agent**:
1. Receive citation list from Step 1
2. For EACH citation:
   - Search library catalog (WorldCat or local library system)
   - Search Google Books for preview/full text availability
   - Determine if primary or secondary source
   - Note accessibility: full text available, preview only, physical copy only, or unavailable
3. Expand search using discovery methods:
   - Google Scholar "Cited by" for influential sources
   - "Customers also bought" recommendations on book sites
   - Author's other publications related to topic
   - Related works suggested by library catalogs
4. For each source, record:
   - Title, author, year, type (primary/secondary)
   - Accessibility status
   - Brief relevance note (1 sentence: why this matters)
   - Assign preliminary relevance score (1-5, where 5 = highly relevant)
5. Build source inventory with all metadata

**Deliverable**: Source inventory with 20+ sources, each with complete metadata and relevance scores

**Quality Gate 2**: STOP if <20 sources collected OR <50% accessible. Agent must continue discovery.

**Handoff**: Pass source inventory to Step 3 (analyst)

---

### STEP 3: Source Classification (Validation Phase)
**Agent**: `analyst`
**Supporting**: `researcher` (for domain context clarification)

**Objective**: Classify sources by type, evaluate credibility, and assign reading priority

**Instructions to Agent**:
1. Receive source inventory from Step 2
2. For EACH source, perform classification:

   **A. Primary vs Secondary Determination**:
   - Primary source: Original documents, eyewitness accounts, artifacts, contemporary records, original research data
   - Secondary source: Analysis/interpretation of primary sources, textbooks, encyclopedias, review articles
   - Mark classification

   **B. Credibility Evaluation** (score 1-5, where 5 = highly credible):
   - Author credentials: Is author an expert in this domain?
   - Publication venue: Peer-reviewed journal? University press? Popular press? Blog?
   - Publication date: Contemporary to events OR retrospective analysis?
   - Citations: Does source cite primary sources and evidence?
   - Assign credibility score

   **C. Bias Detection** (score 1-5, where 5 = high bias risk):
   - Institutional affiliation: Does author have institutional conflicts?
   - Funding sources: Who funded the research/publication?
   - Ideological markers: Identifiable political/ideological leanings?
   - Assign bias risk score

   **D. Reading Priority** (score 1-5, where 5 = must read immediately):
   - Based on: primary vs secondary, credibility, relevance, accessibility
   - Assign priority score

3. Flag sources for specific handling:
   - Priority 4-5: Read immediately in Step 5
   - Priority 1-3: Defer to end of reading queue
   - Conflicting sources: Mark for cross-reference checking
   - High bias sources: Mark for extra scrutiny

**Deliverable**: Classified source inventory with all scores and flags

**Quality Gate 3**: STOP if <5 primary sources identified OR <80% of sources have credibility ≥3. Agent must return to Step 2.

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
   - When was this written/published?
   - What major events were happening at that time?
   - How might the time period influence the author's perspective?

   **B. Cultural Context**:
   - What is the author's cultural/national background?
   - Who was the intended audience?
   - What cultural assumptions might be embedded in the work?

   **C. Historiographical Context**:
   - How does this source fit into scholarly debate on the topic?
   - Is this a mainstream or controversial interpretation?
   - What school of thought does the author represent?

   **D. Translation/Provenance Issues** (if applicable):
   - If translated: What is the translation's reputation?
   - Are there known issues with this translation?
   - Have you compared multiple translations if possible?

4. Create context profile for each source:
   - Author background summary (3-5 sentences)
   - Time period influences
   - Potential biases based on context
   - Reliability caveats or warnings
   - Overall context summary

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
3. For EACH source, create structured notes following this template:

   ```
   SOURCE: [Title] - [Author] ([Year])
   TYPE: [Primary/Secondary] | CREDIBILITY: [Score] | BIAS RISK: [Score]

   KEY CLAIMS:
   - Claim 1 (page X): "[exact quote or paraphrase]"
   - Claim 2 (page Y): "[exact quote or paraphrase]"
   [Continue for all major claims]

   SUPPORTING EVIDENCE:
   - For Claim 1 (pages X-Y): [summary of evidence provided]
   - For Claim 2 (pages Y-Z): [summary of evidence provided]

   CONTRADICTIONS WITH OTHER SOURCES:
   - Conflicts with [Source B] on [specific point] (page X vs page Y in Source B)

   AUTHOR'S BIAS/AGENDA:
   - Observable patterns: [describe any detectable bias]

   QUOTABLE PASSAGES:
   - "[Notable quote 1]" (page X)
   - "[Notable quote 2]" (page Y)

   CROSS-REFERENCES TO OTHER SOURCES:
   - Links to [Source C] on topic of [X]
   - Supports/refutes claims in [Source D]
   ```

4. Tag notes with searchable keywords:
   - `#primary-source` or `#secondary-source`
   - `#key-claim`
   - `#needs-verification` (for claims requiring fact-checking)
   - `#high-confidence` or `#uncertain`
   - `#[topic-keyword]` (e.g., #byzantine-trade, #renaissance-art)

5. Create cross-reference links between related notes across sources

**Deliverable**: 50+ structured notes with page references, 20+ direct quotes with page numbers, 5+ cross-source links

**Quality Gate 5**: STOP if <50 notes OR <20 quotes with pages OR <5 cross-links. Agent must re-read sources for missed content.

**Handoff**: Pass all notes to Step 6 (coordinator)

---

### STEP 6: Synthesis & Dot Connecting (Integration Phase)
**Agent**: `coordinator`
**Supporting**: `researcher` (pattern recognition), `analyst` (validation)

**Objective**: Synthesize insights across all sources into evidence-based thesis and final report

**Instructions to Agent**:

**Phase A - Pattern Recognition** (researcher performs):
1. Review ALL notes from Step 5
2. Identify recurring themes that appear across multiple sources
3. Find areas of agreement (3+ sources making similar claims)
4. Find areas of disagreement or contradiction
5. Map evidence chains: "Source A supports Claim X, which is corroborated by Sources B and C"
6. Look for gaps: claims without sufficient evidence

**Phase B - Thesis Formation** (researcher performs):
1. Let thesis EMERGE from evidence (do NOT impose preconceived thesis)
2. Draft 1-2 sentence thesis statement that:
   - Makes a clear argument or interpretation
   - Is directly supported by evidence from sources
   - Acknowledges the scope/limitations of available evidence
3. List supporting evidence (minimum 5 sources)
4. List counter-evidence or alternative interpretations
5. Identify limitations (e.g., "Only 2 primary sources available", "Sources all Western European perspective")

**Phase C - Validation** (analyst performs):
1. Check for logical fallacies:
   - Circular reasoning
   - Confirmation bias
   - Unsupported leaps
2. Verify EVERY major claim has source citation
3. Flag any unsupported assertions
4. Confirm primary sources are cited for key claims
5. Assess overall argument strength

**Phase D - Final Report Creation** (coordinator performs):
1. Compile findings into structured research report:

   ```
   RESEARCH REPORT: [Topic]

   THESIS:
   [1-2 sentence evidence-based thesis statement]

   SUPPORTING EVIDENCE:
   1. [Claim]
      - Supported by [Source 1, page X], [Source 2, page Y], [Source 3, page Z]
   2. [Claim]
      - Supported by [Source A, page N], [Source B, page M]
   [Continue for all supporting claims]

   LIMITATIONS & CAVEATS:
   - [Limitation 1]: [Explanation]
   - [Limitation 2]: [Explanation]

   COUNTER-EVIDENCE OR ALTERNATIVE INTERPRETATIONS:
   - [Alternative view]: [Source and page]

   PRIMARY SOURCES REFERENCED:
   1. [Source 1] - [Author], [Year] - [Brief description]
   2. [Source 2] - [Author], [Year] - [Brief description]

   SECONDARY SOURCES REFERENCED:
   1. [Source A] - [Author], [Year] - [Brief description]
   2. [Source B] - [Author], [Year] - [Brief description]

   RESEARCH METHODOLOGY NOTES:
   - Total sources consulted: [N]
   - Primary sources: [N]
   - Secondary sources: [N]
   - Time periods covered: [List]
   - Geographic perspectives: [List]
   ```

**Deliverable**: Final research report with thesis supported by 5+ sources, 2+ primary sources cited, NO unsupported claims, 1+ limitation acknowledged

**Quality Gate 6 (FINAL)**:
- **GO Criteria**: Thesis supported by ≥5 sources, ≥2 primary sources, NO unsupported claims, ≥1 limitation stated, analyst validation passed
- **NO-GO Criteria**: Return to Phase B for thesis revision

**Final Output**: Complete research report ready for use

---

## Red's Research Principles (Embedded in SOP)

This SOP implements Red's (OSP) core research methodology:

| Principle | Implementation in SOP |
|-----------|----------------------|
| **"Trust No One"** | Step 3: Credibility evaluation + bias detection for ALL sources |
| **"Context is Everything"** | Step 4: Temporal, cultural, historiographical context for all major sources |
| **"Thesis from Evidence"** | Step 6 Phase B: Thesis must EMERGE from evidence, not be imposed |
| **"Wikipedia is a Gateway"** | Step 1: Mine Wikipedia references as starting point, not endpoint |
| **"Primary Sources Matter"** | Step 3: Classify all sources, require ≥2 primary sources in final report |
| **"Page Numbers Save Lives"** | Step 5: All quotes and claims must include page references |

---

## Success Criteria

### Quantitative Metrics:
- ✅ ≥20 sources in inventory
- ✅ ≥5 primary sources identified
- ✅ ≥80% sources with credibility score ≥3
- ✅ ≥50 notes captured
- ✅ ≥20 quotes with page numbers
- ✅ ≥5 cross-source links
- ✅ Thesis supported by ≥5 sources
- ✅ ≥2 primary sources cited in thesis
- ✅ Complete in 6-10 hours

### Qualitative Metrics:
- ✅ Can explain historical/cultural context of major sources
- ✅ Identified biases in ≥3 sources
- ✅ Thesis emerges from evidence (not imposed)
- ✅ All major claims have source citations
- ✅ Identified ≥1 limitation or caveat
- ✅ Acknowledged alternative interpretations where they exist

---

## Integration Points

### Before This SOP:
- Use `intent-analyzer` if research question is vague or unclear

### During This SOP:
- Can run parallel `literature-synthesis` if topic has ML research component
- Use `source-credibility-analyzer` for Step 3 if available

### After This SOP:
- Use `academic-reading-workflow` for deep reading of key sources
- Use `research-publication` if producing academic paper from findings

---

## Error Handling

**If Quality Gate fails**:
- Gate 1 fails → Expand Wikipedia search to related articles, try different search terms
- Gate 2 fails → Use different discovery methods (Google Scholar, library catalogs, author searches)
- Gate 3 fails → Return to Step 2 to find more/better sources
- Gate 4 fails → Continue contextual research on additional sources
- Gate 5 fails → Re-read priority sources with focus on missed details
- Gate 6 fails → Revise thesis to better match evidence, add more supporting sources

**If source access blocked**:
- Try alternative libraries/catalogs
- Use preview pages if full text unavailable
- Document limitation in final report
- Proceed if ≥50% sources accessible

**If conflicting sources found**:
- Document the conflict explicitly
- Present both interpretations
- Explain which has stronger evidence and why
- Mark as area requiring further research

---

## Agent Coordination Notes

- **researcher** performs most work (Steps 1, 2, 4, 5, Phase A & B of Step 6)
- **analyst** provides validation and quality checks (Steps 3, Phase C of Step 6)
- **coordinator** orchestrates final synthesis (Phase D of Step 6)

All agents must:
- Store deliverables with WHO/WHEN/PROJECT/WHY tags
- Use Memory MCP for cross-session persistence
- Follow 12fa coordination protocols
- Execute hooks for session tracking

---

**END OF SOP**
