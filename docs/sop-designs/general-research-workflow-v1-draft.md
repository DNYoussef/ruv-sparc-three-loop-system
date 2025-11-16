# General Research Workflow - SOP Design (v1 Draft)

## Overview
Implements Red's (OSP) 6-phase research methodology for general-purpose investigation across history, mythology, literature, and non-ML domains. Uses systematic source discovery, credibility evaluation, and evidence-based synthesis.

## When to Use This Skill
- Researching historical events, mythological topics, literary analysis
- Investigating topics outside ML/academic research scope
- Need to evaluate primary vs secondary sources
- Building evidence-based arguments from diverse sources

## Prerequisites
- Research topic/question clearly defined
- Access to Wikipedia, Google Books, library catalogs
- Memory MCP for storing findings
- 2-8 hours for complete workflow

## Agent Coordination

| Phase | Primary Agent | Supporting Agents | Duration |
|-------|--------------|-------------------|----------|
| 1. Wikipedia Mining | researcher | - | 15-30 min |
| 2. Source Discovery | researcher | - | 1-2 hours |
| 3. Source Classification | analyst | researcher | 30-60 min |
| 4. Contextual Analysis | researcher | analyst | 1-2 hours |
| 5. Note-Taking | researcher | - | 2-3 hours |
| 6. Synthesis | coordinator | researcher, analyst | 1-2 hours |

**Total Duration**: 6-10 hours

## Workflow

### Phase 1: Wikipedia Mining (Gateway Phase)
**Agent**: researcher
**Goal**: Extract reference trail from Wikipedia as research starting point

**Actions**:
1. Search Wikipedia for research topic
2. Read article for overview context
3. Navigate to "References" and "Further Reading" sections
4. Extract ALL citations with:
   - Author names
   - Publication titles
   - Years
   - ISBN/DOI when available
5. Categorize citations:
   - Books (primary sources, secondary sources)
   - Academic papers
   - News articles
   - Websites
6. Store in Memory MCP with tags: `project:research-topic`, `phase:wikipedia-mining`, `intent:source-discovery`

**Quality Check**: ≥10 unique citations extracted, ≥3 categories represented

**Commands**:
```bash
# Store Wikipedia citations
npx claude-flow@alpha memory store \
  --key "research/[topic]/wikipedia-refs" \
  --value "[JSON array of citations]" \
  --tags "phase:wikipedia-mining,source:wikipedia"
```

---

### Phase 2: Source Discovery (Expansion Phase)
**Agent**: researcher
**Goal**: Find and collect actual sources from library catalogs, Google Books, databases

**Actions**:
1. Retrieve Wikipedia citations from Memory MCP
2. For EACH citation:
   - Search library catalog (WorldCat/local library)
   - Search Google Books for preview/full text
   - Check if primary or secondary source
   - Note accessibility (full text, preview only, physical copy needed)
3. Expand search with related works:
   - "Customers who bought this also bought..."
   - "Cited by" on Google Scholar
   - Author's other publications
4. Create source inventory with metadata:
   - Title, author, year, type (primary/secondary)
   - Accessibility status
   - Relevance score (1-5)
5. Store in Memory MCP

**Quality Check**: ≥20 sources collected, ≥50% accessible (full text or preview)

**Commands**:
```bash
# Store source inventory
npx claude-flow@alpha memory store \
  --key "research/[topic]/source-inventory" \
  --value "[JSON array of sources with metadata]" \
  --tags "phase:source-discovery,status:accessible"
```

---

### Phase 3: Source Classification (Validation Phase)
**Agent**: analyst
**Supporting**: researcher (for context)

**Goal**: Classify sources as primary vs secondary, evaluate credibility

**Actions**:
1. Retrieve source inventory from Memory MCP
2. For EACH source:
   - **Primary vs Secondary**:
     - Primary: Original documents, eyewitness accounts, artifacts, original research
     - Secondary: Analysis of primary sources, textbooks, encyclopedias
   - **Credibility Evaluation**:
     - Author credentials (expert in field? biases?)
     - Publication venue (peer-reviewed? popular press? blog?)
     - Publication date (contemporary? retrospective?)
     - Citations/references (does it cite primary sources?)
   - **Bias Detection**:
     - Author's institutional affiliation
     - Funding sources
     - Ideological leanings (if identifiable)
3. Assign scores:
   - Credibility: 1-5 (5 = highly credible)
   - Bias risk: 1-5 (5 = high bias risk)
   - Priority: 1-5 (5 = must read)
4. Flag sources for:
   - Immediate reading (priority 4-5)
   - Deferred reading (priority 1-3)
   - Cross-reference checking (conflicting claims)

**Quality Check**: ALL sources classified, ≥5 primary sources identified, ≥80% credibility ≥3

**Commands**:
```bash
# Store classified sources
npx claude-flow@alpha memory store \
  --key "research/[topic]/classified-sources" \
  --value "[JSON with credibility scores]" \
  --tags "phase:classification,validated:true"
```

---

### Phase 4: Contextual Analysis (Deep Dive Phase)
**Agent**: researcher
**Supporting**: analyst (for verification)

**Goal**: Understand sources within historical, cultural, temporal context

**Actions**:
1. Retrieve classified sources (priority 4-5 first)
2. For EACH high-priority source:
   - **Temporal Context**: When was this written? What was happening at that time?
   - **Cultural Context**: Author's cultural background? Intended audience?
   - **Historiographical Context**: How does this fit into scholarly debate?
   - **Translation Issues**: If translated, what's the provenance? Multiple translations compared?
3. Create context notes:
   - Author background (3-5 sentences)
   - Time period influences
   - Potential biases based on context
   - Reliability caveats
4. Store context alongside source metadata

**Quality Check**: ≥10 sources contextualized, ≥3 temporal periods represented

**Commands**:
```bash
# Store contextual analysis
npx claude-flow@alpha memory store \
  --key "research/[topic]/context-analysis" \
  --value "[JSON with context notes per source]" \
  --tags "phase:contextualization,depth:deep"
```

---

### Phase 5: Comprehensive Note-Taking (Capture Phase)
**Agent**: researcher
**Goal**: Extract quotes, ideas, and page numbers from sources

**Actions**:
1. Retrieve contextualized sources
2. For EACH source (read in priority order):
   - Read thoroughly with critical lens
   - Capture notes with structure:
     ```markdown
     ## [Source Title] - [Author] ([Year])

     ### Key Claims
     - Claim 1 (p. 42): "[exact quote]"
     - Claim 2 (p. 87): [paraphrase with page ref]

     ### Supporting Evidence
     - Evidence for Claim 1 (p. 45-48): ...

     ### Contradictions/Tensions
     - Conflicts with [Other Source] on [point] (p. 92 vs p. 15)

     ### Author's Bias/Agenda
     - Observable leanings: ...

     ### Quotable Passages
     - "Quote 1" (p. 123)
     - "Quote 2" (p. 156)
     ```
   - Use inline tags: #primary-source, #key-claim, #needs-verification
   - Link related notes across sources
3. Store notes in Memory MCP with searchable keywords

**Quality Check**: ≥50 notes captured, ≥20 direct quotes with page numbers, ≥5 cross-source links

**Commands**:
```bash
# Store comprehensive notes
npx claude-flow@alpha memory store \
  --key "research/[topic]/notes/[source-id]" \
  --value "[Markdown notes]" \
  --tags "phase:note-taking,source:[title],searchable:true"
```

---

### Phase 6: Synthesis & Dot Connecting (Integration Phase)
**Agent**: coordinator
**Supporting**: researcher (synthesis), analyst (validation)

**Goal**: Connect insights across sources to form evidence-based conclusions

**Actions**:
1. **Pattern Recognition** (researcher):
   - Retrieve ALL notes from Memory MCP
   - Identify recurring themes across sources
   - Find agreements and disagreements
   - Map evidence chains (Source A → supports → Claim X)
2. **Thesis Formation** (researcher):
   - Let thesis emerge FROM evidence (not imposed on it)
   - Draft 1-2 sentence thesis statement
   - List supporting evidence (≥5 sources)
   - List counter-evidence or limitations
3. **Validation** (analyst):
   - Check for circular reasoning
   - Verify all claims have source citations
   - Flag unsupported assertions
   - Confirm primary sources cited for key claims
4. **Output Creation** (coordinator):
   - Structured research summary:
     ```markdown
     # Research Topic: [Topic]

     ## Thesis
     [Evidence-based thesis statement]

     ## Supporting Evidence
     1. [Claim] - Supported by [Source 1, p. X], [Source 2, p. Y]
     2. [Claim] - Supported by [Source 3, p. Z]

     ## Limitations/Caveats
     - [Limitation 1] - Only [N] primary sources available
     - [Limitation 2] - Conflicting accounts on [point]

     ## Primary Sources Referenced
     1. [Source 1] - [Author], [Year]
     2. [Source 2] - [Author], [Year]

     ## Secondary Sources Referenced
     1. [Source A] - [Author], [Year]
     2. [Source B] - [Author], [Year]
     ```
5. Store final synthesis in Memory MCP

**Quality Check**: Thesis supported by ≥5 sources, ≥2 primary sources cited, NO unsupported claims, ≥1 limitation acknowledged

**Commands**:
```bash
# Store synthesis
npx claude-flow@alpha memory store \
  --key "research/[topic]/synthesis" \
  --value "[Final research summary]" \
  --tags "phase:synthesis,validated:true,complete:true"

# Export for use
npx claude-flow@alpha memory retrieve \
  --key "research/[topic]/synthesis"
```

---

## Quality Gates

### Gate 1: After Wikipedia Mining
- **Requirement**: ≥10 citations extracted, ≥3 source categories
- **GO Criteria**: Citations stored in Memory MCP with proper tags
- **NO-GO Action**: Expand Wikipedia search to related articles

### Gate 2: After Source Classification
- **Requirement**: ≥20 sources collected, ≥5 primary sources, ≥80% credibility ≥3
- **GO Criteria**: All sources classified and scored
- **NO-GO Action**: Return to Phase 2 for more source discovery

### Gate 3: After Note-Taking
- **Requirement**: ≥50 notes, ≥20 quotes with page numbers, ≥5 cross-source links
- **GO Criteria**: Notes searchable and tagged in Memory MCP
- **NO-GO Action**: Re-read priority sources for missed insights

### Gate 4: After Synthesis
- **Requirement**: Thesis supported by ≥5 sources, ≥2 primary, NO unsupported claims
- **GO Criteria**: Validation passed by analyst agent
- **NO-GO Action**: Return to Phase 6 for revised synthesis

---

## Success Metrics

### Quantitative
- **Source Diversity**: ≥20 sources, ≥5 primary, ≥3 publication decades
- **Evidence Density**: ≥5 sources supporting thesis
- **Note Depth**: ≥50 notes captured with page references
- **Credibility**: ≥80% sources with credibility score ≥3
- **Time Efficiency**: Complete workflow in 6-10 hours

### Qualitative
- **Context Richness**: Can explain each source's historical/cultural context
- **Bias Awareness**: Identified biases in ≥3 sources
- **Synthesis Quality**: Thesis emerges from evidence (not imposed)
- **Citation Rigor**: All major claims have source citations with page numbers
- **Cross-Source Analysis**: Identified ≥2 agreements and ≥1 disagreement across sources

---

## Integration with Existing Pipeline

### Memory MCP Integration
- All phases store findings with WHO/WHEN/PROJECT/WHY tags
- Cross-session persistence for long research projects
- Vector search for finding related past research

### Agent Coordination
- Uses existing researcher, analyst, coordinator agents
- Follows 12fa agent protocols
- Hooks integration for session tracking

### Complementary Skills
- **Before**: Use `intent-analyzer` to clarify research question
- **During**: Use `source-credibility-analyzer` for Phase 3 (if available)
- **After**: Use `academic-reading-workflow` for deep source reading (if available)
- **Parallel**: Can run alongside `literature-synthesis` for ML research topics

---

## Example Workflow Execution

```bash
# Start research workflow
claude-code invoke-skill general-research-workflow \
  --topic "Byzantine Empire's influence on Renaissance Italy" \
  --duration "8 hours"

# Workflow spawns agents automatically:
# Phase 1: researcher → Wikipedia mining (30 min)
# Phase 2: researcher → Source discovery (2 hours)
# Phase 3: analyst → Source classification (1 hour)
# Phase 4: researcher → Contextual analysis (2 hours)
# Phase 5: researcher → Note-taking (2 hours)
# Phase 6: coordinator → Synthesis (30 min)

# Retrieve final synthesis
npx claude-flow@alpha memory retrieve \
  --key "research/byzantine-renaissance/synthesis"
```

---

## Red's Core Principles Embedded in SOP

1. ✅ **"Trust No One"** → Phase 3 credibility evaluation, bias detection
2. ✅ **"Context is Everything"** → Phase 4 temporal/cultural/historiographical analysis
3. ✅ **"Thesis from Evidence"** → Phase 6 synthesis (thesis emerges, not imposed)
4. ✅ **"Wikipedia is a Gateway"** → Phase 1 reference mining
5. ✅ **"Primary Sources Matter"** → Phase 3 classification, ≥2 primary sources required
6. ✅ **"Page Numbers Save Lives"** → Phase 5 quote extraction with page refs

---

## Next Steps After SOP Design
1. **Optimize with prompt-architect** → Improve agent coordination, add error handling
2. **Build with skill-forge** → Create actual skill files in `.claude/skills/research/`
3. **Test with real research topic** → Validate workflow with 8-hour test run
4. **Integrate with existing skills** → Connect to deep-research-orchestrator as general research branch
