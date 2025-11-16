# Academic Reading Workflow - SOP (Standard Operating Procedure)

**Version**: 1.0 (Draft)
**Type**: Sequential Agent Workflow
**Duration**: 2-6 hours per source
**Agents Required**: researcher, analyst

---

## Purpose

Systematic methodology for reading academic papers, books, and complex texts using Blue's (OSP) 3-phase approach: summary-first reading, active annotation with searchable notes, and evidence-based writing.

---

## When to Invoke

**Trigger Conditions**:
- Reading academic papers, books, or complex texts
- Need to deeply understand and retain information
- Building searchable knowledge base from readings
- Preparing to write essays or analyses based on sources

**Do NOT Use For**:
- Quick skimming (<30 min)
- Casual reading without note-taking needs
- Fiction/entertainment reading

---

## Agent Sequence

| Step | Agent | Role | Input | Output | Duration |
|------|-------|------|-------|--------|----------|
| 1 | researcher | Summary-First Reading | Source(s) to read | Roadmap summary + reading plan | 15-30 min |
| 2 | researcher | Deep Reading + Annotation | Reading plan | Annotated text with searchable notes | 1-4 hours |
| 3 | analyst | Annotation Quality Check | Annotated notes | Validated note index | 15-30 min |
| 4 | researcher | Evidence-Based Writing (Optional) | Notes + prompt | Draft with citations | 1-2 hours |

**Total Duration**: 2-6 hours per source (Steps 1-3), +1-2 hours if writing (Step 4)

---

## Step-by-Step Procedure

### STEP 1: Summary-First Reading (Roadmap Phase)
**Agent**: researcher
**Objective**: Create reading roadmap BEFORE deep dive to avoid getting lost

**Instructions to Agent**:
1. **Read summary materials FIRST** (in order of priority):
   - Abstract (for papers)
   - Introduction and Conclusion chapters (for books)
   - Table of Contents
   - Chapter/section headers
   - Wikipedia article on the topic (if available)
   - Existing book reviews or paper summaries

2. Create reading roadmap:
   - Identify main argument/thesis
   - List key sections/chapters (with page ranges)
   - Note what you already know vs what's new
   - Identify which sections are critical vs supplementary
   - Estimate reading time per section

3. Establish reading focus:
   - What question am I trying to answer with this reading?
   - Which sections directly address my research question?
   - What can I skim vs read carefully?

4. Create reading plan:
   - **Critical sections** (read carefully, annotate heavily): [List with pages]
   - **Supplementary sections** (skim for key points): [List with pages]
   - **Skip sections** (not relevant): [List with pages]

**Deliverable**:
```markdown
# READING ROADMAP: [Source Title]

## Main Argument
[1-2 sentence thesis]

## Critical Sections (Read Carefully)
1. [Chapter/Section] (pages X-Y) - [Why critical]
2. [Chapter/Section] (pages A-B) - [Why critical]

## Supplementary Sections (Skim)
1. [Chapter/Section] (pages M-N) - [What to extract]

## Skip Sections
1. [Chapter/Section] - [Why skipping]

## Reading Focus Question
[What am I trying to learn from this source?]

## Estimated Time: X hours
```

**Quality Gate 1**: STOP if roadmap is vague. Must have clear thesis + critical sections identified.

**Handoff**: Pass reading plan to Step 2 (researcher)

---

### STEP 2: Deep Reading + Active Annotation
**Agent**: researcher
**Objective**: Read with "searchable margin notes" - create "command-F in real life"

**Instructions to Agent**:
Follow reading plan from Step 1. For EACH critical section:

#### A. Read Actively (Not Passively)
- Read with pen/highlighter in hand (or digital equivalent)
- Pause at end of each paragraph to check understanding
- Ask: "What's the point of this paragraph?"
- Don't just highlight—ANNOTATE with your thoughts

#### B. Create Searchable Margin Notes
Use this annotation system:

**REQUIRED Components** (for EVERY annotated passage):
- ✅ **Keyword tags**: 2-5 keywords for this passage (e.g., #methodology, #key-claim, #evidence)
- ✅ **Page number**: Exact page reference
- ✅ **Your own summary**: Paraphrase in YOUR words (not copy-paste)
- ⚠️ **Direct quote** (if quotable): Exact text in "quotes" with page
- ⚠️ **Why this matters**: How does this connect to your reading focus?

**Annotation Template**:
```markdown
## PAGE X: [Your Title for This Section]

**Keywords**: #keyword1 #keyword2 #keyword3

**Summary**: [Your paraphrase in 1-3 sentences]

**Quote** (if notable): "Exact text here" (p. X)

**Why This Matters**: [Connection to your research question]

**Links**: See also [Page Y], Conflicts with [Source B, p. Z]
```

#### C. Annotation Principles (Blue's Rules)

**1. "Command-F in Real Life"**:
- Tag EVERY concept with searchable keywords
- Later, you can search your notes for #methodology to find ALL passages about methods
- Consistent keyword vocabulary (use same tags for same concepts)

**2. "Margin Notes Are For Future You"**:
- Don't just write "important!" - write WHY it's important
- Include enough context that you can understand the note 6 months later
- Link related passages ("See also page 42")

**3. "Paraphrase > Highlighting"**:
- Highlighting alone doesn't create understanding
- Force yourself to paraphrase in YOUR words
- If you can't paraphrase it, you didn't understand it—re-read

**4. "Direct Quotes Are Precious"**:
- Only quote when the EXACT wording matters
- Quote when it's so well-said you want to use it in writing
- Always include page number for later citation

**5. "Track Changes in Your Understanding"**:
- If page 87 contradicts what you thought on page 15, NOTE IT
- "Initially thought X (p. 15), but author clarifies Y (p. 87)"
- This prevents confusion later

#### D. Annotation Levels by Section Type

**Critical Sections** (from Step 1 plan):
- Annotate EVERY major claim
- Extract 2-5 quotes
- Create 5-10 margin notes per section
- Heavy keyword tagging

**Supplementary Sections**:
- Annotate only key points
- 1-2 quotes if notable
- 2-3 margin notes per section
- Light keyword tagging

**Skipped Sections**:
- Quick skim, minimal/no annotation

#### E. Store Notes in Searchable Format

Store all annotations in Memory MCP with:
- **WHO**: researcher
- **WHEN**: [timestamp]
- **PROJECT**: [research-topic]
- **WHY**: reading-annotation
- **TAGS**: All keyword tags for searchability

**Deliverable**:
- Annotated text/document
- 20-50 searchable margin notes (depending on source length)
- Keyword index (all tags used)

**Quality Gate 2**: STOP if <20 margin notes for full paper/chapter OR <5 keywords used. Insufficient annotation depth.

**Handoff**: Pass annotated notes to Step 3 (analyst)

---

### STEP 3: Annotation Quality Check
**Agent**: analyst
**Objective**: Validate that annotations are searchable, useful, and complete

**Instructions to Agent**:

#### A. Check Annotation Completeness
For each annotation, verify:
- ✅ Has keyword tags? (≥2 keywords)
- ✅ Has page number?
- ✅ Has paraphrase in reader's own words?
- ⚠️ Has direct quote if claiming to quote?
- ⚠️ Links to related passages if claiming connection?

**Flag incomplete annotations** → return to Step 2 for revision

#### B. Check Keyword Consistency
1. Extract all keywords used
2. Check for duplicates/synonyms:
   - Is "#method" and "#methodology" referring to same thing? → Standardize
   - Is "#keypoint" too vague? → Make specific
3. Verify keywords are searchable:
   - Avoid one-off keywords that won't help future searches
   - Use domain-standard terms when possible

**Create Keyword Index**:
```markdown
# KEYWORD INDEX: [Source Title]

## Keywords Used (Alphabetical)
- #argument (pages 15, 42, 88)
- #bias (pages 23, 67)
- #evidence (pages 15, 28, 35, 49, 72)
- #key-claim (pages 12, 34, 56)
- #limitation (pages 89, 91)
- #methodology (pages 8, 10, 45)

Total keywords: 6
Total annotations: 32
Average annotations per keyword: 5.3
```

#### C. Check Paraphrase Quality
Sample 5-10 annotations at random. For each:
- Is the paraphrase in the reader's OWN words?
- Does it capture the essence of the original passage?
- Is it understandable without referring back to source?

**If >30% of paraphrases are just slightly reworded quotes** → return to Step 2

#### D. Assess Searchability
Test the annotation system:
1. Pick a concept from the source (e.g., "research methodology")
2. Search annotations using keywords (e.g., #methodology)
3. Can you find ALL relevant passages quickly?
4. Are the notes useful on their own?

**Quality Gate 3**:
- **GO**: ≥20 annotations, ≥5 keywords, <30% quote-paraphrases, searchable index works
- **NO-GO**: Return to Step 2 for additional annotation

**Deliverable**:
- Validated annotation set
- Keyword index
- Quality assessment report

**Handoff**: Pass validated notes to Step 4 (optional) OR store for future use

---

### STEP 4: Evidence-Based Writing (Optional)
**Agent**: researcher
**Objective**: Use annotations to write essay/analysis with proper citations

**Instructions to Agent** (if writing based on this reading):

#### A. Pre-Writing: Gather Relevant Annotations
1. Identify your writing focus/thesis
2. Search annotations using keywords
3. Group annotations by theme:
   - Arguments supporting your thesis
   - Counter-arguments or limitations
   - Evidence/data
   - Methodological notes

#### B. Blue's Writing Principles

**1. "Thesis Comes LAST, Not First"**:
- Don't impose a thesis on the evidence
- Let your thesis EMERGE from the annotations
- If the reading contradicts your initial hypothesis, acknowledge it

**2. "Relativist Arguments" (Acknowledge Complexity)**:
- Avoid absolute statements ("This PROVES...")
- Use nuanced language ("This suggests...", "Evidence indicates...")
- Acknowledge limitations and counter-evidence
- Example: "While Smith argues X, this interpretation is challenged by Jones (p. 42) who notes..."

**3. "Every Claim Needs a Source"**:
- NEVER make unsupported assertions
- Every factual claim = citation with page number
- Use your annotations to find exact page references
- Format: "Author argues that [claim] (p. X)"

**4. "Write Like You Speak (Then Edit)"**:
- First draft: Write naturally, explain as if talking to friend
- Use your paraphrases from annotations (they're already in your words!)
- Later: Polish for formal tone while keeping clarity

#### C. Writing Workflow

**Draft 1: Evidence Gathering**
- Pull relevant annotations for each paragraph
- Copy page references into draft outline
- Ensure every major claim has ≥1 citation

**Draft 2: Logical Flow**
- Organize evidence into coherent argument
- Check for circular reasoning
- Ensure each paragraph builds on previous

**Draft 3: Citation Check**
- Verify EVERY claim has source + page
- Use consistent citation format
- Add counter-evidence sections where appropriate

**Draft 4: Relativism Check**
- Replace absolute statements with qualified ones
- Add acknowledgments of complexity
- Include "However" and "On the other hand" sections

**Deliverable**:
- Draft essay with complete citations
- All major claims sourced to page numbers
- Acknowledgment of limitations/counter-evidence

**Quality Gate 4** (if writing):
- **GO**: Every major claim cited, relativist language used, no unsupported assertions
- **NO-GO**: Add missing citations, qualify absolute statements

---

## Blue's Core Principles Embedded in SOP

| Principle | Implementation |
|-----------|---------------|
| **"Read the Roadmap Before You Get Lost"** | Step 1: Summary-first approach, create reading plan BEFORE diving in |
| **"Annotation is Command-F in Real Life"** | Step 2: Keyword tagging system for searchable notes |
| **"Write Like You Speak"** | Step 4: Draft naturally using your paraphrases, polish later |
| **"Thesis Comes LAST, Not First"** | Step 4: Let thesis EMERGE from annotations, not imposed |
| **"Every Claim Needs a Source"** | Step 4: All assertions require citation + page number |
| **"Paraphrase > Highlighting"** | Step 2: Force paraphrase in own words, not just copy-paste |

---

## Success Metrics

### Quantitative
- ✅ Reading roadmap created (Step 1)
- ✅ ≥20 margin notes for full paper/chapter
- ✅ ≥5 consistent keywords used
- ✅ ≥2 keywords per annotation
- ✅ Page numbers for ALL quotes and major claims
- ✅ <30% quote-paraphrases (most should be genuine paraphrases)
- ✅ Keyword index searchable

### Qualitative (Step 3 validation)
- ✅ Can find relevant passages using keyword search
- ✅ Paraphrases understandable without source
- ✅ Annotations useful 6 months later
- ✅ Links between related passages documented
- ✅ If writing: Every claim cited, relativist language used

---

## Integration with Other Skills

### Before This Skill:
- Use `general-research-workflow` Step 2-3 to find and classify sources FIRST
- Prioritize which sources to read deeply using credibility/priority scores

### During This Skill:
- Can annotate multiple sources in parallel
- Use same keyword vocabulary across all sources for cross-source searchability

### After This Skill:
- Use annotations in `general-research-workflow` Step 5 (Note-Taking)
- Use annotations in `research-publication` for paper writing
- Export keyword index to build personal knowledge base

---

## Error Handling

| Failure Mode | Gate | Resolution |
|--------------|------|------------|
| Vague roadmap | 1 | Re-read abstract/intro, clarify main argument |
| Getting lost while reading | 1 | Return to roadmap, refocus on critical sections |
| <20 annotations | 2 | Extend reading time, annotate more thoroughly |
| <5 keywords | 2 | Review notes, add more specific keywords |
| >30% quote-paraphrases | 3 | Force genuine paraphrasing, re-read if needed |
| Keyword inconsistency | 3 | Standardize terms, update annotations |
| Can't find passages via keywords | 3 | Add more keywords, improve tagging |
| Unsupported claims in writing | 4 | Add citations, return to annotations for page refs |

---

## Example Workflow Execution

```
Source: "Byzantium and the Renaissance" by N.G. Wilson (300 pages)

Step 1 (30 min):
- Read abstract, intro, conclusion
- Roadmap: Main argument = "Greek scholars fleeing Constantinople transformed Italian humanism"
- Critical sections: Chapters 2-4 (Greek migration), Chapter 7 (Manuscript collections)
- Skim: Chapters 1, 5-6
- Skip: Appendices
- Reading focus: "How did Greek scholarship specifically influence Renaissance Italy?"

Step 2 (3 hours):
- Read Chapters 2-4 carefully (120 pages)
- Created 35 margin notes with keywords: #greek-migration #humanist-education #manuscripts #bessarion #plethon #venice #florence
- Extracted 8 direct quotes
- Linked related passages across chapters
- Skimmed Chapters 1, 5-6 (10 notes)
- Total: 45 annotations

Step 3 (20 min):
- Analyst validation: All notes have ≥2 keywords, page numbers present
- Keyword index: 12 keywords, average 3.75 uses each
- Paraphrase check: 90% genuine paraphrases (good)
- Searchability test: Searched #manuscripts, found all 8 relevant passages → PASS

Step 4 (Optional, 1.5 hours if writing):
- Gathered annotations for thesis: "Greek scholars brought pedagogical methods + manuscript collections"
- Draft paragraph 1: Used notes from p. 45, 67, 88 about education methods
- All claims cited (p. 45), (p. 67), (p. 88)
- Added relativist qualifier: "Wilson argues that... however this interpretation..."
```

---

## Next Steps After SOP Design
1. **Optimize with prompt-architect** → Add few-shot examples, improve annotation template
2. **Build with skill-forge** → Create actual skill in `.claude/skills/`
3. **Test with real academic paper** → Validate with 20-page paper reading
4. **Integrate with general-research-workflow** → Connect Step 5 note-taking to this annotation system

---

**END OF SOP - Version 1.0 Draft**
