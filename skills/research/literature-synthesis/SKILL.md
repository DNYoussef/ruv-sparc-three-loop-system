---
name: literature-synthesis
description: Systematic literature review and synthesis for Deep Research SOP Pipeline
  A. Use when starting research projects, conducting SOTA analysis, identifying research
  gaps, or preparing academic papers. Implements PRISMA-compliant systematic review
  methodology with automated search, screening, and synthesis across ArXiv, Semantic
  Scholar, and Papers with Code.
version: 1.0.0
category: research
tags:
- research
- analysis
- planning
author: ruv
---

# Literature Synthesis

Conduct systematic literature reviews following PRISMA guidelines, synthesizing state-of-the-art research to identify gaps and opportunities for Deep Research SOP Phase 1.

## Overview

**Purpose**: Systematic literature review identifying SOTA methods, research gaps, and opportunities

**When to Use**:
- Starting new research projects (Phase 1 of Deep Research SOP)
- Conducting state-of-the-art (SOTA) analysis
- Identifying research gaps and opportunities
- Preparing related work sections for papers
- Validating novelty claims for proposed methods
- Quality Gate 1 requirement

**Quality Gate**: Required for Quality Gate 1 (minimum 50 papers)

**Prerequisites**:
- Research question formulated
- Search databases accessible (ArXiv, Semantic Scholar, Papers with Code)
- Reference management tool available (Zotero, Mendeley, BibTeX)

**Outputs**:
- Literature review document (50-100+ papers)
- SOTA performance benchmarks table
- Research gap analysis
- Hypothesis formulation
- Citation database (BibTeX)
- PRISMA flow diagram (if systematic review)

**Time Estimate**: 1-2 weeks
- Database search: 2-4 hours
- Screening: 1-2 days
- Full-text review: 3-5 days
- Synthesis: 2-3 days
- Writing: 1-2 days

**Agents Used**: researcher

---

## Quick Start

### 1. Define Search Query
```bash
# Store research question in memory
npx claude-flow@alpha memory store \
  --key "sop/literature-review/research-question" \
  --value "How does multi-scale attention improve long-range dependency modeling in vision transformers?"

# Define search terms
search_terms="(multi-scale OR hierarchical) AND (attention OR transformer) AND (vision OR image)"
```

### 2. Database Search
```bash
# Search ArXiv
python scripts/search_arxiv.py \
  --query "$search_terms" \
  --start-date "2020-01-01" \
  --max-results 500 \
  --output literature/arxiv_results.json

# Search Semantic Scholar
python scripts/search_semantic_scholar.py \
  --query "$search_terms" \
  --fields "title,abstract,authors,year,citationCount,venue" \
  --min-citations 10 \
  --output literature/semantic_scholar_results.json

# Search Papers with Code
python scripts/search_papers_with_code.py \
  --task "image-classification" \
  --method "transformer" \
  --output literature/pwc_results.json
```

### 3. Screening and Selection
```bash
# Title/abstract screening
python scripts/screen_papers.py \
  --input literature/*_results.json \
  --inclusion-criteria literature/inclusion_criteria.yaml \
  --output literature/screened_papers.json

# Full-text review
python scripts/full_text_review.py \
  --input literature/screened_papers.json \
  --download-dir literature/pdfs/ \
  --output literature/selected_papers.json
```

### 4. Synthesis
```bash
# Extract SOTA benchmarks
python scripts/extract_sota_benchmarks.py \
  --papers literature/selected_papers.json \
  --datasets "ImageNet,CIFAR-10,CIFAR-100" \
  --output literature/sota_benchmarks.csv

# Identify research gaps
python scripts/identify_gaps.py \
  --papers literature/selected_papers.json \
  --output literature/research_gaps.md
```

### 5. Generate Literature Review
```bash
# Generate review document
python scripts/generate_literature_review.py \
  --papers literature/selected_papers.json \
  --benchmarks literature/sota_benchmarks.csv \
  --gaps literature/research_gaps.md \
  --template templates/literature_review_template.md \
  --output docs/literature_review.md
```

---

## Detailed Instructions

### Phase 1: Search Strategy Development (2-4 hours)

**Objective**: Define comprehensive search strategy

**Steps**:

#### 1.1 Formulate Research Question (PICO Framework)
```markdown
## Research Question (PICO)

**Population**: Vision transformer models
**Intervention**: Multi-scale attention mechanisms
**Comparison**: Standard single-scale attention
**Outcome**: Image classification accuracy, computational efficiency
```

#### 1.2 Define Inclusion/Exclusion Criteria
```yaml
# literature/inclusion_criteria.yaml

inclusion:
  - Published 2020 or later
  - Peer-reviewed (conferences: NeurIPS, ICML, CVPR, ICCV, ECCV; journals: TPAMI, IJCV)
  - Focuses on attention mechanisms in vision
  - Reports quantitative results on standard benchmarks
  - Available in English

exclusion:
  - Preprints without peer review (unless highly cited >100)
  - Not related to computer vision
  - No experimental results
  - Duplicate publications (keep most recent)
  - Surveys and position papers (catalog separately)
```

#### 1.3 Define Search Terms
```python
# scripts/search_terms.py

search_terms = {
    "attention_terms": [
        "attention mechanism",
        "self-attention",
        "multi-head attention",
        "cross-attention",
        "multi-scale attention"
    ],
    "architecture_terms": [
        "vision transformer",
        "ViT",
        "Swin Transformer",
        "hierarchical transformer",
        "pyramid transformer"
    ],
    "task_terms": [
        "image classification",
        "object detection",
        "semantic segmentation"
    ]
}

# Construct boolean query
query = " OR ".join(search_terms["attention_terms"]) + " AND " + \
        " OR ".join(search_terms["architecture_terms"])
```

**Deliverable**: Search strategy document

---

### Phase 2: Database Search (2-4 hours)

**Objective**: Retrieve papers from multiple databases

**Steps**:

#### 2.1 ArXiv Search
```python
# scripts/search_arxiv.py
import arxiv

client = arxiv.Client()
search = arxiv.Search(
    query="(multi-scale OR hierarchical) AND (attention OR transformer) AND vision",
    max_results=500,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending
)

results = []
for paper in client.results(search):
    results.append({
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "abstract": paper.summary,
        "published": paper.published.isoformat(),
        "arxiv_id": paper.entry_id.split("/")[-1],
        "pdf_url": paper.pdf_url,
        "categories": paper.categories
    })

# Save results
import json
with open("literature/arxiv_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Retrieved {len(results)} papers from ArXiv")
```

#### 2.2 Semantic Scholar Search
```python
# scripts/search_semantic_scholar.py
import requests

API_KEY = "YOUR_S2_API_KEY"
ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"

params = {
    "query": "(multi-scale OR hierarchical) AND (attention OR transformer) AND vision",
    "fields": "title,abstract,authors,year,citationCount,venue,publicationDate",
    "limit": 100,
    "minCitationCount": 10,
    "year": "2020-"
}

response = requests.get(ENDPOINT, params=params, headers={"x-api-key": API_KEY})
papers = response.json()["data"]

# Save results
with open("literature/semantic_scholar_results.json", "w") as f:
    json.dump(papers, f, indent=2)

print(f"Retrieved {len(papers)} papers from Semantic Scholar")
```

#### 2.3 Papers with Code Search
```bash
# Download Papers with Code dataset
curl -o literature/pwc_dataset.json \
  https://paperswithcode.com/api/v1/papers/?tasks=image-classification&methods=transformer

# Filter relevant papers
python scripts/filter_pwc_papers.py \
  --input literature/pwc_dataset.json \
  --keywords "attention,multi-scale,hierarchical" \
  --output literature/pwc_results.json
```

**Deliverable**: Combined database of ~200-500 papers

---

### Phase 3: Screening (1-2 days)

**Objective**: Screen papers based on inclusion/exclusion criteria

**Steps**:

#### 3.1 Remove Duplicates
```python
# scripts/remove_duplicates.py
from fuzzywuzzy import fuzz

def find_duplicates(papers, threshold=90):
    duplicates = []
    for i, p1 in enumerate(papers):
        for j, p2 in enumerate(papers[i+1:], i+1):
            similarity = fuzz.ratio(p1["title"], p2["title"])
            if similarity >= threshold:
                # Keep the one with more citations
                keep = i if p1.get("citationCount", 0) >= p2.get("citationCount", 0) else j
                remove = j if keep == i else i
                duplicates.append(remove)
    return list(set(duplicates))

# Remove duplicates
papers = load_papers("literature/*_results.json")
duplicates = find_duplicates(papers)
unique_papers = [p for i, p in enumerate(papers) if i not in duplicates]

print(f"Removed {len(duplicates)} duplicates, {len(unique_papers)} unique papers remain")
```

#### 3.2 Title/Abstract Screening
```python
# scripts/screen_papers.py
import yaml

# Load inclusion criteria
with open("literature/inclusion_criteria.yaml") as f:
    criteria = yaml.safe_load(f)

def screen_paper(paper, criteria):
    """Screen paper based on title and abstract."""
    # Check publication year
    year = int(paper.get("year", paper.get("published", "2000")[:4]))
    if year < 2020:
        return False, "Published before 2020"

    # Check for relevant keywords in title/abstract
    text = f"{paper['title']} {paper.get('abstract', '')}".lower()
    relevant_keywords = ["attention", "transformer", "vision", "image"]
    if not any(kw in text for kw in relevant_keywords):
        return False, "Not relevant to research question"

    # Check for exclusion criteria
    if "survey" in text or "review" in text:
        return False, "Survey or review paper"

    return True, "Included"

# Screen all papers
screened = []
for paper in unique_papers:
    include, reason = screen_paper(paper, criteria)
    paper["screening_decision"] = "include" if include else "exclude"
    paper["screening_reason"] = reason
    if include:
        screened.append(paper)

print(f"{len(screened)}/{len(unique_papers)} papers passed title/abstract screening")
```

**Deliverable**: ~50-100 papers after screening

---

### Phase 4: Full-Text Review (3-5 days)

**Objective**: Review full papers for final inclusion

**Steps**:

#### 4.1 Download PDFs
```python
# scripts/download_pdfs.py
import requests
from pathlib import Path

pdf_dir = Path("literature/pdfs")
pdf_dir.mkdir(exist_ok=True)

for paper in screened_papers:
    pdf_url = paper.get("pdf_url") or paper.get("openAccessPdf", {}).get("url")
    if pdf_url:
        filename = f"{paper['arxiv_id'] or paper['paperId']}.pdf"
        response = requests.get(pdf_url)
        (pdf_dir / filename).write_bytes(response.content)
        print(f"Downloaded: {filename}")
```

#### 4.2 Extract Key Information
```python
# scripts/extract_paper_info.py

def extract_info(paper_pdf):
    """Extract key information from paper."""
    info = {
        "methods": [],  # Novel methods proposed
        "datasets": [],  # Datasets used
        "metrics": {},  # Performance metrics
        "comparisons": [],  # Baseline comparisons
        "limitations": [],  # Reported limitations
        "future_work": []  # Suggested future work
    }

    # Use PyPDF2 or pdfplumber to extract text
    # Use regex or NLP to extract structured information
    # (Implementation details omitted for brevity)

    return info

# Extract info from all papers
for paper in screened_papers:
    pdf_path = f"literature/pdfs/{paper['arxiv_id']}.pdf"
    if Path(pdf_path).exists():
        paper["extracted_info"] = extract_info(pdf_path)
```

#### 4.3 Final Inclusion Decision
```python
# Manual review criteria
def final_review(paper):
    """Final inclusion decision based on full text."""
    info = paper.get("extracted_info", {})

    # Must report quantitative results
    if not info.get("metrics"):
        return False, "No quantitative results"

    # Must use standard benchmarks
    standard_datasets = ["ImageNet", "CIFAR-10", "CIFAR-100", "COCO"]
    if not any(ds in info.get("datasets", []) for ds in standard_datasets):
        return False, "No standard benchmarks"

    # Must compare with baselines
    if not info.get("comparisons"):
        return False, "No baseline comparisons"

    return True, "Included"

# Final review
selected_papers = []
for paper in screened_papers:
    include, reason = final_review(paper)
    if include:
        selected_papers.append(paper)

print(f"{len(selected_papers)} papers selected for synthesis")
```

**Deliverable**: 50-70 final selected papers

---

### Phase 5: Synthesis (2-3 days)

**Objective**: Synthesize findings into coherent narrative

**Steps**:

#### 5.1 Extract SOTA Benchmarks
```python
# scripts/extract_sota_benchmarks.py
import pandas as pd

benchmarks = []
for paper in selected_papers:
    metrics = paper["extracted_info"]["metrics"]
    for dataset, results in metrics.items():
        benchmarks.append({
            "paper": paper["title"],
            "year": paper["year"],
            "method": paper["extracted_info"]["methods"][0] if paper["extracted_info"]["methods"] else "N/A",
            "dataset": dataset,
            "metric": results.get("accuracy") or results.get("top1"),
            "citation_count": paper.get("citationCount", 0)
        })

df = pd.DataFrame(benchmarks)
df.to_csv("literature/sota_benchmarks.csv", index=False)

# Find best performance per dataset
sota = df.groupby("dataset")["metric"].max()
print("State-of-the-Art Performance:")
print(sota)
```

#### 5.2 Identify Research Gaps
```python
# scripts/identify_gaps.py

gaps = {
    "methodological_gaps": [],
    "application_gaps": [],
    "evaluation_gaps": []
}

# Methodological gaps
methods_used = set()
for paper in selected_papers:
    methods_used.update(paper["extracted_info"]["methods"])

# Check for unexplored combinations
all_attention_types = ["self-attention", "cross-attention", "multi-scale", "sparse", "local", "global"]
explored = [m for m in all_attention_types if any(m in method for method in methods_used)]
unexplored = [m for m in all_attention_types if m not in explored]

gaps["methodological_gaps"] = unexplored

# Application gaps
datasets_used = set()
for paper in selected_papers:
    datasets_used.update(paper["extracted_info"]["datasets"])

standard_datasets = ["ImageNet", "CIFAR-10", "COCO", "ADE20K", "Cityscapes"]
underexplored = [d for d in standard_datasets if d not in datasets_used]

gaps["application_gaps"] = underexplored

# Write gaps analysis
with open("literature/research_gaps.md", "w") as f:
    f.write("# Research Gaps Analysis\n\n")
    f.write("## Methodological Gaps\n")
    for gap in gaps["methodological_gaps"]:
        f.write(f"- {gap}: Not explored in reviewed papers\n")
    f.write("\n## Application Gaps\n")
    for gap in gaps["application_gaps"]:
        f.write(f"- {gap}: Underexplored dataset\n")
```

#### 5.3 Formulate Hypotheses
```markdown
# Hypotheses (based on research gaps)

## H1: Multi-Scale Local-Global Attention
**Gap**: No papers combine local and global attention at multiple scales
**Hypothesis**: Combining local (window-based) and global (full-image) attention at multiple scales will improve long-range dependency modeling while maintaining computational efficiency
**Expected Improvement**: +2-5% accuracy on ImageNet
**Testable**: Compare multi-scale local-global vs. single-scale global attention

## H2: Sparse Attention for High-Resolution Images
**Gap**: Limited exploration of sparse attention for high-resolution inputs (>1024px)
**Hypothesis**: Learned sparse attention patterns can reduce computational cost O(n²) → O(n log n) for high-resolution images without accuracy loss
**Expected Improvement**: 2-3x speedup with <1% accuracy drop
**Testable**: Compare sparse vs. dense attention on high-res datasets
```

**Deliverable**: Research gaps and hypotheses document

---

### Phase 6: Writing (1-2 days)

**Objective**: Write comprehensive literature review

**Steps**:

#### 6.1 Structure Literature Review
```markdown
# Literature Review: Multi-Scale Attention in Vision Transformers

## 1. Introduction
- Research question and motivation
- Scope of review (50+ papers, 2020-2025)
- Organization of review

## 2. Background
- Vision Transformers fundamentals
- Attention mechanisms overview
- Multi-scale representations in vision

## 3. Attention Mechanisms in Vision
### 3.1 Self-Attention
- Vanilla self-attention (Vaswani et al., 2017)
- Vision Transformer (Dosovitskiy et al., 2021)

### 3.2 Hierarchical Attention
- Swin Transformer (Liu et al., 2021)
- Pyramid Vision Transformer (Wang et al., 2021)

### 3.3 Multi-Scale Attention
- CrossViT (Chen et al., 2021)
- Multi-Scale Vision Transformer (Fan et al., 2021)

## 4. State-of-the-Art Performance
[Table of SOTA benchmarks]

## 5. Research Gaps
- Methodological gaps
- Application gaps
- Evaluation gaps

## 6. Proposed Direction
- Hypotheses based on gaps
- Expected contributions

## 7. Conclusion
```

#### 6.2 Generate BibTeX Citations
```python
# scripts/generate_bibtex.py

def paper_to_bibtex(paper):
    """Convert paper to BibTeX entry."""
    authors = " and ".join(paper["authors"])
    year = paper["year"]
    title = paper["title"]
    venue = paper.get("venue", "arXiv")

    bibtex_key = f"{paper['authors'][0].split()[-1]}{year}"

    return f"""@inproceedings{{{bibtex_key},
  title={{{title}}},
  author={{{authors}}},
  booktitle={{{venue}}},
  year={{{year}}}
}}"""

# Generate bibliography
with open("literature/bibliography.bib", "w") as f:
    for paper in selected_papers:
        f.write(paper_to_bibtex(paper) + "\n\n")
```

**Deliverable**: Complete literature review document

---

### Phase 7: PRISMA Flow Diagram (Optional, 1 hour)

**Objective**: Create PRISMA flow diagram for systematic reviews

```python
# scripts/generate_prisma_diagram.py

prisma_data = {
    "identification": {
        "records_identified": 500,
        "duplicates_removed": 150
    },
    "screening": {
        "records_screened": 350,
        "records_excluded": 250
    },
    "eligibility": {
        "full_text_assessed": 100,
        "full_text_excluded": 30,
        "exclusion_reasons": {
            "no_quantitative_results": 15,
            "no_standard_benchmarks": 10,
            "not_peer_reviewed": 5
        }
    },
    "included": {
        "studies_included": 70
    }
}

# Generate PRISMA diagram using GraphViz
# (Implementation omitted for brevity)
```

**Deliverable**: PRISMA flow diagram

---

## Integration with Deep Research SOP

### Pipeline Integration
- **Pipeline A (Literature Synthesis)**: This skill implements the complete literature review phase
- **Phase 1 (Foundations)**: Required before baseline replication
- **Quality Gate 1**: Minimum 50 papers required

### Agent Coordination
```
researcher agent:
  - Conducts database searches
  - Screens papers
  - Extracts key information
  - Synthesizes findings
  - Formulates hypotheses
```

### Memory Coordination
```bash
# Store literature review in memory
npx claude-flow@alpha memory store \
  --key "sop/phase1/literature-review" \
  --value "$(cat docs/literature_review.md)"

# Store SOTA benchmarks
npx claude-flow@alpha memory store \
  --key "sop/phase1/sota-benchmarks" \
  --value "$(cat literature/sota_benchmarks.csv)"
```

---

## Troubleshooting

### Issue: Too few papers found (<50)
**Solution**: Broaden search terms, expand date range, include preprints

### Issue: Too many papers (>200)
**Solution**: Add inclusion criteria (minimum citations, specific venues)

### Issue: Gate 1 validation fails due to incomplete review
**Solution**: Ensure ≥50 papers reviewed, SOTA benchmarks documented, research gaps identified

---

## Related Skills and Commands

### Prerequisites
- None (first skill in Deep Research SOP workflow)

### Next Steps
- `baseline-replication` - Replicate SOTA baselines identified in literature review
- `gate-validation --gate 1` - Validate Phase 1 completion

### Related Commands
- `/prisma-init` - Initialize PRISMA systematic review (researcher agent)

---

## References

### Systematic Review Methodologies
- PRISMA 2020 (Page et al., 2021): Systematic review reporting guidelines
- Cochrane Handbook: Gold standard for systematic reviews

### Search Databases
- ArXiv: https://arxiv.org/
- Semantic Scholar: https://www.semanticscholar.org/
- Papers with Code: https://paperswithcode.com/

### Reference Management
- Zotero: https://www.zotero.org/
- BibTeX format specification