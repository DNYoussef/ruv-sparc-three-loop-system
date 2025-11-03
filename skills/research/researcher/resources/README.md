# Researcher Resources

## Overview

This directory contains production-grade scripts, templates, and utilities for systematic research workflows. All components integrate with the 3-level research methodology (Basic, Multi-Source, Deep Dive) defined in the main researcher skill.

## Directory Structure

```
resources/
├── README.md (this file)
├── scripts/
│   ├── research-orchestrator.py     # Multi-source research aggregation
│   ├── sota-analyzer.js             # State-of-the-art analysis with citations
│   ├── paper-summarizer.sh          # Academic paper extraction
│   └── knowledge-synthesizer.py     # Cross-reference synthesis
└── templates/
    ├── research-plan.yaml           # Systematic research workflow
    ├── literature-review.json       # Paper organization template
    └── findings-report.yaml         # Research deliverables template
```

## Production Scripts

### research-orchestrator.py (280+ lines)
**Purpose**: Multi-source research aggregation with credibility scoring
**Use Cases**:
- Level 2: Multi-source research (3-5 sources)
- Level 3: Deep dive research (10+ sources)
**Features**:
- Parallel API queries (Gemini Search, Semantic Scholar, ArXiv)
- Automatic credibility scoring (0-100%)
- Source deduplication and ranking
- Markdown report generation

**Usage**:
```bash
python research-orchestrator.py \
  --query "quantum computing error correction" \
  --sources 10 \
  --min-credibility 85 \
  --output research-report.md
```

### sota-analyzer.js (300+ lines)
**Purpose**: State-of-the-art analysis with citation management
**Use Cases**:
- Academic literature review
- Technology trend analysis
- Competitive landscape research
**Features**:
- Papers with Code integration
- Citation network analysis
- Reproducibility scoring
- Performance metrics comparison

**Usage**:
```bash
node sota-analyzer.js \
  --domain "computer-vision" \
  --task "object-detection" \
  --years 2020-2024 \
  --output sota-report.json
```

### paper-summarizer.sh (220+ lines)
**Purpose**: Academic paper extraction and summarization
**Use Cases**:
- PDF parsing and text extraction
- Section-based summarization
- Key findings extraction
**Features**:
- PDF to text conversion (pdftotext, tesseract OCR)
- Section detection (abstract, methods, results, conclusion)
- Citation extraction
- LaTeX formula support

**Usage**:
```bash
./paper-summarizer.sh \
  --pdf paper.pdf \
  --extract-citations \
  --output summary.md
```

### knowledge-synthesizer.py (350+ lines)
**Purpose**: Cross-reference synthesis and conflict resolution
**Use Cases**:
- Level 3: Deep dive synthesis
- Multi-source fact checking
- Consensus building
**Features**:
- Claim extraction and clustering
- Source agreement analysis
- Contradiction detection
- Evidence-based synthesis

**Usage**:
```bash
python knowledge-synthesizer.py \
  --sources source1.json source2.json source3.json \
  --mode consensus \
  --output synthesis-report.md
```

## Templates

### research-plan.yaml
Systematic research workflow template with:
- Research objectives
- Source prioritization
- Timeline milestones
- Quality checkpoints

### literature-review.json
Paper organization template with:
- Citation metadata
- Key findings extraction
- Methodology notes
- Relevance scoring

### findings-report.yaml
Research deliverables template with:
- Executive summary
- Detailed findings
- Source bibliography
- Recommendations

## Integration with Main Skill

All scripts and templates integrate seamlessly with the researcher skill workflow:

1. **Level 1 (Basic)**: Use Gemini Search directly, no scripts needed
2. **Level 2 (Multi-Source)**: Use `research-orchestrator.py` for 3-5 source aggregation
3. **Level 3 (Deep Dive)**: Use full script suite for comprehensive research

## Requirements

**Python Dependencies**:
```bash
pip install requests beautifulsoup4 scholarly arxiv semanticscholar pyyaml
```

**Node.js Dependencies**:
```bash
npm install axios cheerio citation-js json2csv
```

**Shell Tools**:
- `pdftotext` (poppler-utils)
- `tesseract` (OCR)
- `jq` (JSON processing)

## Quality Standards

All scripts follow production-grade quality standards:
- ✅ Input validation and error handling
- ✅ Comprehensive logging
- ✅ Configurable via CLI arguments
- ✅ JSON/YAML/Markdown output formats
- ✅ Credibility scoring (0-100%)
- ✅ Source citation with links
- ✅ Reproducible execution

## Examples

See `examples/` directory for complete workflow examples demonstrating:
- Technology research (example-1)
- Academic literature review (example-2)
- Competitive analysis (example-3)

Each example shows end-to-end research process using scripts and templates.
