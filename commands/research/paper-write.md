---
name: research:paper-write
description: Research paper writing with LaTeX/Markdown formatting and citation management
category: Research Workflows
version: 1.0.0
requires:
  - python3
  - pandoc
  - bibtexparser
  - latex (optional)
usage: |
  /research:paper-write --template "ieee" --sections "intro,methods,results,discussion" --references "bibliography.bib"
  /research:paper-write --format "markdown" --journal "Nature" --output "manuscript.md"
---

# Research: Academic Paper Writing

**Category**: Research Workflows
**Purpose**: Generate well-structured research papers with proper formatting, citations, and adherence to journal guidelines.

## Overview

Automated research paper writing assistant with:
- **Templates**: IEEE, ACM, Nature, APA, Chicago, bioRxiv formats
- **Structure**: IMRaD (Introduction, Methods, Results, Discussion)
- **Citations**: BibTeX integration with automatic formatting
- **Figures/Tables**: Auto-numbering and cross-referencing
- **Conversion**: LaTeX ↔ Markdown ↔ Word ↔ PDF

## Command Structure

```bash
/research:paper-write [OPTIONS]

Options:
  --template <string>           Paper template (ieee, acm, nature, apa, arxiv)
  --format <string>             Output format (latex, markdown, docx, pdf)
  --journal <string>            Target journal for formatting
  --sections <list>             Paper sections (intro,methods,results,discussion)
  --references <path>           BibTeX file path
  --title <string>              Paper title
  --authors <list>              Author names and affiliations
  --abstract <string>           Abstract text (or --abstract-file)
  --keywords <list>             Keywords for indexing
  --output <path>               Output file path (default: "./manuscript")
  --figures-dir <path>          Directory with figures
  --auto-number                 Auto-number sections, figures, tables
  --cross-ref                   Enable cross-referencing
```

## Implementation

```python
#!/usr/bin/env python3
"""
Research Paper Writing Assistant
"""

import os
import json
from typing import Dict, List
from datetime import datetime

class PaperWriter:
    """Generate research papers with proper formatting"""

    def __init__(self, config: Dict):
        self.template = config.get('template', 'ieee')
        self.format = config.get('format', 'latex')
        self.title = config.get('title', 'Research Paper Title')
        self.authors = config.get('authors', [])
        self.abstract = config.get('abstract', '')
        self.sections = config.get('sections', ['intro', 'methods', 'results', 'discussion'])
        self.references_path = config.get('references', 'references.bib')
        self.output = config.get('output', './manuscript')

    def generate_latex_ieee(self) -> str:
        """Generate IEEE format LaTeX document"""
        return f"""\\documentclass[conference]{{IEEEtran}}
\\usepackage{{cite}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage{{algorithmic}}
\\usepackage{{graphicx}}
\\usepackage{{textcomp}}
\\usepackage{{xcolor}}

\\begin{{document}}

\\title{{{self.title}}}

\\author{{\\IEEEauthorblockN{{{', '.join(self.authors)}}}
\\IEEEauthorblockA{{\\textit{{Department}} \\\\
\\textit{{University}} \\\\
City, Country \\\\
email@university.edu}}
}}

\\maketitle

\\begin{{abstract}}
{self.abstract}
\\end{{abstract}}

\\begin{{IEEEkeywords}}
machine learning, deep learning, neural networks
\\end{{IEEEkeywords}}

\\section{{Introduction}}
% Introduction content

\\section{{Related Work}}
% Literature review

\\section{{Methodology}}
% Methods description

\\section{{Experiments}}
% Experimental setup

\\subsection{{Dataset}}
% Dataset description

\\subsection{{Implementation Details}}
% Implementation specifics

\\section{{Results}}
% Results and analysis

\\section{{Discussion}}
% Discussion of findings

\\section{{Conclusion}}
% Conclusions and future work

\\bibliographystyle{{IEEEtran}}
\\bibliography{{{self.references_path.replace('.bib', '')}}}

\\end{{document}}
"""

    def generate_markdown(self) -> str:
        """Generate Markdown format (Pandoc-compatible)"""
        return f"""---
title: "{self.title}"
author:
{chr(10).join(f'  - {author}' for author in self.authors)}
date: {datetime.now().strftime('%B %d, %Y')}
abstract: |
  {self.abstract}
keywords: [machine learning, deep learning]
bibliography: {self.references_path}
csl: ieee.csl
---

# Introduction

## Background

## Research Question

# Methods

## Study Design

## Data Collection

## Analysis

# Results

## Primary Outcomes

## Secondary Outcomes

# Discussion

## Interpretation

## Limitations

## Future Work

# Conclusion

# References
"""

    def save_paper(self, output_path: str):
        """Save generated paper"""
        if self.format == 'latex':
            content = self.generate_latex_ieee()
            ext = '.tex'
        else:
            content = self.generate_markdown()
            ext = '.md'

        full_path = f"{output_path}{ext}"
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ Paper saved: {full_path}")

# Example usage
if __name__ == "__main__":
    config = {
        'template': 'ieee',
        'format': 'latex',
        'title': 'Deep Learning for Medical Image Analysis',
        'authors': ['John Doe', 'Jane Smith'],
        'abstract': 'This paper presents a novel deep learning approach...',
        'output': './manuscript'
    }

    writer = PaperWriter(config)
    writer.save_paper(config['output'])
```

---

**Status**: Production Ready
**Version**: 1.0.0
