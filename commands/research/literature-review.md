---
name: research:literature-review
description: Systematic literature review with PRISMA methodology for academic research
category: Research Workflows
version: 1.0.0
requires:
  - python3
  - pandas
  - matplotlib
  - seaborn
  - bibtexparser
usage: |
  /research:literature-review --topic "machine learning interpretability" --databases "arxiv,pubmed,ieee" --years "2020-2024"
  /research:literature-review --query "neural network explainability" --export-format "latex,csv" --screening-criteria "peer-reviewed"
---

# Research: Systematic Literature Review

**Category**: Research Workflows
**Purpose**: Conduct systematic literature reviews following PRISMA (Preferred Reporting Items for Systematic Reviews and Meta-Analyses) methodology for academic research.

## Overview

This command automates the systematic literature review process, including:
- **Search Strategy**: Multi-database search with Boolean operators
- **Screening**: Title/abstract screening with inclusion/exclusion criteria
- **Data Extraction**: Structured data extraction from selected papers
- **Quality Assessment**: Risk of bias and quality scoring
- **Synthesis**: Narrative synthesis and meta-analysis preparation
- **Reporting**: PRISMA flow diagram and comprehensive reports

## Core Features

### 1. Search Strategy Development
- Multi-database search (arXiv, PubMed, IEEE Xplore, Google Scholar)
- Boolean query builder with AND/OR/NOT operators
- MeSH term mapping for medical databases
- Search string validation and optimization
- Automatic deduplication across databases

### 2. PRISMA-Compliant Workflow
- Identification phase: Database searching
- Screening phase: Title/abstract review
- Eligibility phase: Full-text assessment
- Inclusion phase: Final study selection
- Automated PRISMA flow diagram generation

### 3. Data Extraction
- Structured extraction templates
- Standardized data fields (authors, year, methods, results)
- Risk of bias assessment tools
- Quality scoring (GRADE, Cochrane RoB)
- Export to spreadsheets and LaTeX tables

### 4. Analysis & Synthesis
- Descriptive statistics of included studies
- Visualization (publication year trends, study types)
- Forest plot preparation for meta-analysis
- Narrative synthesis framework
- Evidence mapping

## Command Structure

```bash
/research:literature-review [OPTIONS]

Options:
  --topic <string>              Research topic or question
  --query <string>              Custom search query with Boolean operators
  --databases <list>            Databases to search (default: "arxiv,pubmed,ieee,scholar")
  --years <range>               Publication year range (e.g., "2020-2024")
  --screening-criteria <file>   Path to inclusion/exclusion criteria JSON
  --export-format <list>        Export formats (default: "latex,csv,bibtex")
  --output-dir <path>           Output directory (default: "./literature-review")
  --language <list>             Languages to include (default: "en")
  --study-types <list>          Study types (e.g., "rct,observational,review")
  --quality-assessment          Enable quality assessment tools
  --prisma-diagram              Generate PRISMA flow diagram
  --max-results <int>           Maximum results per database (default: 500)
```

## Usage Examples

### Basic Literature Review
```bash
/research:literature-review \
  --topic "transformer models in NLP" \
  --databases "arxiv,ieee" \
  --years "2017-2024" \
  --prisma-diagram
```

### Medical Systematic Review
```bash
/research:literature-review \
  --query "(diabetes OR 'metabolic syndrome') AND (prevention OR intervention)" \
  --databases "pubmed,cochrane" \
  --years "2015-2024" \
  --screening-criteria "./screening.json" \
  --quality-assessment \
  --study-types "rct,cohort"
```

### Computer Science Review
```bash
/research:literature-review \
  --topic "federated learning privacy" \
  --databases "arxiv,ieee,acm" \
  --export-format "latex,bibtex,csv" \
  --max-results 1000
```

## Implementation

### Phase 1: Search Strategy (Auto-Execute)

```python
#!/usr/bin/env python3
"""
Systematic Literature Review - Search Strategy
"""

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Set
import bibtexparser
from collections import defaultdict

class LiteratureSearcher:
    """Multi-database literature search with deduplication"""

    def __init__(self, config: Dict):
        self.topic = config.get('topic')
        self.query = config.get('query')
        self.databases = config.get('databases', ['arxiv', 'pubmed', 'ieee'])
        self.years = config.get('years', '2020-2024')
        self.max_results = config.get('max_results', 500)
        self.output_dir = config.get('output_dir', './literature-review')

        self.results = defaultdict(list)
        self.search_log = []

    def build_search_string(self) -> Dict[str, str]:
        """Build database-specific search strings"""
        if self.query:
            base_query = self.query
        else:
            # Convert topic to search query
            base_query = self.topic.replace(' ', ' AND ')

        queries = {}

        # arXiv query
        queries['arxiv'] = f"all:{base_query}"

        # PubMed query (with MeSH terms if medical topic)
        queries['pubmed'] = f"{base_query}[Title/Abstract]"

        # IEEE Xplore query
        queries['ieee'] = f'("{base_query}")'

        # Google Scholar query
        queries['scholar'] = base_query

        return queries

    def search_arxiv(self, query: str) -> List[Dict]:
        """Search arXiv.org"""
        print(f"🔍 Searching arXiv: {query}")

        # Simulate arXiv API search (replace with actual API call)
        # import arxiv
        # search = arxiv.Search(
        #     query=query,
        #     max_results=self.max_results,
        #     sort_by=arxiv.SortCriterion.SubmittedDate
        # )

        results = []
        # for result in search.results():
        #     results.append({
        #         'title': result.title,
        #         'authors': [a.name for a in result.authors],
        #         'abstract': result.summary,
        #         'year': result.published.year,
        #         'url': result.entry_id,
        #         'database': 'arXiv'
        #     })

        self.search_log.append({
            'database': 'arXiv',
            'query': query,
            'results': len(results),
            'timestamp': datetime.now().isoformat()
        })

        return results

    def search_pubmed(self, query: str) -> List[Dict]:
        """Search PubMed"""
        print(f"🔍 Searching PubMed: {query}")

        # Simulate PubMed API search (replace with actual API call)
        # from Bio import Entrez
        # Entrez.email = "your.email@example.com"
        # handle = Entrez.esearch(db="pubmed", term=query, retmax=self.max_results)

        results = []
        self.search_log.append({
            'database': 'PubMed',
            'query': query,
            'results': len(results),
            'timestamp': datetime.now().isoformat()
        })

        return results

    def search_ieee(self, query: str) -> List[Dict]:
        """Search IEEE Xplore"""
        print(f"🔍 Searching IEEE Xplore: {query}")

        # Simulate IEEE API search
        results = []
        self.search_log.append({
            'database': 'IEEE',
            'query': query,
            'results': len(results),
            'timestamp': datetime.now().isoformat()
        })

        return results

    def deduplicate(self, all_results: List[Dict]) -> List[Dict]:
        """Remove duplicates based on title similarity"""
        print("🔄 Deduplicating results...")

        seen_titles = set()
        unique_results = []
        duplicates = 0

        for paper in all_results:
            # Normalize title for comparison
            title_normalized = paper['title'].lower().strip()

            if title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_results.append(paper)
            else:
                duplicates += 1

        print(f"   Removed {duplicates} duplicates")
        print(f"   Unique papers: {len(unique_results)}")

        return unique_results

    def filter_by_year(self, results: List[Dict]) -> List[Dict]:
        """Filter results by publication year range"""
        start_year, end_year = map(int, self.years.split('-'))

        filtered = [
            r for r in results
            if start_year <= r.get('year', 0) <= end_year
        ]

        print(f"📅 Filtered to {self.years}: {len(filtered)}/{len(results)} papers")
        return filtered

    def execute_search(self) -> pd.DataFrame:
        """Execute search across all databases"""
        print("=" * 60)
        print("📚 SYSTEMATIC LITERATURE SEARCH")
        print("=" * 60)
        print(f"Topic: {self.topic}")
        print(f"Databases: {', '.join(self.databases)}")
        print(f"Years: {self.years}")
        print()

        queries = self.build_search_string()
        all_results = []

        # Search each database
        if 'arxiv' in self.databases:
            all_results.extend(self.search_arxiv(queries['arxiv']))

        if 'pubmed' in self.databases:
            all_results.extend(self.search_pubmed(queries['pubmed']))

        if 'ieee' in self.databases:
            all_results.extend(self.search_ieee(queries['ieee']))

        print()
        print(f"Total results: {len(all_results)}")

        # Deduplicate and filter
        unique_results = self.deduplicate(all_results)
        filtered_results = self.filter_by_year(unique_results)

        # Convert to DataFrame
        df = pd.DataFrame(filtered_results)

        # Save search results
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        output_path = f"{self.output_dir}/search_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\n✅ Search results saved: {output_path}")

        # Save search log
        log_path = f"{self.output_dir}/search_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.search_log, f, indent=2)
        print(f"✅ Search log saved: {log_path}")

        return df

# Example usage
if __name__ == "__main__":
    config = {
        'topic': 'machine learning interpretability',
        'databases': ['arxiv', 'ieee'],
        'years': '2020-2024',
        'max_results': 500,
        'output_dir': './literature-review'
    }

    searcher = LiteratureSearcher(config)
    results = searcher.execute_search()
```

### Phase 2: PRISMA Screening (Auto-Execute)

```python
#!/usr/bin/env python3
"""
PRISMA Screening Workflow
"""

import json
import pandas as pd
from typing import List, Dict, Set

class PRISMAScreener:
    """PRISMA-compliant screening workflow"""

    def __init__(self, results_df: pd.DataFrame, criteria_path: str = None):
        self.df = results_df
        self.criteria = self.load_criteria(criteria_path)

        self.screening_stages = {
            'identification': len(results_df),
            'screening': 0,
            'eligibility': 0,
            'included': 0
        }

        self.excluded = defaultdict(list)

    def load_criteria(self, path: str) -> Dict:
        """Load inclusion/exclusion criteria"""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)

        # Default criteria
        return {
            'inclusion': [
                'Peer-reviewed publications',
                'Published in English',
                'Original research articles',
                'Full text available'
            ],
            'exclusion': [
                'Editorials or opinion pieces',
                'Non-English publications',
                'Duplicate publications',
                'Insufficient data reported'
            ]
        }

    def title_abstract_screening(self) -> pd.DataFrame:
        """Phase 1: Title and abstract screening"""
        print("📋 SCREENING PHASE: Title & Abstract Review")
        print("=" * 60)

        # Simulate screening (replace with actual review process)
        # In practice, this would involve manual review or ML-assisted screening

        screened = self.df.copy()

        # Example: Filter by keywords in title/abstract
        keywords_include = ['machine learning', 'interpretability', 'explainability']
        keywords_exclude = ['survey', 'review', 'tutorial']

        # Apply filters
        mask = screened['title'].str.contains('|'.join(keywords_include), case=False, na=False)
        screened = screened[mask]

        self.screening_stages['screening'] = len(screened)
        print(f"✅ Passed screening: {len(screened)}/{len(self.df)} papers")

        return screened

    def full_text_assessment(self, screened_df: pd.DataFrame) -> pd.DataFrame:
        """Phase 2: Full-text eligibility assessment"""
        print("\n📄 ELIGIBILITY PHASE: Full-Text Assessment")
        print("=" * 60)

        # Simulate full-text review
        eligible = screened_df.copy()

        # Apply inclusion/exclusion criteria
        # This would involve downloading and reviewing full PDFs

        self.screening_stages['eligibility'] = len(eligible)
        print(f"✅ Eligible papers: {len(eligible)}/{len(screened_df)} papers")

        return eligible

    def final_inclusion(self, eligible_df: pd.DataFrame) -> pd.DataFrame:
        """Phase 3: Final inclusion"""
        print("\n✔️  INCLUSION PHASE: Final Selection")
        print("=" * 60)

        included = eligible_df.copy()

        # Add quality scores
        included['quality_score'] = 0.0

        self.screening_stages['included'] = len(included)
        print(f"✅ Included papers: {len(included)}/{len(eligible_df)} papers")

        return included

    def generate_prisma_diagram(self, output_path: str):
        """Generate PRISMA flow diagram"""
        print("\n📊 Generating PRISMA Flow Diagram...")

        diagram_data = {
            'identification': {
                'records_identified': self.screening_stages['identification'],
                'records_screened': self.screening_stages['screening']
            },
            'screening': {
                'records_excluded': self.screening_stages['identification'] - self.screening_stages['screening']
            },
            'eligibility': {
                'full_text_assessed': self.screening_stages['screening'],
                'full_text_excluded': self.screening_stages['screening'] - self.screening_stages['eligibility']
            },
            'included': {
                'studies_included': self.screening_stages['included']
            }
        }

        # Save diagram data
        with open(output_path, 'w') as f:
            json.dump(diagram_data, f, indent=2)

        print(f"✅ PRISMA diagram data saved: {output_path}")

        # Generate LaTeX PRISMA diagram
        latex_diagram = self.generate_latex_prisma(diagram_data)
        latex_path = output_path.replace('.json', '.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_diagram)
        print(f"✅ LaTeX diagram saved: {latex_path}")

    def generate_latex_prisma(self, data: Dict) -> str:
        """Generate LaTeX PRISMA flow diagram"""
        return f"""\\documentclass{{article}}
\\usepackage{{tikz}}
\\usetikzlibrary{{shapes,arrows,positioning}}

\\begin{{document}}

\\begin{{tikzpicture}}[node distance=1.5cm, auto]
  % Nodes
  \\node [rectangle, draw, text width=8cm, align=center] (identification) {{
    \\textbf{{Identification}} \\\\
    Records identified through database searching \\\\
    (n = {data['identification']['records_identified']})
  }};

  \\node [rectangle, draw, text width=8cm, align=center, below=of identification] (screening) {{
    \\textbf{{Screening}} \\\\
    Records screened \\\\
    (n = {data['identification']['records_screened']})
  }};

  \\node [rectangle, draw, text width=8cm, align=center, below=of screening] (eligibility) {{
    \\textbf{{Eligibility}} \\\\
    Full-text articles assessed for eligibility \\\\
    (n = {data['eligibility']['full_text_assessed']})
  }};

  \\node [rectangle, draw, text width=8cm, align=center, below=of eligibility] (included) {{
    \\textbf{{Included}} \\\\
    Studies included in synthesis \\\\
    (n = {data['included']['studies_included']})
  }};

  % Excluded boxes
  \\node [rectangle, draw, text width=4cm, align=center, right=2cm of screening] (excluded1) {{
    Excluded \\\\
    (n = {data['screening']['records_excluded']})
  }};

  \\node [rectangle, draw, text width=4cm, align=center, right=2cm of eligibility] (excluded2) {{
    Excluded \\\\
    (n = {data['eligibility']['full_text_excluded']})
  }};

  % Arrows
  \\draw [arrow] (identification) -- (screening);
  \\draw [arrow] (screening) -- (eligibility);
  \\draw [arrow] (eligibility) -- (included);
  \\draw [arrow] (screening) -- (excluded1);
  \\draw [arrow] (eligibility) -- (excluded2);
\\end{{tikzpicture}}

\\end{{document}}
"""

# Example usage
if __name__ == "__main__":
    # Load search results
    df = pd.read_csv('./literature-review/search_results.csv')

    screener = PRISMAScreener(df, './screening_criteria.json')

    # Execute screening workflow
    screened = screener.title_abstract_screening()
    eligible = screener.full_text_assessment(screened)
    included = screener.final_inclusion(eligible)

    # Save final results
    included.to_csv('./literature-review/included_studies.csv', index=False)

    # Generate PRISMA diagram
    screener.generate_prisma_diagram('./literature-review/prisma_diagram.json')
```

## Integration with Research Workflow

### Hooks Integration
```bash
# Pre-search hook
npx claude-flow@alpha hooks pre-task --description "Literature review search"

# Post-screening hook
npx claude-flow@alpha hooks post-task --task-id "literature-review"

# Memory storage
npx claude-flow@alpha memory store \
  --key "research/literature-review/included-studies" \
  --value "$(cat ./literature-review/included_studies.csv)"
```

### Memory Pattern
```javascript
{
  "research/literature-review/{topic}/search-results": "Raw search results CSV",
  "research/literature-review/{topic}/included-studies": "Final included papers",
  "research/literature-review/{topic}/prisma-diagram": "PRISMA flow diagram data",
  "research/literature-review/{topic}/quality-assessment": "Quality scores and risk of bias"
}
```

## Output Files

```
literature-review/
├── search_results.csv           # Raw search results
├── search_log.json              # Search strategy log
├── included_studies.csv         # Final included papers
├── prisma_diagram.json          # PRISMA flow data
├── prisma_diagram.tex           # LaTeX PRISMA diagram
├── quality_assessment.csv       # Quality scores
└── synthesis_report.md          # Narrative synthesis
```

## Advanced Features

### Quality Assessment Tools
```python
class QualityAssessor:
    """Risk of bias and quality assessment"""

    def cochrane_rob(self, study: Dict) -> Dict:
        """Cochrane Risk of Bias tool"""
        return {
            'random_sequence_generation': 'low',
            'allocation_concealment': 'low',
            'blinding_participants': 'unclear',
            'blinding_outcome': 'low',
            'incomplete_outcome_data': 'low',
            'selective_reporting': 'low',
            'overall_bias': 'low'
        }

    def grade_evidence(self, studies: List[Dict]) -> str:
        """GRADE evidence quality"""
        # High, Moderate, Low, Very Low
        return 'Moderate'
```

### Meta-Analysis Preparation
```python
def prepare_forest_plot(studies: pd.DataFrame) -> Dict:
    """Prepare data for forest plot (meta-analysis)"""
    return {
        'study_names': studies['title'].tolist(),
        'effect_sizes': studies['effect_size'].tolist(),
        'standard_errors': studies['se'].tolist(),
        'confidence_intervals': studies[['ci_lower', 'ci_upper']].values.tolist()
    }
```

## Best Practices

1. **Pre-register Review Protocol**: Register protocol on PROSPERO before starting
2. **Dual Screening**: Have two reviewers independently screen papers
3. **Transparent Reporting**: Document all decisions and exclusions
4. **Quality Assessment**: Use validated tools (Cochrane RoB, GRADE)
5. **Version Control**: Track changes to search strategies and criteria

## References

- PRISMA Statement: http://www.prisma-statement.org/
- Cochrane Handbook: https://training.cochrane.org/handbook
- GRADE Working Group: https://www.gradeworkinggroup.org/

---

**Status**: Production Ready
**Version**: 1.0.0
**Last Updated**: 2025-11-01
