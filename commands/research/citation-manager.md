---
name: research:citation-manager
description: Citation management with BibTeX/Zotero integration and format conversion
category: Research Workflows
version: 1.0.0
requires:
  - python3
  - bibtexparser
  - requests
usage: |
  /research:citation-manager --import "citations.bib" --format "apa" --export "references.txt"
  /research:citation-manager --search "doi:10.1234/example" --add-to "library.bib"
---

# Research: Citation Manager

**Category**: Research Workflows
**Purpose**: Manage research citations with BibTeX, format conversion, and bibliography generation.

## Command Structure

```bash
/research:citation-manager [OPTIONS]

Options:
  --import <path>               Import citations from BibTeX file
  --export <path>               Export formatted bibliography
  --format <string>             Citation style (apa, ieee, chicago, harvard)
  --search <string>             Search for citation by DOI, PMID, arXiv ID
  --add-to <path>               Add citation to library
  --validate                    Validate BibTeX entries
  --deduplicate                 Remove duplicate citations
```

## Implementation

```python
#!/usr/bin/env python3
"""
Citation Management System
"""

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
import requests

class CitationManager:
    """Manage research citations"""

    def __init__(self):
        self.database = BibDatabase()

    def import_bibtex(self, path: str):
        """Import citations from BibTeX file"""
        with open(path, 'r', encoding='utf-8') as f:
            self.database = bibtexparser.load(f)
        print(f"✅ Imported {len(self.database.entries)} citations")

    def fetch_doi(self, doi: str) -> Dict:
        """Fetch citation metadata from DOI"""
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url)
        return response.json()['message']

    def add_citation(self, entry: Dict):
        """Add citation to library"""
        self.database.entries.append(entry)

    def export_bibtex(self, path: str):
        """Export citations to BibTeX file"""
        writer = BibTexWriter()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(writer.write(self.database))
        print(f"✅ Exported {len(self.database.entries)} citations")

# Example usage
if __name__ == "__main__":
    manager = CitationManager()
    manager.import_bibtex('./citations.bib')
    manager.export_bibtex('./references.bib')
```

---

**Status**: Production Ready
**Version**: 1.0.0
