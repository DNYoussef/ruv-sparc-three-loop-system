#!/usr/bin/env python3
"""
Research Orchestrator - Multi-Source Research Aggregation
Version: 1.0.0
Purpose: Parallel API queries with credibility scoring for systematic research

Features:
- Multi-source aggregation (Gemini Search, Semantic Scholar, ArXiv)
- Automatic credibility scoring (0-100%)
- Source deduplication and ranking
- Markdown report generation
- Parallel API execution for speed

Usage:
    python research-orchestrator.py \
      --query "quantum computing error correction" \
      --sources 10 \
      --min-credibility 85 \
      --output research-report.md
"""

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse

try:
    import aiohttp
    import requests
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Error: Missing required dependencies. Install with: pip install aiohttp requests beautifulsoup4")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('research-orchestrator.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Represents a single research source with metadata"""
    title: str
    url: str
    snippet: str
    source_type: str  # "academic", "documentation", "blog", "news"
    credibility_score: float  # 0-100
    publication_date: Optional[str] = None
    authors: List[str] = None
    citations: int = 0

    def __post_init__(self):
        if self.authors is None:
            self.authors = []

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_hash(self) -> str:
        """Generate unique hash for deduplication"""
        content = f"{self.title}|{self.url}".lower()
        return hashlib.md5(content.encode()).hexdigest()


class CredibilityScorer:
    """Calculate credibility scores based on multiple factors"""

    # Tier scoring based on domain reliability
    DOMAIN_TIERS = {
        'tier1': {
            'score': 90,
            'domains': [
                'arxiv.org', 'nature.com', 'science.org', 'acm.org',
                'ieee.org', 'springer.com', 'pubmed.ncbi.nlm.nih.gov',
                'docs.python.org', 'golang.org', 'rust-lang.org'
            ]
        },
        'tier2': {
            'score': 75,
            'domains': [
                'github.com', 'stackoverflow.com', 'medium.com',
                'towardsdatascience.com', 'venturebeat.com', 'techcrunch.com'
            ]
        },
        'tier3': {
            'score': 60,
            'domains': ['reddit.com', 'dev.to', 'hashnode.com']
        }
    }

    @classmethod
    def calculate_score(cls, source: Source) -> float:
        """
        Calculate credibility score (0-100) based on:
        - Authority (30%): Domain tier, author expertise
        - Accuracy (25%): Citation count, peer review status
        - Objectivity (20%): Source type, bias indicators
        - Currency (15%): Publication recency
        - Coverage (10%): Content depth
        """
        score = 0.0

        # Authority (30%)
        domain = urlparse(source.url).netloc.lower()
        for tier_name, tier_data in cls.DOMAIN_TIERS.items():
            if any(d in domain for d in tier_data['domains']):
                score += tier_data['score'] * 0.30
                break
        else:
            score += 50 * 0.30  # Default for unknown domains

        # Accuracy (25%)
        if source.source_type == 'academic':
            score += 90 * 0.25
        elif source.citations > 100:
            score += 85 * 0.25
        elif source.citations > 10:
            score += 70 * 0.25
        else:
            score += 60 * 0.25

        # Objectivity (20%)
        if source.source_type in ['academic', 'documentation']:
            score += 85 * 0.20
        elif source.source_type == 'news':
            score += 70 * 0.20
        else:
            score += 60 * 0.20

        # Currency (15%)
        if source.publication_date:
            try:
                pub_year = int(source.publication_date[:4])
                current_year = datetime.now().year
                age = current_year - pub_year
                if age <= 1:
                    score += 90 * 0.15
                elif age <= 3:
                    score += 75 * 0.15
                elif age <= 5:
                    score += 60 * 0.15
                else:
                    score += 40 * 0.15
            except:
                score += 50 * 0.15
        else:
            score += 50 * 0.15

        # Coverage (10%)
        snippet_words = len(source.snippet.split())
        if snippet_words > 100:
            score += 80 * 0.10
        elif snippet_words > 50:
            score += 70 * 0.10
        else:
            score += 60 * 0.10

        return min(100.0, max(0.0, score))


class ResearchOrchestrator:
    """Orchestrate multi-source research aggregation"""

    def __init__(self, query: str, max_sources: int = 10, min_credibility: float = 70.0):
        self.query = query
        self.max_sources = max_sources
        self.min_credibility = min_credibility
        self.sources: List[Source] = []
        self.seen_hashes: Set[str] = set()

    async def fetch_gemini_search(self, session: aiohttp.ClientSession) -> List[Source]:
        """Fetch results from Gemini Search API (simulated)"""
        logger.info("Fetching Gemini Search results...")
        # Note: In production, use actual Gemini Search API
        # This is a placeholder for demonstration
        return []

    async def fetch_semantic_scholar(self, session: aiohttp.ClientSession) -> List[Source]:
        """Fetch results from Semantic Scholar API"""
        logger.info("Fetching Semantic Scholar results...")
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': self.query,
            'limit': self.max_sources,
            'fields': 'title,abstract,url,year,authors,citationCount'
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    sources = []
                    for paper in data.get('data', [])[:self.max_sources]:
                        source = Source(
                            title=paper.get('title', 'Unknown'),
                            url=paper.get('url', ''),
                            snippet=paper.get('abstract', '')[:300],
                            source_type='academic',
                            publication_date=str(paper.get('year', '')),
                            authors=[a.get('name', '') for a in paper.get('authors', [])],
                            citations=paper.get('citationCount', 0)
                        )
                        sources.append(source)
                    logger.info(f"Retrieved {len(sources)} papers from Semantic Scholar")
                    return sources
        except Exception as e:
            logger.error(f"Semantic Scholar API error: {e}")

        return []

    async def fetch_arxiv(self, session: aiohttp.ClientSession) -> List[Source]:
        """Fetch results from ArXiv API"""
        logger.info("Fetching ArXiv results...")
        url = "http://export.arxiv.org/api/query"
        params = {
            'search_query': f'all:{self.query}',
            'start': 0,
            'max_results': self.max_sources
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    xml_text = await response.text()
                    soup = BeautifulSoup(xml_text, 'xml')
                    sources = []

                    for entry in soup.find_all('entry')[:self.max_sources]:
                        source = Source(
                            title=entry.title.text.strip(),
                            url=entry.id.text.strip(),
                            snippet=entry.summary.text.strip()[:300],
                            source_type='academic',
                            publication_date=entry.published.text[:10],
                            authors=[author.find('name').text for author in entry.find_all('author')]
                        )
                        sources.append(source)

                    logger.info(f"Retrieved {len(sources)} papers from ArXiv")
                    return sources
        except Exception as e:
            logger.error(f"ArXiv API error: {e}")

        return []

    async def aggregate_sources(self) -> List[Source]:
        """Aggregate sources from all APIs in parallel"""
        logger.info(f"Starting research aggregation for query: '{self.query}'")

        async with aiohttp.ClientSession() as session:
            tasks = [
                self.fetch_gemini_search(session),
                self.fetch_semantic_scholar(session),
                self.fetch_arxiv(session)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            all_sources = []
            for result in results:
                if isinstance(result, list):
                    all_sources.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"API error: {result}")

            # Deduplicate and score
            for source in all_sources:
                source_hash = source.get_hash()
                if source_hash not in self.seen_hashes:
                    source.credibility_score = CredibilityScorer.calculate_score(source)
                    if source.credibility_score >= self.min_credibility:
                        self.sources.append(source)
                        self.seen_hashes.add(source_hash)

            # Sort by credibility score (descending)
            self.sources.sort(key=lambda s: s.credibility_score, reverse=True)

            # Limit to max_sources
            self.sources = self.sources[:self.max_sources]

            logger.info(f"Aggregated {len(self.sources)} unique sources (min credibility: {self.min_credibility})")
            return self.sources

    def generate_markdown_report(self) -> str:
        """Generate comprehensive markdown research report"""
        report_lines = [
            f"# Research Report: {self.query}",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Sources**: {len(self.sources)}",
            f"**Minimum Credibility**: {self.min_credibility}%",
            f"\n---\n",
            "## Executive Summary\n",
            f"This report aggregates {len(self.sources)} high-credibility sources on the topic of '{self.query}'. ",
            f"All sources meet the minimum credibility threshold of {self.min_credibility}%.\n",
            "\n## Sources by Credibility\n"
        ]

        for i, source in enumerate(self.sources, 1):
            report_lines.extend([
                f"\n### {i}. {source.title}",
                f"**Credibility Score**: {source.credibility_score:.1f}% | **Type**: {source.source_type}",
                f"**URL**: [{source.url}]({source.url})",
                f"**Published**: {source.publication_date or 'Unknown'} | **Citations**: {source.citations}",
                f"**Authors**: {', '.join(source.authors[:3])}{'...' if len(source.authors) > 3 else ''}",
                f"\n**Summary**: {source.snippet}...\n"
            ])

        # Add bibliography
        report_lines.extend([
            "\n---\n",
            "## Bibliography\n"
        ])

        for i, source in enumerate(self.sources, 1):
            authors = ', '.join(source.authors[:3])
            if len(source.authors) > 3:
                authors += ' et al.'
            report_lines.append(
                f"{i}. {authors}. ({source.publication_date or 'n.d.'}). "
                f"{source.title}. Retrieved from {source.url}\n"
            )

        return '\n'.join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Research Orchestrator - Multi-Source Research Aggregation'
    )
    parser.add_argument('--query', required=True, help='Research query')
    parser.add_argument('--sources', type=int, default=10, help='Maximum sources (default: 10)')
    parser.add_argument('--min-credibility', type=float, default=70.0,
                       help='Minimum credibility score 0-100 (default: 70)')
    parser.add_argument('--output', default='research-report.md', help='Output file (default: research-report.md)')
    parser.add_argument('--json', action='store_true', help='Also output JSON format')

    args = parser.parse_args()

    # Validate arguments
    if args.min_credibility < 0 or args.min_credibility > 100:
        logger.error("Minimum credibility must be between 0 and 100")
        sys.exit(1)

    # Run orchestrator
    orchestrator = ResearchOrchestrator(
        query=args.query,
        max_sources=args.sources,
        min_credibility=args.min_credibility
    )

    # Aggregate sources (async)
    sources = asyncio.run(orchestrator.aggregate_sources())

    # Generate markdown report
    markdown_report = orchestrator.generate_markdown_report()
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(markdown_report)
    logger.info(f"Markdown report saved to: {args.output}")

    # Generate JSON if requested
    if args.json:
        json_output = args.output.replace('.md', '.json')
        json_data = {
            'query': args.query,
            'generated': datetime.now().isoformat(),
            'total_sources': len(sources),
            'min_credibility': args.min_credibility,
            'sources': [s.to_dict() for s in sources]
        }
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"JSON data saved to: {json_output}")

    logger.info(f"Research orchestration complete! Found {len(sources)} sources.")


if __name__ == '__main__':
    main()
