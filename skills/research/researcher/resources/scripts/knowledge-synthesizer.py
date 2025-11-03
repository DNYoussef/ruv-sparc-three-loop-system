#!/usr/bin/env python3
"""
Knowledge Synthesizer - Cross-Reference Synthesis and Conflict Resolution
Version: 1.0.0
Purpose: Synthesize knowledge from multiple sources with conflict resolution

Features:
- Claim extraction and clustering
- Source agreement analysis
- Contradiction detection
- Evidence-based synthesis
- Consensus building

Usage:
    python knowledge-synthesizer.py \
      --sources source1.json source2.json source3.json \
      --mode consensus \
      --output synthesis-report.md
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Set, Tuple, Optional

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Error: Missing ML dependencies. Install with: pip install numpy scikit-learn")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('knowledge-synthesizer.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Claim:
    """Represents a factual claim from a source"""
    text: str
    source_id: str
    source_credibility: float
    section: str = "general"
    confidence: float = 0.5
    supporting_evidence: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.text.lower())


@dataclass
class ClaimCluster:
    """Group of similar claims from multiple sources"""
    representative_claim: str
    claims: List[Claim] = field(default_factory=list)
    agreement_score: float = 0.0
    consensus: Optional[str] = None
    contradictions: List[str] = field(default_factory=list)

    def calculate_agreement(self):
        """Calculate agreement score based on source credibility and count"""
        if not self.claims:
            self.agreement_score = 0.0
            return

        # Weighted by source credibility
        total_weight = sum(c.source_credibility for c in self.claims)
        avg_credibility = total_weight / len(self.claims)

        # Agreement score: combination of source count and credibility
        source_count_score = min(100, len(self.claims) * 20)  # Max 5 sources = 100
        credibility_score = avg_credibility

        self.agreement_score = (source_count_score * 0.4 + credibility_score * 0.6)

    def detect_contradictions(self):
        """Detect contradicting claims using negation patterns"""
        negation_words = {'not', 'no', 'never', 'without', 'cannot', 'isn\'t', 'aren\'t', 'doesn\'t'}

        for i, claim1 in enumerate(self.claims):
            words1 = set(claim1.text.lower().split())
            has_negation1 = bool(words1 & negation_words)

            for claim2 in self.claims[i+1:]:
                words2 = set(claim2.text.lower().split())
                has_negation2 = bool(words2 & negation_words)

                # If one has negation and other doesn't, potential contradiction
                if has_negation1 != has_negation2:
                    # Check if they're talking about similar things
                    overlap = len(words1 & words2) / len(words1 | words2)
                    if overlap > 0.3:
                        self.contradictions.append(
                            f"Source {claim1.source_id}: {claim1.text}\n"
                            f"Source {claim2.source_id}: {claim2.text}"
                        )

    def build_consensus(self):
        """Build consensus statement from claims"""
        if not self.claims:
            return

        # Sort by credibility
        sorted_claims = sorted(self.claims, key=lambda c: c.source_credibility, reverse=True)

        # Use highest credibility claim as base
        base_claim = sorted_claims[0].text

        # Add attribution
        source_count = len(self.claims)
        avg_credibility = sum(c.source_credibility for c in self.claims) / source_count

        self.consensus = (
            f"{base_claim} "
            f"(Supported by {source_count} source{'s' if source_count > 1 else ''}, "
            f"avg credibility: {avg_credibility:.1f}%)"
        )


class KnowledgeSynthesizer:
    """Synthesize knowledge from multiple sources"""

    def __init__(self, source_files: List[str], mode: str = 'consensus'):
        self.source_files = source_files
        self.mode = mode  # 'consensus', 'comprehensive', 'conflict-analysis'
        self.sources: List[Dict] = []
        self.all_claims: List[Claim] = []
        self.claim_clusters: List[ClaimCluster] = []

    def load_sources(self):
        """Load source JSON files"""
        logger.info(f"Loading {len(self.source_files)} source files...")

        for source_file in self.source_files:
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_data = json.load(f)
                    self.sources.append(source_data)
                    logger.info(f"Loaded: {source_file}")
            except Exception as e:
                logger.error(f"Failed to load {source_file}: {e}")

        logger.info(f"Successfully loaded {len(self.sources)} sources")

    def extract_claims(self):
        """Extract claims from sources"""
        logger.info("Extracting claims from sources...")

        for source in self.sources:
            source_id = source.get('query', source.get('id', 'unknown'))

            # Extract from source results
            for result in source.get('sources', []):
                snippet = result.get('snippet', '')
                credibility = result.get('credibility_score', 70.0)

                # Split snippet into sentences (simple approach)
                sentences = re.split(r'[.!?]+', snippet)

                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:  # Minimum claim length
                        claim = Claim(
                            text=sentence,
                            source_id=source_id,
                            source_credibility=credibility
                        )
                        self.all_claims.append(claim)

        logger.info(f"Extracted {len(self.all_claims)} claims")

    def cluster_claims(self, similarity_threshold: float = 0.6):
        """Cluster similar claims using TF-IDF and cosine similarity"""
        logger.info("Clustering similar claims...")

        if not self.all_claims:
            logger.warning("No claims to cluster")
            return

        # Create TF-IDF matrix
        claim_texts = [c.text for c in self.all_claims]
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

        try:
            tfidf_matrix = vectorizer.fit_transform(claim_texts)

            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Cluster using simple threshold-based approach
            visited = set()

            for i in range(len(self.all_claims)):
                if i in visited:
                    continue

                # Create new cluster
                cluster = ClaimCluster(representative_claim=self.all_claims[i].text)
                cluster.claims.append(self.all_claims[i])
                visited.add(i)

                # Find similar claims
                for j in range(i + 1, len(self.all_claims)):
                    if j not in visited and similarity_matrix[i][j] >= similarity_threshold:
                        cluster.claims.append(self.all_claims[j])
                        visited.add(j)

                # Calculate agreement and detect contradictions
                cluster.calculate_agreement()
                cluster.detect_contradictions()
                cluster.build_consensus()

                self.claim_clusters.append(cluster)

            logger.info(f"Created {len(self.claim_clusters)} claim clusters")

        except Exception as e:
            logger.error(f"Clustering failed: {e}")

    def generate_synthesis_report(self) -> str:
        """Generate comprehensive synthesis report"""
        logger.info("Generating synthesis report...")

        # Sort clusters by agreement score
        self.claim_clusters.sort(key=lambda c: c.agreement_score, reverse=True)

        lines = [
            "# Knowledge Synthesis Report",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Sources**: {len(self.sources)}",
            f"**Total Claims**: {len(self.all_claims)}",
            f"**Claim Clusters**: {len(self.claim_clusters)}",
            f"**Mode**: {self.mode}",
            "\n---\n",
            "## Executive Summary\n",
            f"This synthesis aggregates knowledge from {len(self.sources)} sources, ",
            f"extracting {len(self.all_claims)} factual claims and organizing them into ",
            f"{len(self.claim_clusters)} thematic clusters. Conflicts and consensus are identified.\n"
        ]

        # High agreement claims (consensus)
        lines.append("\n## Consensus Findings (High Agreement)\n")
        high_agreement = [c for c in self.claim_clusters if c.agreement_score >= 70]

        for i, cluster in enumerate(high_agreement[:10], 1):
            lines.extend([
                f"\n### {i}. {cluster.representative_claim[:100]}...",
                f"**Agreement Score**: {cluster.agreement_score:.1f}%",
                f"**Sources**: {len(cluster.claims)}",
                f"\n**Consensus**: {cluster.consensus}\n"
            ])

        # Contradictions
        lines.append("\n## Detected Contradictions\n")
        contradicting_clusters = [c for c in self.claim_clusters if c.contradictions]

        if contradicting_clusters:
            for i, cluster in enumerate(contradicting_clusters[:5], 1):
                lines.extend([
                    f"\n### Contradiction {i}",
                    f"**Topic**: {cluster.representative_claim[:100]}...",
                    f"\n**Conflicting Claims**:\n"
                ])
                for contradiction in cluster.contradictions:
                    lines.append(f"```\n{contradiction}\n```\n")
        else:
            lines.append("No major contradictions detected.\n")

        # Moderate agreement (needs further investigation)
        lines.append("\n## Moderate Agreement (Further Investigation Needed)\n")
        moderate_agreement = [c for c in self.claim_clusters
                             if 40 <= c.agreement_score < 70]

        for i, cluster in enumerate(moderate_agreement[:5], 1):
            lines.extend([
                f"\n### {i}. {cluster.representative_claim[:100]}...",
                f"**Agreement Score**: {cluster.agreement_score:.1f}%",
                f"**Sources**: {len(cluster.claims)}",
                f"\n**Note**: Limited source agreement. Verify independently.\n"
            ])

        # Source summary
        lines.append("\n---\n\n## Source Summary\n")
        for i, source in enumerate(self.sources, 1):
            source_id = source.get('query', source.get('id', 'unknown'))
            source_count = len(source.get('sources', []))
            lines.append(f"{i}. **{source_id}**: {source_count} sources\n")

        return '\n'.join(lines)

    def save_report(self, output_path: str):
        """Save synthesis report to file"""
        report = self.generate_synthesis_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Synthesis report saved to: {output_path}")

        # Also save JSON data
        json_path = output_path.replace('.md', '.json')
        json_data = {
            'generated': datetime.now().isoformat(),
            'sources': len(self.sources),
            'total_claims': len(self.all_claims),
            'clusters': len(self.claim_clusters),
            'high_agreement_count': len([c for c in self.claim_clusters if c.agreement_score >= 70]),
            'contradictions_count': len([c for c in self.claim_clusters if c.contradictions])
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"JSON data saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Synthesizer - Cross-Reference Synthesis'
    )
    parser.add_argument('--sources', nargs='+', required=True,
                       help='Source JSON files to synthesize')
    parser.add_argument('--mode', default='consensus',
                       choices=['consensus', 'comprehensive', 'conflict-analysis'],
                       help='Synthesis mode (default: consensus)')
    parser.add_argument('--similarity', type=float, default=0.6,
                       help='Claim similarity threshold 0-1 (default: 0.6)')
    parser.add_argument('--output', default='synthesis-report.md',
                       help='Output file (default: synthesis-report.md)')

    args = parser.parse_args()

    # Validate source files
    for source_file in args.sources:
        if not source_file.endswith('.json'):
            logger.error(f"Source file must be JSON: {source_file}")
            sys.exit(1)

    # Run synthesizer
    synthesizer = KnowledgeSynthesizer(args.sources, args.mode)
    synthesizer.load_sources()
    synthesizer.extract_claims()
    synthesizer.cluster_claims(similarity_threshold=args.similarity)
    synthesizer.save_report(args.output)

    logger.info("Knowledge synthesis complete!")


if __name__ == '__main__':
    main()
