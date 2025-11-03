#!/usr/bin/env python3
"""
Intent Classifier - Probabilistic Intent Category Analysis

Classifies user requests into intent categories using pattern matching and
probabilistic analysis. Outputs confidence scores for each category and flags
multi-intent or ambiguous requests requiring clarification.

Usage:
    python intent-classifier.py "user request text"
    python intent-classifier.py --verbose "user request text"
    echo "user request" | python intent-classifier.py --stdin

Categories:
    - Creative: Writing, design, ideation, content generation
    - Analytical: Evaluation, comparison, assessment, analysis
    - Technical: Coding, debugging, building, implementation
    - Learning: Explanation, teaching, understanding, clarification
    - Decision: Choosing options, recommendations, planning
    - Problem-Solving: Debugging, optimization, fixing issues

Output: JSON with category probabilities and metadata
"""

import sys
import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class IntentAnalysis:
    """Structured intent analysis result"""
    primary_category: str
    confidence: float
    categories: Dict[str, float]
    signals: List[str]
    multi_intent: bool
    ambiguous: bool
    requires_clarification: bool
    reasoning: str


class IntentClassifier:
    """Classify user intent using pattern-based probabilistic analysis"""

    # Intent category patterns with weighted signals
    PATTERNS = {
        'creative': {
            'verbs': ['write', 'create', 'design', 'generate', 'compose', 'draft', 'brainstorm', 'ideate', 'come up with'],
            'nouns': ['content', 'story', 'article', 'design', 'idea', 'concept', 'creative'],
            'phrases': ['come up with', 'help me write', 'generate ideas'],
            'weight': 1.0
        },
        'analytical': {
            'verbs': ['analyze', 'evaluate', 'assess', 'compare', 'review', 'examine', 'investigate', 'study'],
            'nouns': ['analysis', 'evaluation', 'assessment', 'comparison', 'review', 'tradeoff'],
            'phrases': ['what are the pros and cons', 'compare and contrast', 'evaluate whether'],
            'weight': 1.0
        },
        'technical': {
            'verbs': ['code', 'implement', 'build', 'fix', 'debug', 'develop', 'program', 'configure'],
            'nouns': ['code', 'function', 'bug', 'error', 'implementation', 'software', 'system'],
            'phrases': ['write code', 'fix the bug', 'implement this', 'build a system'],
            'weight': 1.0
        },
        'learning': {
            'verbs': ['explain', 'teach', 'show', 'clarify', 'help understand', 'demonstrate'],
            'nouns': ['explanation', 'understanding', 'concept', 'tutorial', 'lesson'],
            'phrases': ['how does', 'why does', 'what is', "i don't understand", 'teach me', 'explain to me'],
            'weight': 1.0
        },
        'decision': {
            'verbs': ['recommend', 'suggest', 'choose', 'decide', 'should'],
            'nouns': ['recommendation', 'suggestion', 'decision', 'choice', 'option'],
            'phrases': ['should i', 'which is better', 'what do you recommend', 'help me choose', 'which one'],
            'weight': 1.0
        },
        'problem_solving': {
            'verbs': ['solve', 'troubleshoot', 'diagnose', 'resolve', 'fix', 'optimize'],
            'nouns': ['problem', 'issue', 'solution', 'fix', 'optimization', 'improvement'],
            'phrases': ['not working', 'having trouble', 'keep getting error', 'make it better'],
            'weight': 1.0
        }
    }

    # Confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        'high': 0.80,       # Proceed with dominant interpretation
        'moderate': 0.50,   # Proceed but acknowledge assumption
        'low': 0.50         # Seek clarification
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _extract_signals(self, text: str, category: str) -> List[Tuple[str, str]]:
        """Extract matching signals for a category"""
        text_lower = text.lower()
        patterns = self.PATTERNS[category]
        signals = []

        # Check verbs
        for verb in patterns['verbs']:
            if re.search(r'\b' + re.escape(verb) + r'\b', text_lower):
                signals.append(('verb', verb))

        # Check nouns
        for noun in patterns['nouns']:
            if re.search(r'\b' + re.escape(noun) + r'\b', text_lower):
                signals.append(('noun', noun))

        # Check phrases
        for phrase in patterns['phrases']:
            if phrase in text_lower:
                signals.append(('phrase', phrase))

        return signals

    def _calculate_score(self, signals: List[Tuple[str, str]]) -> float:
        """Calculate weighted score from signals"""
        if not signals:
            return 0.0

        # Weight: phrases > verbs > nouns
        weights = {'phrase': 2.0, 'verb': 1.5, 'noun': 1.0}
        total_weight = sum(weights[signal_type] for signal_type, _ in signals)

        # Normalize to 0-1 range with diminishing returns for multiple signals
        # Score = 1 - e^(-weight/3) caps around 0.95 for many signals
        import math
        score = 1 - math.exp(-total_weight / 3)
        return min(score, 1.0)

    def classify(self, text: str) -> IntentAnalysis:
        """Classify intent with confidence scores"""

        # Extract signals for each category
        category_signals = {}
        category_scores = {}

        for category in self.PATTERNS.keys():
            signals = self._extract_signals(text, category)
            category_signals[category] = signals
            category_scores[category] = self._calculate_score(signals)

        # Normalize scores to probabilities
        total_score = sum(category_scores.values())
        if total_score > 0:
            probabilities = {cat: score/total_score for cat, score in category_scores.items()}
        else:
            # No clear signals - uniform distribution
            probabilities = {cat: 1.0/len(self.PATTERNS) for cat in self.PATTERNS.keys()}

        # Determine primary category and confidence
        primary_category = max(probabilities, key=probabilities.get)
        confidence = probabilities[primary_category]

        # Detect multi-intent (multiple categories > 0.25 probability)
        high_prob_categories = [cat for cat, prob in probabilities.items() if prob > 0.25]
        multi_intent = len(high_prob_categories) > 1

        # Determine if ambiguous (requires clarification)
        ambiguous = confidence < self.CONFIDENCE_THRESHOLDS['low'] or multi_intent
        requires_clarification = ambiguous

        # Build reasoning
        reasoning_parts = []
        if multi_intent:
            categories_str = ', '.join(high_prob_categories)
            reasoning_parts.append(f"Multiple intent categories detected: {categories_str}")

        if confidence >= self.CONFIDENCE_THRESHOLDS['high']:
            reasoning_parts.append(f"High confidence ({confidence:.2f}) in {primary_category} intent")
        elif confidence >= self.CONFIDENCE_THRESHOLDS['moderate']:
            reasoning_parts.append(f"Moderate confidence ({confidence:.2f}) in {primary_category} intent - proceed with assumption acknowledgment")
        else:
            reasoning_parts.append(f"Low confidence ({confidence:.2f}) - clarification recommended")

        # Identify strongest signals
        primary_signals = category_signals[primary_category]
        if primary_signals:
            top_signals = [signal for _, signal in primary_signals[:3]]
            reasoning_parts.append(f"Key signals: {', '.join(top_signals)}")

        reasoning = '. '.join(reasoning_parts)

        # Format all detected signals for output
        all_signals = []
        for category, signals in category_signals.items():
            if signals and probabilities[category] > 0.1:  # Only report non-trivial categories
                signal_strs = [f"{signal} ({signal_type})" for signal_type, signal in signals]
                all_signals.extend(signal_strs)

        return IntentAnalysis(
            primary_category=primary_category,
            confidence=confidence,
            categories=probabilities,
            signals=all_signals,
            multi_intent=multi_intent,
            ambiguous=ambiguous,
            requires_clarification=requires_clarification,
            reasoning=reasoning
        )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Classify user intent into categories')
    parser.add_argument('text', nargs='?', help='User request text to classify')
    parser.add_argument('--stdin', action='store_true', help='Read from stdin')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Get input text
    if args.stdin:
        text = sys.stdin.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)

    if not text:
        print(json.dumps({'error': 'Empty input text'}))
        sys.exit(1)

    # Classify intent
    classifier = IntentClassifier(verbose=args.verbose)
    result = classifier.classify(text)

    # Output JSON
    output = asdict(result)
    if args.verbose:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(json.dumps(output))


if __name__ == '__main__':
    main()
