#!/usr/bin/env python3
"""
Prompt Analyzer - Comprehensive prompt analysis and evaluation tool.

This script analyzes prompts across 6 dimensions:
1. Intent and Clarity Assessment
2. Structural Organization
3. Context Sufficiency
4. Technique Application
5. Failure Mode Detection
6. Formatting and Accessibility

Usage:
    python prompt-analyzer.py <prompt_file>
    python prompt-analyzer.py --text "Your prompt here"
    python prompt-analyzer.py --batch prompts/*.txt
"""

import re
import sys
import json
import argparse
from typing import Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter


@dataclass
class AnalysisResult:
    """Results from prompt analysis."""
    clarity_score: float
    structure_score: float
    context_score: float
    technique_score: float
    failure_score: float
    formatting_score: float
    overall_score: float
    recommendations: List[str]
    detected_patterns: List[str]
    anti_patterns: List[str]
    word_count: int
    sentence_count: int
    complexity_level: str


class PromptAnalyzer:
    """Analyze prompts using evidence-based evaluation framework."""

    # Common anti-patterns to detect
    ANTI_PATTERNS = {
        r'\b(quickly|fast|brief)\b': 'Vague urgency without clear criteria',
        r'\b(better|improve|enhance)\b(?! (?:by|through|via))': 'Vague improvement without specifics',
        r'\b(obviously|clearly|simply)\b': 'Assumption of shared understanding',
        r'\b(analyze|review|check)\b(?! (?:for|to|by))': 'Vague directive without goal',
        r'(?i)(do|make|create|build)(?! \w+ (?:that|which|to))': 'Incomplete instruction',
    }

    # Evidence-based techniques to detect
    TECHNIQUES = {
        'self_consistency': [
            r'validate|verify|cross-check|alternative\s+(?:perspective|interpretation)',
            r'consider\s+(?:multiple|different|various)\s+(?:perspectives|approaches)',
        ],
        'chain_of_thought': [
            r'step\s+by\s+step|explain\s+(?:your|the)\s+reasoning',
            r'think\s+through|show\s+(?:your|the)\s+thinking',
        ],
        'program_of_thought': [
            r'calculate|compute|solve\s+(?:step\s+by\s+step)?',
            r'show\s+(?:all\s+)?(?:intermediate\s+)?(?:steps|calculations)',
        ],
        'plan_and_solve': [
            r'first,?\s+(?:create|develop|plan)',
            r'then\s+execute|finally,?\s+verify',
        ],
        'few_shot': [
            r'(?:here\s+(?:are|is)\s+)?(?:examples?|demonstrations?)',
            r'input:.*output:|example\s+\d+:',
        ],
    }

    # Structural delimiters
    DELIMITERS = [
        r'```', r'<[^>]+>', r'---+', r'\*\*\*+',
        r'#{1,6}\s+', r'\d+\.\s+', r'[-*]\s+',
    ]

    def __init__(self):
        self.results = []

    def analyze(self, prompt: str) -> AnalysisResult:
        """Perform comprehensive prompt analysis."""
        # Calculate basic metrics
        word_count = len(prompt.split())
        sentence_count = len(re.findall(r'[.!?]+', prompt))

        # Analyze each dimension
        clarity = self._analyze_clarity(prompt)
        structure = self._analyze_structure(prompt)
        context = self._analyze_context(prompt)
        technique = self._analyze_techniques(prompt)
        failure = self._analyze_failure_modes(prompt)
        formatting = self._analyze_formatting(prompt)

        # Calculate overall score (weighted average)
        overall = (
            clarity * 0.25 +
            structure * 0.20 +
            context * 0.20 +
            technique * 0.15 +
            failure * 0.10 +
            formatting * 0.10
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            clarity, structure, context, technique, failure, formatting, prompt
        )

        # Detect patterns and anti-patterns
        detected_patterns = self._detect_patterns(prompt)
        anti_patterns = self._detect_anti_patterns(prompt)

        # Determine complexity level
        complexity = self._assess_complexity(word_count, sentence_count, prompt)

        return AnalysisResult(
            clarity_score=clarity,
            structure_score=structure,
            context_score=context,
            technique_score=technique,
            failure_score=failure,
            formatting_score=formatting,
            overall_score=overall,
            recommendations=recommendations,
            detected_patterns=detected_patterns,
            anti_patterns=anti_patterns,
            word_count=word_count,
            sentence_count=sentence_count,
            complexity_level=complexity,
        )

    def _analyze_clarity(self, prompt: str) -> float:
        """Evaluate intent and clarity."""
        score = 100.0

        # Check for action verbs at beginning
        first_sentence = prompt.split('.')[0].lower() if '.' in prompt else prompt.lower()
        action_verbs = ['analyze', 'create', 'build', 'implement', 'design', 'evaluate', 'generate']
        if not any(verb in first_sentence for verb in action_verbs):
            score -= 15

        # Check for success criteria
        if not re.search(r'(?:should|must|will|ensure|verify)\s+\w+', prompt):
            score -= 10

        # Check for ambiguous pronouns
        ambiguous_count = len(re.findall(r'\b(?:it|this|that|they)\b', prompt))
        score -= min(ambiguous_count * 2, 20)

        # Check for question marks (unclear directives)
        question_marks = prompt.count('?')
        if question_marks > 2:
            score -= 10

        return max(0.0, score)

    def _analyze_structure(self, prompt: str) -> float:
        """Evaluate structural organization."""
        score = 100.0

        # Check for hierarchical structure
        headers = len(re.findall(r'(?:^|\n)#{1,6}\s+', prompt))
        numbered_lists = len(re.findall(r'(?:^|\n)\d+\.\s+', prompt))
        bullet_lists = len(re.findall(r'(?:^|\n)[-*]\s+', prompt))

        if headers == 0 and numbered_lists == 0 and bullet_lists == 0:
            score -= 25

        # Check for delimiter usage
        delimiter_count = sum(
            len(re.findall(pattern, prompt)) for pattern in self.DELIMITERS
        )
        if delimiter_count == 0 and len(prompt.split()) > 100:
            score -= 20

        # Check for logical flow (transition words)
        transitions = len(re.findall(
            r'\b(?:first|second|then|next|finally|however|therefore|additionally)\b',
            prompt.lower()
        ))
        if transitions == 0 and len(prompt.split()) > 200:
            score -= 15

        return max(0.0, score)

    def _analyze_context(self, prompt: str) -> float:
        """Evaluate context sufficiency."""
        score = 100.0

        # Check for explicit constraints
        constraints = len(re.findall(
            r'\b(?:must|should|cannot|do not|ensure|require|constraint)\b',
            prompt.lower()
        ))
        if constraints == 0:
            score -= 20

        # Check for background context
        context_indicators = len(re.findall(
            r'\b(?:background|context|purpose|audience|goal|objective)\b',
            prompt.lower()
        ))
        if context_indicators == 0 and len(prompt.split()) > 100:
            score -= 15

        # Check for assumptions
        if 'assume' not in prompt.lower() and 'given' not in prompt.lower():
            score -= 10

        return max(0.0, score)

    def _analyze_techniques(self, prompt: str) -> float:
        """Evaluate evidence-based technique application."""
        score = 0.0
        techniques_found = []

        for technique, patterns in self.TECHNIQUES.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    techniques_found.append(technique)
                    score += 20
                    break

        # Cap at 100
        return min(100.0, score)

    def _analyze_failure_modes(self, prompt: str) -> float:
        """Detect common anti-patterns and failure modes."""
        score = 100.0

        for pattern, description in self.ANTI_PATTERNS.items():
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            if matches:
                score -= min(len(matches) * 5, 20)

        # Check for contradictory requirements
        if re.search(r'(?:comprehensive|detailed).*(?:brief|concise|short)', prompt, re.IGNORECASE):
            score -= 15

        # Check for edge case handling
        if 'edge case' not in prompt.lower() and 'if.*then' not in prompt.lower():
            score -= 10

        return max(0.0, score)

    def _analyze_formatting(self, prompt: str) -> float:
        """Evaluate formatting and accessibility."""
        score = 100.0

        # Check for excessive length without structure
        word_count = len(prompt.split())
        if word_count > 800:
            headers = len(re.findall(r'(?:^|\n)#{1,6}\s+', prompt))
            if headers < 3:
                score -= 20

        # Check for whitespace usage
        lines = prompt.split('\n')
        blank_lines = sum(1 for line in lines if line.strip() == '')
        if word_count > 200 and blank_lines == 0:
            score -= 15

        # Check for consistent delimiter usage
        delimiter_types = sum(
            1 for pattern in self.DELIMITERS
            if re.search(pattern, prompt)
        )
        if delimiter_types > 4:
            score -= 10  # Too many delimiter types

        return max(0.0, score)

    def _detect_patterns(self, prompt: str) -> List[str]:
        """Detect evidence-based patterns in use."""
        patterns = []
        for technique, pattern_list in self.TECHNIQUES.items():
            for pattern in pattern_list:
                if re.search(pattern, prompt, re.IGNORECASE):
                    patterns.append(technique)
                    break
        return list(set(patterns))

    def _detect_anti_patterns(self, prompt: str) -> List[str]:
        """Detect anti-patterns and issues."""
        issues = []
        for pattern, description in self.ANTI_PATTERNS.items():
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(description)
        return list(set(issues))

    def _generate_recommendations(
        self, clarity: float, structure: float, context: float,
        technique: float, failure: float, formatting: float, prompt: str
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if clarity < 70:
            recommendations.append(
                "Improve clarity: Start with specific action verbs and define success criteria explicitly."
            )

        if structure < 70:
            recommendations.append(
                "Add structure: Use headers, numbered lists, or delimiters to organize complex prompts."
            )

        if context < 70:
            recommendations.append(
                "Provide context: Add background, constraints, and explicit assumptions."
            )

        if technique < 40:
            recommendations.append(
                "Apply techniques: Consider self-consistency, chain-of-thought, or few-shot examples."
            )

        if failure < 70:
            recommendations.append(
                "Address edge cases: Specify handling for boundary conditions and potential failures."
            )

        if formatting < 70:
            recommendations.append(
                "Improve formatting: Add whitespace, consistent delimiters, and visual hierarchy."
            )

        return recommendations

    def _assess_complexity(self, word_count: int, sentence_count: int, prompt: str) -> str:
        """Assess prompt complexity level."""
        avg_words_per_sentence = word_count / max(sentence_count, 1)

        if word_count < 200:
            return "Simple"
        elif word_count < 500:
            return "Medium"
        elif word_count < 800:
            return "Complex"
        else:
            return "Very Complex"

    def analyze_file(self, filepath: Path) -> AnalysisResult:
        """Analyze prompt from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            prompt = f.read()
        return self.analyze(prompt)

    def analyze_batch(self, filepaths: List[Path]) -> List[Tuple[Path, AnalysisResult]]:
        """Analyze multiple prompts."""
        results = []
        for filepath in filepaths:
            result = self.analyze_file(filepath)
            results.append((filepath, result))
        return results

    def export_json(self, result: AnalysisResult, output_path: Path = None):
        """Export analysis results to JSON."""
        data = asdict(result)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        else:
            print(json.dumps(data, indent=2))


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze prompts for quality and effectiveness'
    )
    parser.add_argument('prompt', nargs='?', help='Prompt file to analyze')
    parser.add_argument('--text', help='Analyze text directly')
    parser.add_argument('--batch', nargs='+', help='Analyze multiple files')
    parser.add_argument('--json', action='store_true', help='Output JSON format')
    parser.add_argument('--output', help='Output file path')

    args = parser.parse_args()

    analyzer = PromptAnalyzer()

    # Determine analysis mode
    if args.text:
        result = analyzer.analyze(args.text)
        if args.json:
            analyzer.export_json(result, Path(args.output) if args.output else None)
        else:
            print_result(result, "Direct Input")

    elif args.batch:
        filepaths = [Path(p) for p in args.batch]
        results = analyzer.analyze_batch(filepaths)
        for filepath, result in results:
            print(f"\n{'='*60}")
            print(f"Analysis for: {filepath}")
            print('='*60)
            print_result(result, str(filepath))

    elif args.prompt:
        filepath = Path(args.prompt)
        result = analyzer.analyze_file(filepath)
        if args.json:
            analyzer.export_json(result, Path(args.output) if args.output else None)
        else:
            print_result(result, str(filepath))

    else:
        parser.print_help()
        sys.exit(1)


def print_result(result: AnalysisResult, source: str):
    """Print analysis results in human-readable format."""
    print(f"\n📊 Prompt Analysis Results for: {source}")
    print(f"{'='*60}\n")

    # Scores
    print("📈 Dimension Scores:")
    print(f"  Clarity & Intent:    {result.clarity_score:5.1f}/100")
    print(f"  Structure:           {result.structure_score:5.1f}/100")
    print(f"  Context:             {result.context_score:5.1f}/100")
    print(f"  Techniques:          {result.technique_score:5.1f}/100")
    print(f"  Failure Handling:    {result.failure_score:5.1f}/100")
    print(f"  Formatting:          {result.formatting_score:5.1f}/100")
    print(f"\n  ⭐ Overall Score:     {result.overall_score:5.1f}/100")

    # Metrics
    print(f"\n📏 Metrics:")
    print(f"  Word Count:          {result.word_count}")
    print(f"  Sentence Count:      {result.sentence_count}")
    print(f"  Complexity Level:    {result.complexity_level}")

    # Patterns
    if result.detected_patterns:
        print(f"\n✅ Detected Patterns:")
        for pattern in result.detected_patterns:
            print(f"  • {pattern}")

    # Anti-patterns
    if result.anti_patterns:
        print(f"\n⚠️  Anti-Patterns:")
        for pattern in result.anti_patterns:
            print(f"  • {pattern}")

    # Recommendations
    if result.recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
