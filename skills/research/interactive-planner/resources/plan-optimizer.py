#!/usr/bin/env python3
"""
Plan Optimizer - Intelligent Planning Workflow Optimization

Analyzes requirements specifications and suggests optimal question batches,
identifies redundant questions, optimizes question ordering for better UX,
and recommends follow-up questions based on gaps.

Usage:
    python plan-optimizer.py --spec requirements.json --output optimized-plan.yaml
"""

import json
import yaml
import argparse
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re


@dataclass
class QuestionDependency:
    """Represents dependency between questions."""
    question_id: str
    depends_on: List[str]
    reason: str


@dataclass
class OptimizationSuggestion:
    """Optimization suggestion for planning workflow."""
    suggestion_type: str  # reorder, remove, add, merge, split
    question_id: str
    reason: str
    confidence: float  # 0-1
    details: Dict[str, Any] = field(default_factory=dict)


class PlanOptimizer:
    """Optimize interactive planning workflows."""

    def __init__(self, spec_data: Dict[str, Any]):
        self.spec = spec_data
        self.dependencies: List[QuestionDependency] = []
        self.suggestions: List[OptimizationSuggestion] = []
        self.question_categories = defaultdict(list)
        self.answered_questions: Set[str] = set()
        self.missing_topics: List[str] = []

    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive analysis."""
        print("Analyzing requirements specification...")

        # Analyze existing answers
        self._analyze_existing_answers()

        # Identify missing topics
        self._identify_missing_topics()

        # Detect question dependencies
        self._detect_dependencies()

        # Generate optimization suggestions
        self._generate_suggestions()

        # Create optimized plan
        optimized_plan = self._create_optimized_plan()

        return optimized_plan

    def _analyze_existing_answers(self):
        """Analyze existing answered questions."""
        if 'answers' in self.spec and isinstance(self.spec['answers'], list):
            for answer in self.spec['answers']:
                question = answer.get('question', '').lower()
                self.answered_questions.add(question)

                # Categorize question
                category = self._categorize_question(question)
                self.question_categories[category].append(answer)

    def _categorize_question(self, question: str) -> str:
        """Categorize question by domain."""
        question_lower = question.lower()

        # Pattern matching for categories
        patterns = {
            'core_functionality': [
                r'purpose', r'goal', r'feature', r'functionality',
                r'what.*do', r'capability'
            ],
            'technical_architecture': [
                r'framework', r'stack', r'database', r'technology',
                r'backend', r'api', r'architecture'
            ],
            'user_experience': [
                r'user', r'interface', r'ui', r'ux', r'design',
                r'accessibility', r'interaction'
            ],
            'quality_scale': [
                r'test', r'quality', r'performance', r'scale',
                r'speed', r'reliability', r'coverage'
            ],
            'constraints_context': [
                r'timeline', r'budget', r'constraint', r'deadline',
                r'resource', r'limit', r'compliance'
            ]
        }

        for category, category_patterns in patterns.items():
            for pattern in category_patterns:
                if re.search(pattern, question_lower):
                    return category

        return 'other'

    def _identify_missing_topics(self):
        """Identify critical topics not yet covered."""
        # Essential topics for complete specification
        essential_topics = {
            'project_type': ['project type', 'application type', 'what kind'],
            'primary_goal': ['purpose', 'goal', 'objective'],
            'tech_stack': ['framework', 'technology', 'stack'],
            'database': ['database', 'storage', 'persistence'],
            'authentication': ['authentication', 'auth', 'login'],
            'testing': ['test', 'testing', 'coverage'],
            'deployment': ['deployment', 'deploy', 'hosting'],
            'timeline': ['timeline', 'deadline', 'schedule'],
            'quality_level': ['quality', 'standard', 'grade']
        }

        for topic, keywords in essential_topics.items():
            topic_covered = False

            for answered_q in self.answered_questions:
                if any(keyword in answered_q for keyword in keywords):
                    topic_covered = True
                    break

            if not topic_covered:
                self.missing_topics.append(topic)

    def _detect_dependencies(self):
        """Detect dependencies between questions."""
        # Example: Database questions depend on backend/stack questions
        dependency_rules = [
            {
                'question_pattern': r'database',
                'depends_on_patterns': [r'backend', r'stack', r'technology'],
                'reason': 'Database choice depends on backend technology'
            },
            {
                'question_pattern': r'deployment',
                'depends_on_patterns': [r'stack', r'framework'],
                'reason': 'Deployment strategy depends on tech stack'
            },
            {
                'question_pattern': r'authentication',
                'depends_on_patterns': [r'user', r'backend'],
                'reason': 'Auth method depends on user requirements and backend'
            },
            {
                'question_pattern': r'testing',
                'depends_on_patterns': [r'quality', r'complexity'],
                'reason': 'Testing approach depends on quality requirements'
            }
        ]

        for rule in dependency_rules:
            # Find questions matching this pattern
            matching_questions = []
            for answered_q in self.answered_questions:
                if re.search(rule['question_pattern'], answered_q):
                    matching_questions.append(answered_q)

            # Find dependencies
            for question in matching_questions:
                dependencies = []
                for dep_pattern in rule['depends_on_patterns']:
                    for answered_q in self.answered_questions:
                        if re.search(dep_pattern, answered_q):
                            dependencies.append(answered_q)

                if dependencies:
                    self.dependencies.append(QuestionDependency(
                        question_id=question,
                        depends_on=dependencies,
                        reason=rule['reason']
                    ))

    def _generate_suggestions(self):
        """Generate optimization suggestions."""
        # Suggest adding questions for missing topics
        for topic in self.missing_topics:
            self.suggestions.append(OptimizationSuggestion(
                suggestion_type='add',
                question_id=f'missing_{topic}',
                reason=f'Critical topic "{topic}" not covered',
                confidence=0.9,
                details={'topic': topic}
            ))

        # Suggest reordering based on dependencies
        for dep in self.dependencies:
            self.suggestions.append(OptimizationSuggestion(
                suggestion_type='reorder',
                question_id=dep.question_id,
                reason=f'Should be asked after: {", ".join(dep.depends_on)}',
                confidence=0.7,
                details={'depends_on': dep.depends_on}
            ))

        # Check for category imbalance
        category_counts = {cat: len(qs) for cat, qs in self.question_categories.items()}
        total_questions = sum(category_counts.values())

        for category, count in category_counts.items():
            if total_questions > 0:
                percentage = count / total_questions

                # Too few questions in important category
                if category in ['core_functionality', 'technical_architecture'] and percentage < 0.15:
                    self.suggestions.append(OptimizationSuggestion(
                        suggestion_type='add',
                        question_id=f'expand_{category}',
                        reason=f'Category "{category}" underrepresented ({percentage:.0%})',
                        confidence=0.8,
                        details={'category': category, 'current_count': count}
                    ))

                # Too many questions in one category
                if percentage > 0.4:
                    self.suggestions.append(OptimizationSuggestion(
                        suggestion_type='split',
                        question_id=f'split_{category}',
                        reason=f'Category "{category}" has too many questions ({percentage:.0%})',
                        confidence=0.6,
                        details={'category': category, 'current_count': count}
                    ))

    def _create_optimized_plan(self) -> Dict[str, Any]:
        """Create optimized planning workflow."""
        plan = {
            'analysis': {
                'total_questions_analyzed': len(self.answered_questions),
                'categories_covered': list(self.question_categories.keys()),
                'missing_topics': self.missing_topics,
                'dependencies_found': len(self.dependencies)
            },
            'suggestions': [
                {
                    'type': s.suggestion_type,
                    'question': s.question_id,
                    'reason': s.reason,
                    'confidence': s.confidence,
                    'details': s.details
                }
                for s in sorted(self.suggestions, key=lambda x: x.confidence, reverse=True)
            ],
            'recommended_batches': self._generate_recommended_batches(),
            'dependencies': [
                {
                    'question': d.question_id,
                    'depends_on': d.depends_on,
                    'reason': d.reason
                }
                for d in self.dependencies
            ]
        }

        return plan

    def _generate_recommended_batches(self) -> List[Dict[str, Any]]:
        """Generate recommended question batches."""
        batches = []

        # Batch 1: Foundational questions (if missing)
        foundational_topics = ['project_type', 'primary_goal', 'complexity']
        foundational_missing = [t for t in foundational_topics if t in self.missing_topics]

        if foundational_missing:
            batches.append({
                'batch_number': 1,
                'name': 'Foundational Questions',
                'topics': foundational_missing,
                'priority': 'critical',
                'questions_needed': min(4, len(foundational_missing))
            })

        # Batch 2: Technical architecture (if missing)
        tech_topics = ['tech_stack', 'database', 'backend']
        tech_missing = [t for t in tech_topics if t in self.missing_topics]

        if tech_missing:
            batches.append({
                'batch_number': 2,
                'name': 'Technical Architecture',
                'topics': tech_missing,
                'priority': 'high',
                'questions_needed': min(4, len(tech_missing))
            })

        # Batch 3: Quality & Testing (if missing)
        quality_topics = ['testing', 'quality_level', 'performance']
        quality_missing = [t for t in quality_topics if t in self.missing_topics]

        if quality_missing:
            batches.append({
                'batch_number': 3,
                'name': 'Quality & Scale',
                'topics': quality_missing,
                'priority': 'medium',
                'questions_needed': min(4, len(quality_missing))
            })

        # Batch 4: Constraints & Context (if missing)
        constraint_topics = ['timeline', 'deployment', 'authentication']
        constraint_missing = [t for t in constraint_topics if t in self.missing_topics]

        if constraint_missing:
            batches.append({
                'batch_number': 4,
                'name': 'Constraints & Context',
                'topics': constraint_missing,
                'priority': 'medium',
                'questions_needed': min(4, len(constraint_missing))
            })

        return batches

    def export_yaml(self, output_file: str):
        """Export optimized plan to YAML."""
        plan = self.analyze()
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(plan, f, default_flow_style=False, allow_unicode=True)

        print(f"Optimized plan exported to {output_file}")

    def export_json(self, output_file: str):
        """Export optimized plan to JSON."""
        plan = self.analyze()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        print(f"Optimized plan exported to {output_file}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Optimize interactive planning workflows"
    )
    parser.add_argument(
        '--spec',
        required=True,
        help='Path to requirements specification JSON file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output optimized plan file'
    )
    parser.add_argument(
        '--format',
        choices=['yaml', 'json'],
        default='yaml',
        help='Output format (default: yaml)'
    )

    args = parser.parse_args()

    # Load specification
    with open(args.spec, 'r', encoding='utf-8') as f:
        spec_data = json.load(f)

    # Optimize plan
    optimizer = PlanOptimizer(spec_data)

    # Export
    if args.format == 'yaml':
        optimizer.export_yaml(args.output)
    else:
        optimizer.export_json(args.output)

    print(f"\nAnalysis Summary:")
    print(f"  Missing topics: {len(optimizer.missing_topics)}")
    print(f"  Dependencies detected: {len(optimizer.dependencies)}")
    print(f"  Optimization suggestions: {len(optimizer.suggestions)}")


if __name__ == "__main__":
    main()
