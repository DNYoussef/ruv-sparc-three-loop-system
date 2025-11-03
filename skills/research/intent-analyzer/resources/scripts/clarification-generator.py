#!/usr/bin/env python3
"""
Clarification Generator - Strategic Socratic Questioning

Generates strategic clarifying questions for ambiguous user requests using
Socratic questioning techniques. Produces targeted questions designed to
disambiguate intent, reveal constraints, gather context, and validate assumptions.

Usage:
    python clarification-generator.py --type disambiguation --interpretations "A,B,C"
    python clarification-generator.py --type constraint_revelation --domain technical
    python clarification-generator.py --type context_gathering
    python clarification-generator.py --type assumption_validation --assumption "user wants production code"

Question Types:
    - disambiguation: Choose between competing interpretations
    - constraint_revelation: Surface unstated constraints
    - context_gathering: Build essential understanding
    - assumption_validation: Verify implicit assumptions

Output: JSON with generated questions and usage guidance
"""

import sys
import json
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class ClarificationQuestion:
    """Structured clarification question"""
    question: str
    type: str
    purpose: str
    alternatives: Optional[List[str]] = None
    follow_up_guidance: Optional[str] = None


class ClarificationGenerator:
    """Generate strategic Socratic questions for intent clarification"""

    # Disambiguation question templates
    DISAMBIGUATION_TEMPLATES = [
        "Are you looking to {option_a} or {option_b}?",
        "Is your goal {goal_a} or {goal_b}?",
        "Would you prefer {approach_a} or {approach_b}?",
        "Do you need help {task_a} or {task_b}?",
        "Are you trying to {intent_a} or {intent_b}?",
    ]

    # Constraint revelation questions by domain
    CONSTRAINT_QUESTIONS = {
        'purpose': [
            "What will you use this for?",
            "What problem are you trying to solve?",
            "What's the end goal of this task?",
        ],
        'audience': [
            "Who is the audience for this?",
            "Who will be using/reading this?",
            "Is this for technical or non-technical users?",
        ],
        'experience': [
            "What have you tried already?",
            "What's your experience level with this?",
            "Are you familiar with [technology/concept]?",
        ],
        'timeline': [
            "What's your timeline for this?",
            "Is this time-sensitive?",
            "When do you need this by?",
        ],
        'technical': [
            "Are there specific technologies you need to use?",
            "What's your current tech stack?",
            "Do you have any technical constraints?",
        ],
        'resources': [
            "What resources do you have available?",
            "Are there budget or licensing constraints?",
            "Do you have access to [tool/service]?",
        ],
        'quality': [
            "Is this for production use or learning?",
            "How important is [performance/security/maintainability]?",
            "What quality standards need to be met?",
        ]
    }

    # Context gathering questions
    CONTEXT_TEMPLATES = [
        "What's the broader context for this request?",
        "Can you tell me more about your situation?",
        "What led you to this question?",
        "Are there specific requirements or constraints I should know about?",
        "What would success look like for you?",
        "Is there important background I should understand?",
    ]

    # Assumption validation templates
    ASSUMPTION_TEMPLATES = [
        "I'm assuming {assumption}. Is that correct?",
        "Should I focus on {aspect_a} or is {aspect_b} more important?",
        "I'm interpreting this as {interpretation}. Does that match what you meant?",
        "Am I right in thinking you need {inferred_need}?",
        "It seems like {observation}. Is that accurate?",
    ]

    def __init__(self, max_questions: int = 3):
        self.max_questions = max_questions

    def generate_disambiguation(self, interpretations: List[str]) -> List[ClarificationQuestion]:
        """Generate questions to disambiguate between interpretations"""
        questions = []

        if len(interpretations) < 2:
            return questions

        # Binary comparison for 2 interpretations
        if len(interpretations) == 2:
            template = self.DISAMBIGUATION_TEMPLATES[0]
            question_text = template.format(
                option_a=interpretations[0],
                option_b=interpretations[1]
            )

            questions.append(ClarificationQuestion(
                question=question_text,
                type='disambiguation',
                purpose=f'Distinguish between: {interpretations[0]} vs {interpretations[1]}',
                alternatives=interpretations,
                follow_up_guidance='Use response to select dominant interpretation'
            ))

        # Multiple choice for 3+ interpretations
        else:
            question_text = "Which of these best describes what you're looking for?"

            questions.append(ClarificationQuestion(
                question=question_text,
                type='disambiguation',
                purpose=f'Select from {len(interpretations)} interpretations',
                alternatives=interpretations,
                follow_up_guidance='Present numbered options: ' + ', '.join(f'{i+1}. {interp}' for i, interp in enumerate(interpretations))
            ))

        return questions

    def generate_constraint_revelation(self, domains: List[str] = None) -> List[ClarificationQuestion]:
        """Generate questions to reveal unstated constraints"""
        questions = []

        # Default to common high-value domains
        if domains is None:
            domains = ['purpose', 'audience', 'timeline']

        for domain in domains[:self.max_questions]:
            if domain in self.CONSTRAINT_QUESTIONS:
                # Pick most appropriate question from domain
                question_text = self.CONSTRAINT_QUESTIONS[domain][0]

                questions.append(ClarificationQuestion(
                    question=question_text,
                    type='constraint_revelation',
                    purpose=f'Reveal {domain} constraints',
                    follow_up_guidance=f'Use response to understand {domain} requirements'
                ))

        return questions

    def generate_context_gathering(self, focus_areas: List[str] = None) -> List[ClarificationQuestion]:
        """Generate questions to gather essential context"""
        questions = []

        # Default to general context questions
        if focus_areas is None:
            question_texts = self.CONTEXT_TEMPLATES[:self.max_questions]
        else:
            # Customize based on focus areas
            question_texts = []
            for area in focus_areas:
                if area == 'background':
                    question_texts.append("What's the broader context for this request?")
                elif area == 'requirements':
                    question_texts.append("Are there specific requirements or constraints I should know about?")
                elif area == 'success_criteria':
                    question_texts.append("What would success look like for you?")

        for question_text in question_texts[:self.max_questions]:
            questions.append(ClarificationQuestion(
                question=question_text,
                type='context_gathering',
                purpose='Build essential understanding of situation',
                follow_up_guidance='Use response to inform overall approach'
            ))

        return questions

    def generate_assumption_validation(self, assumptions: List[str]) -> List[ClarificationQuestion]:
        """Generate questions to validate implicit assumptions"""
        questions = []

        for assumption in assumptions[:self.max_questions]:
            # Use first template by default
            template = self.ASSUMPTION_TEMPLATES[0]
            question_text = template.format(assumption=assumption)

            questions.append(ClarificationQuestion(
                question=question_text,
                type='assumption_validation',
                purpose=f'Validate assumption: {assumption}',
                follow_up_guidance='If incorrect, adjust interpretation accordingly'
            ))

        return questions

    def generate_adaptive(self,
                         interpretations: List[str] = None,
                         constraints_needed: List[str] = None,
                         assumptions: List[str] = None) -> List[ClarificationQuestion]:
        """Generate adaptive mix of questions based on needs"""
        all_questions = []

        # Prioritize disambiguation if multiple interpretations
        if interpretations and len(interpretations) >= 2:
            all_questions.extend(self.generate_disambiguation(interpretations))

        # Add constraint questions if needed
        if constraints_needed:
            all_questions.extend(self.generate_constraint_revelation(constraints_needed))

        # Add assumption validation if needed
        if assumptions:
            all_questions.extend(self.generate_assumption_validation(assumptions))

        # Limit to max_questions
        return all_questions[:self.max_questions]


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate strategic clarifying questions'
    )

    parser.add_argument(
        '--type',
        choices=['disambiguation', 'constraint_revelation', 'context_gathering',
                'assumption_validation', 'adaptive'],
        default='adaptive',
        help='Type of clarification questions to generate'
    )

    parser.add_argument(
        '--interpretations',
        type=str,
        help='Comma-separated list of possible interpretations (for disambiguation)'
    )

    parser.add_argument(
        '--domains',
        type=str,
        help='Comma-separated constraint domains: purpose, audience, timeline, technical, resources, quality'
    )

    parser.add_argument(
        '--assumptions',
        type=str,
        help='Comma-separated list of assumptions to validate'
    )

    parser.add_argument(
        '--max-questions',
        type=int,
        default=3,
        help='Maximum number of questions to generate (default: 3)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output with guidance'
    )

    args = parser.parse_args()

    # Create generator
    generator = ClarificationGenerator(max_questions=args.max_questions)

    # Generate questions based on type
    questions = []

    if args.type == 'disambiguation':
        if not args.interpretations:
            print(json.dumps({'error': 'Disambiguation requires --interpretations'}))
            sys.exit(1)
        interpretations = [i.strip() for i in args.interpretations.split(',')]
        questions = generator.generate_disambiguation(interpretations)

    elif args.type == 'constraint_revelation':
        domains = None
        if args.domains:
            domains = [d.strip() for d in args.domains.split(',')]
        questions = generator.generate_constraint_revelation(domains)

    elif args.type == 'context_gathering':
        questions = generator.generate_context_gathering()

    elif args.type == 'assumption_validation':
        if not args.assumptions:
            print(json.dumps({'error': 'Assumption validation requires --assumptions'}))
            sys.exit(1)
        assumptions = [a.strip() for a in args.assumptions.split(',')]
        questions = generator.generate_assumption_validation(assumptions)

    elif args.type == 'adaptive':
        interpretations = None
        if args.interpretations:
            interpretations = [i.strip() for i in args.interpretations.split(',')]

        domains = None
        if args.domains:
            domains = [d.strip() for d in args.domains.split(',')]

        assumptions = None
        if args.assumptions:
            assumptions = [a.strip() for a in args.assumptions.split(',')]

        questions = generator.generate_adaptive(
            interpretations=interpretations,
            constraints_needed=domains,
            assumptions=assumptions
        )

    # Format output
    output = {
        'questions': [asdict(q) for q in questions],
        'count': len(questions),
        'type': args.type
    }

    if args.verbose:
        print(json.dumps(output, indent=2))
    else:
        print(json.dumps(output))


if __name__ == '__main__':
    main()
