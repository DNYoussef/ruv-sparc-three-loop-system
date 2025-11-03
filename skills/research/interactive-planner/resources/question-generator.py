#!/usr/bin/env python3
"""
Interactive Question Generator - Advanced Requirements Elicitation
Generates structured multi-select questions for comprehensive project planning.
Supports dynamic question adaptation based on project complexity and domain.
"""

import json
import yaml
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class QuestionCategory(Enum):
    """Categories for organizing questions by domain."""
    CORE_FUNCTIONALITY = "core_functionality"
    TECHNICAL_ARCHITECTURE = "technical_architecture"
    USER_EXPERIENCE = "user_experience"
    QUALITY_SCALE = "quality_scale"
    CONSTRAINTS_CONTEXT = "constraints_context"


class ComplexityLevel(Enum):
    """Project complexity levels affecting question depth."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    LARGE_SCALE = "large_scale"


@dataclass
class QuestionOption:
    """Individual option within a question."""
    label: str
    description: str

    def validate(self) -> bool:
        """Validate option has required fields."""
        return bool(self.label and self.description and len(self.label) <= 50)


@dataclass
class Question:
    """Structured question with multi-select support."""
    question: str
    header: str  # Max 12 chars for UI
    options: List[QuestionOption]
    multiSelect: bool = False
    category: Optional[QuestionCategory] = None

    def validate(self) -> bool:
        """Validate question structure."""
        if not self.question or not self.header:
            return False
        if len(self.header) > 12:
            return False
        if not (2 <= len(self.options) <= 4):
            return False
        return all(opt.validate() for opt in self.options)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/YAML output."""
        return {
            "question": self.question,
            "header": self.header,
            "multiSelect": self.multiSelect,
            "options": [asdict(opt) for opt in self.options]
        }


@dataclass
class QuestionBatch:
    """Batch of up to 4 questions for AskUserQuestion tool."""
    questions: List[Question] = field(default_factory=list)
    batch_number: int = 1
    category: Optional[QuestionCategory] = None

    def add_question(self, question: Question) -> bool:
        """Add question if batch not full."""
        if len(self.questions) >= 4:
            return False
        if not question.validate():
            return False
        self.questions.append(question)
        return True

    def is_full(self) -> bool:
        """Check if batch has 4 questions."""
        return len(self.questions) >= 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary."""
        return {
            "batch_number": self.batch_number,
            "category": self.category.value if self.category else None,
            "questions": [q.to_dict() for q in self.questions]
        }


class QuestionGenerator:
    """Generate question batches based on project parameters."""

    def __init__(self, project_type: str, complexity: ComplexityLevel):
        self.project_type = project_type.lower()
        self.complexity = complexity
        self.batches: List[QuestionBatch] = []

    def generate_core_functionality_questions(self) -> List[Question]:
        """Generate questions about core functionality."""
        questions = [
            Question(
                question="What is the primary purpose of this project?",
                header="Purpose",
                multiSelect=False,
                category=QuestionCategory.CORE_FUNCTIONALITY,
                options=[
                    QuestionOption("New feature", "Add new functionality to existing system"),
                    QuestionOption("Refactoring", "Improve existing code structure/quality"),
                    QuestionOption("Bug fix", "Fix existing defect or issue"),
                    QuestionOption("Performance", "Optimize speed/efficiency/resource usage")
                ]
            ),
            Question(
                question="Which key features are needed? (Select all that apply)",
                header="Features",
                multiSelect=True,
                category=QuestionCategory.CORE_FUNCTIONALITY,
                options=[
                    QuestionOption("User management", "Authentication, authorization, profiles"),
                    QuestionOption("Data processing", "ETL, transformations, analytics"),
                    QuestionOption("Real-time updates", "WebSockets, SSE, live data"),
                    QuestionOption("File handling", "Upload, download, storage, processing")
                ]
            )
        ]

        # Add project-type specific questions
        if self.project_type in ["web", "mobile"]:
            questions.append(
                Question(
                    question="What user actions should be supported?",
                    header="Actions",
                    multiSelect=True,
                    category=QuestionCategory.CORE_FUNCTIONALITY,
                    options=[
                        QuestionOption("CRUD operations", "Create, Read, Update, Delete data"),
                        QuestionOption("Search/filter", "Query and filter large datasets"),
                        QuestionOption("Collaboration", "Multi-user interaction and sharing"),
                        QuestionOption("Analytics", "Dashboards, reports, visualizations")
                    ]
                )
            )

        return questions

    def generate_technical_architecture_questions(self) -> List[Question]:
        """Generate questions about technical architecture."""
        questions = [
            Question(
                question="What technology stack should we use?",
                header="Tech Stack",
                multiSelect=False,
                category=QuestionCategory.TECHNICAL_ARCHITECTURE,
                options=[
                    QuestionOption("React/Next.js", "React 18+ with Next.js framework"),
                    QuestionOption("Vue/Nuxt", "Vue 3 with Nuxt.js framework"),
                    QuestionOption("Node.js/Express", "Backend API with Express.js"),
                    QuestionOption("Python/FastAPI", "Modern Python async API framework")
                ]
            ),
            Question(
                question="What database type is appropriate?",
                header="Database",
                multiSelect=False,
                category=QuestionCategory.TECHNICAL_ARCHITECTURE,
                options=[
                    QuestionOption("PostgreSQL", "Relational DB with advanced features"),
                    QuestionOption("MongoDB", "Document-oriented NoSQL database"),
                    QuestionOption("Redis", "In-memory cache and data structure store"),
                    QuestionOption("Firebase", "Managed cloud database with real-time sync")
                ]
            ),
            Question(
                question="Which backend patterns are needed? (Select all)",
                header="Backend",
                multiSelect=True,
                category=QuestionCategory.TECHNICAL_ARCHITECTURE,
                options=[
                    QuestionOption("REST API", "RESTful HTTP API endpoints"),
                    QuestionOption("GraphQL", "GraphQL API with flexible queries"),
                    QuestionOption("WebSockets", "Bi-directional real-time communication"),
                    QuestionOption("Message Queue", "Async task processing with queues")
                ]
            )
        ]

        return questions

    def generate_quality_scale_questions(self) -> List[Question]:
        """Generate questions about quality and scale requirements."""
        return [
            Question(
                question="What testing coverage is required? (Select all)",
                header="Testing",
                multiSelect=True,
                category=QuestionCategory.QUALITY_SCALE,
                options=[
                    QuestionOption("Unit tests", "Component/function level tests (Jest/Vitest)"),
                    QuestionOption("Integration", "API and service integration tests"),
                    QuestionOption("E2E tests", "End-to-end user workflow tests (Playwright)"),
                    QuestionOption("Performance", "Load testing and benchmarking")
                ]
            ),
            Question(
                question="What quality level is expected?",
                header="Quality",
                multiSelect=False,
                category=QuestionCategory.QUALITY_SCALE,
                options=[
                    QuestionOption("Quick prototype", "Fast MVP with minimal quality checks"),
                    QuestionOption("Production MVP", "Solid MVP with essential quality measures"),
                    QuestionOption("Enterprise grade", "High quality with comprehensive testing"),
                    QuestionOption("Research/experimental", "Exploratory with focus on learning")
                ]
            ),
            Question(
                question="What scalability requirements exist?",
                header="Scale",
                multiSelect=False,
                category=QuestionCategory.QUALITY_SCALE,
                options=[
                    QuestionOption("Single user", "Personal use or small team (<10 users)"),
                    QuestionOption("Small scale", "Hundreds of users, modest traffic"),
                    QuestionOption("Medium scale", "Thousands of users, moderate traffic"),
                    QuestionOption("Large scale", "Millions of users, high traffic/data")
                ]
            )
        ]

    def generate_batches(self, num_batches: int = 5) -> List[QuestionBatch]:
        """Generate specified number of question batches."""
        all_questions: List[Question] = []

        # Generate questions by category
        all_questions.extend(self.generate_core_functionality_questions())
        all_questions.extend(self.generate_technical_architecture_questions())
        all_questions.extend(self.generate_quality_scale_questions())

        # Organize into batches of 4
        self.batches = []
        current_batch = QuestionBatch(batch_number=1)

        for idx, question in enumerate(all_questions):
            if current_batch.is_full():
                self.batches.append(current_batch)
                current_batch = QuestionBatch(batch_number=len(self.batches) + 1)

            current_batch.add_question(question)
            current_batch.category = question.category

        # Add final batch if not empty
        if current_batch.questions:
            self.batches.append(current_batch)

        return self.batches[:num_batches]

    def export_json(self, output_file: str):
        """Export batches to JSON file."""
        data = {
            "project_type": self.project_type,
            "complexity": self.complexity.value,
            "total_batches": len(self.batches),
            "batches": [batch.to_dict() for batch in self.batches]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def export_yaml(self, output_file: str):
        """Export batches to YAML file."""
        data = {
            "project_type": self.project_type,
            "complexity": self.complexity.value,
            "total_batches": len(self.batches),
            "batches": [batch.to_dict() for batch in self.batches]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate structured questions for interactive planning"
    )
    parser.add_argument(
        "--project-type",
        required=True,
        choices=["web", "mobile", "api", "library", "cli"],
        help="Type of project being planned"
    )
    parser.add_argument(
        "--complexity",
        required=True,
        choices=["simple", "moderate", "complex", "large_scale"],
        help="Project complexity level"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=5,
        help="Number of question batches to generate (default: 5)"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "yaml"],
        default="yaml",
        help="Output file format"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to output file"
    )

    args = parser.parse_args()

    # Map complexity string to enum
    complexity_map = {
        "simple": ComplexityLevel.SIMPLE,
        "moderate": ComplexityLevel.MODERATE,
        "complex": ComplexityLevel.COMPLEX,
        "large_scale": ComplexityLevel.LARGE_SCALE
    }

    # Generate questions
    generator = QuestionGenerator(
        project_type=args.project_type,
        complexity=complexity_map[args.complexity]
    )

    batches = generator.generate_batches(args.num_batches)

    print(f"Generated {len(batches)} batches with {sum(len(b.questions) for b in batches)} total questions")

    # Export to file
    if args.output_format == "json":
        generator.export_json(args.output_file)
    else:
        generator.export_yaml(args.output_file)

    print(f"Exported to {args.output_file}")


if __name__ == "__main__":
    main()
