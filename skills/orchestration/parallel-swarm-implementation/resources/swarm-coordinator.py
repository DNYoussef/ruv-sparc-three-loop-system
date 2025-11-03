#!/usr/bin/env python3
"""
Swarm Coordinator - Dynamic Agent+Skill Assignment Matrix Generator
Part of Loop 2: Parallel Swarm Implementation (Enhanced Tier)

This script implements the "Queen Coordinator" logic for analyzing Loop 1 planning
packages and dynamically generating executable agent+skill assignment matrices.
"""

import json
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class TaskType(Enum):
    """Task categorization for agent selection"""
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    TEST = "test"
    QUALITY = "quality"
    DOCS = "docs"
    INFRASTRUCTURE = "infrastructure"


class Complexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"      # 1 agent
    MODERATE = "moderate"  # 2-3 agents
    COMPLEX = "complex"    # 4+ agents


class Priority(Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """Task assignment structure"""
    taskId: str
    description: str
    taskType: TaskType
    complexity: Complexity
    assignedAgent: str
    useSkill: Optional[str]
    customInstructions: str
    priority: Priority
    dependencies: List[str]
    loop1_research: str
    loop1_risk_mitigation: str


@dataclass
class ParallelGroup:
    """Parallel execution group"""
    group: int
    tasks: List[str]
    reason: str


@dataclass
class Statistics:
    """Assignment matrix statistics"""
    totalTasks: int
    skillBasedAgents: int
    customInstructionAgents: int
    uniqueAgents: int
    estimatedParallelism: str


@dataclass
class AgentSkillMatrix:
    """Complete agent+skill assignment matrix"""
    project: str
    loop1_package: str
    tasks: List[Task]
    parallelGroups: List[ParallelGroup]
    statistics: Statistics


class SkillRegistry:
    """Registry of available skills for agent assignment"""

    AVAILABLE_SKILLS = {
        "test": ["tdd-london-swarm", "testing-quality"],
        "quality": ["theater-detection-audit", "functionality-audit", "code-review-assistant"],
        "docs": ["api-docs", "documentation"],
        "database": ["database-schema-design"],
        "api": ["api-development"],
    }

    @classmethod
    def get_skill_for_task(cls, task_type: str, description: str) -> Optional[str]:
        """
        Determine if a specialized skill exists for this task type.

        Args:
            task_type: Type of task (test, quality, docs, etc.)
            description: Task description for context

        Returns:
            Skill name if available, None if custom instructions needed
        """
        skills = cls.AVAILABLE_SKILLS.get(task_type.lower())
        if not skills:
            return None

        # Simple keyword matching - can be enhanced with NLP
        description_lower = description.lower()

        if "tdd" in description_lower or "mock" in description_lower:
            return "tdd-london-swarm"
        elif "theater" in description_lower or "genuine" in description_lower:
            return "theater-detection-audit"
        elif "sandbox" in description_lower or "validate" in description_lower:
            return "functionality-audit"
        elif "review" in description_lower and "quality" in description_lower:
            return "code-review-assistant"
        elif "api" in description_lower and "docs" in description_lower:
            return "api-docs"
        elif "database" in description_lower or "schema" in description_lower:
            return "database-schema-design"

        return None


class AgentRegistry:
    """Registry of 86+ agents with specialization mapping"""

    AGENT_MAPPING = {
        TaskType.BACKEND: ["backend-dev", "system-architect", "coder"],
        TaskType.FRONTEND: ["react-developer", "frontend-dev", "coder"],
        TaskType.DATABASE: ["database-design-specialist", "code-analyzer"],
        TaskType.TEST: ["tester", "tdd-london-swarm"],
        TaskType.QUALITY: ["theater-detection-audit", "functionality-audit", "reviewer"],
        TaskType.DOCS: ["api-docs", "docs-writer", "technical-writing-agent"],
        TaskType.INFRASTRUCTURE: ["cicd-engineer", "system-architect"],
    }

    @classmethod
    def select_agent(cls, task_type: TaskType, complexity: Complexity) -> str:
        """
        Select optimal agent for task based on type and complexity.

        Args:
            task_type: Type of task
            complexity: Complexity level

        Returns:
            Agent identifier from 86-agent registry
        """
        agents = cls.AGENT_MAPPING.get(task_type, ["coder"])

        # For complex tasks, prefer specialized agents
        if complexity == Complexity.COMPLEX and len(agents) > 1:
            return agents[0]  # Most specialized

        # For simple tasks, prefer generalists
        if complexity == Complexity.SIMPLE and len(agents) > 2:
            return agents[-1]  # Most general

        # Default to first (most specialized)
        return agents[0]


class SwarmCoordinator:
    """Queen Coordinator - Meta-orchestration for Loop 2"""

    def __init__(self, loop1_package_path: str):
        """
        Initialize coordinator with Loop 1 planning package.

        Args:
            loop1_package_path: Path to loop1-planning-package.json
        """
        self.loop1_package_path = Path(loop1_package_path)
        self.loop1_data: Dict[str, Any] = {}
        self.tasks: List[Task] = []
        self.parallel_groups: List[ParallelGroup] = []

    def load_loop1_package(self) -> None:
        """Load and parse Loop 1 planning package"""
        if not self.loop1_package_path.exists():
            raise FileNotFoundError(f"Loop 1 package not found: {self.loop1_package_path}")

        with open(self.loop1_package_path, 'r') as f:
            self.loop1_data = json.load(f)

        print(f"✅ Loaded Loop 1 package: {self.loop1_data.get('project', 'Unknown')}")

    def analyze_tasks(self) -> None:
        """
        Analyze Loop 1 plan and create task assignments.

        This implements PHASE 2-4 of Queen's Meta-Analysis SOP:
        - Task analysis
        - Agent selection
        - Skill assignment
        """
        planning = self.loop1_data.get('planning', {})
        enhanced_plan = planning.get('enhanced_plan', {})

        task_counter = 1

        for phase_name, phase_tasks in enhanced_plan.items():
            if not isinstance(phase_tasks, list):
                continue

            for task_desc in phase_tasks:
                task = self._create_task_assignment(
                    task_id=f"task-{task_counter:03d}",
                    description=task_desc,
                    phase=phase_name
                )
                self.tasks.append(task)
                task_counter += 1

        print(f"✅ Analyzed {len(self.tasks)} tasks from Loop 1 plan")

    def _create_task_assignment(self, task_id: str, description: str, phase: str) -> Task:
        """
        Create task assignment with agent and skill selection.

        Args:
            task_id: Unique task identifier
            description: Task description from Loop 1
            phase: Planning phase (foundation, implementation, quality, etc.)

        Returns:
            Complete Task assignment
        """
        # Classify task type
        task_type = self._classify_task_type(description, phase)

        # Determine complexity
        complexity = self._assess_complexity(description)

        # Select optimal agent
        agent = AgentRegistry.select_agent(task_type, complexity)

        # Check for available skill
        skill = SkillRegistry.get_skill_for_task(task_type.value, description)

        # Generate instructions
        if skill:
            custom_instructions = self._generate_skill_context(description, skill)
        else:
            custom_instructions = self._generate_custom_instructions(description, task_type)

        # Extract Loop 1 research and risk mitigation
        research = self._extract_research(task_type)
        risk_mitigation = self._extract_risk_mitigation(task_type)

        # Determine priority
        priority = self._determine_priority(phase, task_type)

        return Task(
            taskId=task_id,
            description=description,
            taskType=task_type,
            complexity=complexity,
            assignedAgent=agent,
            useSkill=skill,
            customInstructions=custom_instructions,
            priority=priority,
            dependencies=[],  # Will be filled in optimize_parallel_groups
            loop1_research=research,
            loop1_risk_mitigation=risk_mitigation
        )

    def _classify_task_type(self, description: str, phase: str) -> TaskType:
        """Classify task based on description and phase"""
        desc_lower = description.lower()

        if "test" in desc_lower or "junit" in desc_lower or "jest" in desc_lower:
            return TaskType.TEST
        elif "api" in desc_lower or "endpoint" in desc_lower or "backend" in desc_lower:
            return TaskType.BACKEND
        elif "ui" in desc_lower or "frontend" in desc_lower or "react" in desc_lower:
            return TaskType.FRONTEND
        elif "database" in desc_lower or "schema" in desc_lower or "sql" in desc_lower:
            return TaskType.DATABASE
        elif "quality" in phase.lower() or "review" in desc_lower or "audit" in desc_lower:
            return TaskType.QUALITY
        elif "docs" in desc_lower or "documentation" in desc_lower:
            return TaskType.DOCS
        elif "docker" in desc_lower or "ci/cd" in desc_lower or "deploy" in desc_lower:
            return TaskType.INFRASTRUCTURE

        # Default to backend for ambiguous tasks
        return TaskType.BACKEND

    def _assess_complexity(self, description: str) -> Complexity:
        """Assess task complexity based on description"""
        desc_lower = description.lower()

        # Complex indicators
        if any(word in desc_lower for word in ["multi", "complex", "distributed", "integrate"]):
            return Complexity.COMPLEX

        # Simple indicators
        if any(word in desc_lower for word in ["simple", "basic", "single", "helper"]):
            return Complexity.SIMPLE

        # Default to moderate
        return Complexity.MODERATE

    def _generate_skill_context(self, description: str, skill: str) -> str:
        """Generate context for skill-based execution"""
        return (
            f"Apply {skill} skill to: {description}\n\n"
            f"Follow skill SOP with these contextual parameters:\n"
            f"- Target coverage: ≥90%\n"
            f"- Zero tolerance for theater\n"
            f"- Coordinate via hooks: pre-task, post-edit, post-task"
        )

    def _generate_custom_instructions(self, description: str, task_type: TaskType) -> str:
        """Generate detailed custom instructions when no skill available"""
        return (
            f"Task: {description}\n\n"
            f"Detailed Implementation Instructions:\n"
            f"1. Load Loop 1 planning context from memory\n"
            f"2. Apply research recommendations from Loop 1\n"
            f"3. Implement with defense-in-depth per risk mitigations\n"
            f"4. Store artifacts in appropriate directories (not root)\n"
            f"5. Use hooks for progress tracking and coordination\n"
            f"6. Validate against Loop 1 requirements\n\n"
            f"Coordination:\n"
            f"- Pre-task: npx claude-flow@alpha hooks pre-task\n"
            f"- Post-edit: npx claude-flow@alpha hooks post-edit --file <file>\n"
            f"- Post-task: npx claude-flow@alpha hooks post-task"
        )

    def _extract_research(self, task_type: TaskType) -> str:
        """Extract relevant research from Loop 1"""
        research = self.loop1_data.get('research', {})
        # Simplified - would do more sophisticated extraction in production
        return json.dumps(research, indent=2)[:200] + "..."

    def _extract_risk_mitigation(self, task_type: TaskType) -> str:
        """Extract relevant risk mitigations from Loop 1"""
        risk_analysis = self.loop1_data.get('risk_analysis', {})
        # Simplified - would do more sophisticated extraction in production
        return json.dumps(risk_analysis, indent=2)[:200] + "..."

    def _determine_priority(self, phase: str, task_type: TaskType) -> Priority:
        """Determine task priority based on phase and type"""
        if "foundation" in phase.lower() or task_type == TaskType.DATABASE:
            return Priority.CRITICAL
        elif task_type in [TaskType.BACKEND, TaskType.TEST]:
            return Priority.HIGH
        elif task_type in [TaskType.QUALITY, TaskType.FRONTEND]:
            return Priority.MEDIUM
        else:
            return Priority.LOW

    def optimize_parallel_groups(self) -> None:
        """
        Optimize tasks into parallel execution groups.

        This implements PHASE 6 of Queen's Meta-Analysis SOP:
        - Identify independent tasks
        - Group dependent tasks
        - Balance agent workload
        - Identify critical path
        """
        # Group 1: Foundation (critical path starters)
        foundation_tasks = [
            t.taskId for t in self.tasks
            if t.priority == Priority.CRITICAL or t.taskType == TaskType.DATABASE
        ]

        if foundation_tasks:
            self.parallel_groups.append(ParallelGroup(
                group=1,
                tasks=foundation_tasks,
                reason="Foundation - must complete first"
            ))

        # Group 2: Implementation (parallel after foundation)
        impl_tasks = [
            t.taskId for t in self.tasks
            if t.taskType in [TaskType.BACKEND, TaskType.FRONTEND]
            and t.taskId not in foundation_tasks
        ]

        if impl_tasks:
            # Add dependencies on foundation
            for task in self.tasks:
                if task.taskId in impl_tasks:
                    task.dependencies = foundation_tasks

            self.parallel_groups.append(ParallelGroup(
                group=2,
                tasks=impl_tasks,
                reason="Parallel implementation after foundation"
            ))

        # Group 3: Testing (parallel, depends on implementation)
        test_tasks = [
            t.taskId for t in self.tasks
            if t.taskType == TaskType.TEST
        ]

        if test_tasks:
            # Add dependencies on implementation
            for task in self.tasks:
                if task.taskId in test_tasks:
                    task.dependencies = impl_tasks if impl_tasks else foundation_tasks

            self.parallel_groups.append(ParallelGroup(
                group=3,
                tasks=test_tasks,
                reason="Parallel testing after implementation"
            ))

        # Group 4: Quality validation (final, depends on all)
        quality_tasks = [
            t.taskId for t in self.tasks
            if t.taskType == TaskType.QUALITY
        ]

        if quality_tasks:
            # Add dependencies on everything before
            all_previous = foundation_tasks + impl_tasks + test_tasks
            for task in self.tasks:
                if task.taskId in quality_tasks:
                    task.dependencies = all_previous

            self.parallel_groups.append(ParallelGroup(
                group=4,
                tasks=quality_tasks,
                reason="Final quality validation"
            ))

        print(f"✅ Optimized into {len(self.parallel_groups)} parallel groups")

    def generate_matrix(self) -> AgentSkillMatrix:
        """
        Generate complete agent+skill assignment matrix.

        Returns:
            Complete AgentSkillMatrix ready for execution
        """
        # Calculate statistics
        skill_based = sum(1 for t in self.tasks if t.useSkill is not None)
        custom_instruction = len(self.tasks) - skill_based
        unique_agents = len(set(t.assignedAgent for t in self.tasks))

        # Estimate parallelism
        if self.parallel_groups:
            max_parallel = max(len(g.tasks) for g in self.parallel_groups)
            speedup = len(self.tasks) / len(self.parallel_groups)
            parallelism = f"{len(self.parallel_groups)} groups, {speedup:.1f}x speedup"
        else:
            parallelism = "Sequential execution"

        statistics = Statistics(
            totalTasks=len(self.tasks),
            skillBasedAgents=skill_based,
            customInstructionAgents=custom_instruction,
            uniqueAgents=unique_agents,
            estimatedParallelism=parallelism
        )

        return AgentSkillMatrix(
            project=self.loop1_data.get('project', 'Unknown Project'),
            loop1_package="integration/loop1-to-loop2",
            tasks=self.tasks,
            parallelGroups=self.parallel_groups,
            statistics=statistics
        )

    def save_matrix(self, output_path: str) -> None:
        """
        Save agent+skill assignment matrix to file.

        Args:
            output_path: Path to save matrix JSON
        """
        matrix = self.generate_matrix()

        # Convert to dict for JSON serialization
        matrix_dict = {
            'project': matrix.project,
            'loop1_package': matrix.loop1_package,
            'tasks': [self._task_to_dict(t) for t in matrix.tasks],
            'parallelGroups': [asdict(g) for g in matrix.parallelGroups],
            'statistics': asdict(matrix.statistics)
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w') as f:
            json.dump(matrix_dict, f, indent=2)

        print(f"✅ Saved agent+skill matrix: {output}")
        print(f"   Total tasks: {matrix.statistics.totalTasks}")
        print(f"   Skill-based: {matrix.statistics.skillBasedAgents}")
        print(f"   Custom instructions: {matrix.statistics.customInstructionAgents}")
        print(f"   Parallelism: {matrix.statistics.estimatedParallelism}")

    def _task_to_dict(self, task: Task) -> Dict[str, Any]:
        """Convert Task to dictionary for JSON serialization"""
        return {
            'taskId': task.taskId,
            'description': task.description,
            'taskType': task.taskType.value,
            'complexity': task.complexity.value,
            'assignedAgent': task.assignedAgent,
            'useSkill': task.useSkill,
            'customInstructions': task.customInstructions,
            'priority': task.priority.value,
            'dependencies': task.dependencies,
            'loop1_research': task.loop1_research,
            'loop1_risk_mitigation': task.loop1_risk_mitigation
        }


def main():
    """Main entry point for swarm coordinator"""
    if len(sys.argv) < 3:
        print("Usage: python swarm-coordinator.py <loop1-package.json> <output-matrix.json>")
        sys.exit(1)

    loop1_package = sys.argv[1]
    output_matrix = sys.argv[2]

    print("=== Swarm Coordinator - Loop 2 Meta-Orchestration ===\n")

    coordinator = SwarmCoordinator(loop1_package)

    # PHASE 1: Load Loop 1 context
    print("PHASE 1: Loading Loop 1 planning package...")
    coordinator.load_loop1_package()

    # PHASE 2-4: Analyze and assign
    print("\nPHASE 2-4: Analyzing tasks and assigning agents+skills...")
    coordinator.analyze_tasks()

    # PHASE 5: Optimize
    print("\nPHASE 5: Optimizing parallel execution groups...")
    coordinator.optimize_parallel_groups()

    # PHASE 6: Generate and save
    print("\nPHASE 6: Generating agent+skill assignment matrix...")
    coordinator.save_matrix(output_matrix)

    print("\n✅ Swarm coordination complete - ready for execution")


if __name__ == "__main__":
    main()
