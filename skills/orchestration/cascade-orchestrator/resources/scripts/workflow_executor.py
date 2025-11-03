#!/usr/bin/env python3
"""
Workflow Executor - Sophisticated Cascade Orchestration Engine
Enhanced with multi-model routing, Codex iteration, and swarm coordination
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StageType(Enum):
    """Types of stages in a cascade"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    CODEX_SANDBOX = "codex-sandbox"
    MULTI_MODEL = "multi-model"
    SWARM_PARALLEL = "swarm-parallel"


class ExecutionStatus(Enum):
    """Execution status for stages"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ModelConfig:
    """AI model configuration"""
    name: str
    context_limit: int
    capabilities: List[str]
    cost_per_1k: float


@dataclass
class StageResult:
    """Result from stage execution"""
    stage_id: str
    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    model_used: Optional[str] = None
    iterations: int = 1


@dataclass
class Stage:
    """Individual stage in a cascade"""
    stage_id: str
    name: str
    stage_type: StageType
    skills: List[str] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    model: str = "auto-select"
    codex_config: Optional[Dict] = None
    swarm_config: Optional[Dict] = None
    memory_config: Optional[Dict] = None
    error_handling: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Cascade:
    """Complete cascade workflow definition"""
    name: str
    description: str
    version: str
    config: Dict[str, Any]
    inputs: Dict[str, Any]
    stages: List[Stage]
    memory: Optional[Dict] = None
    github_integration: Optional[Dict] = None


class ModelSelector:
    """Intelligent AI model selection"""

    MODELS = {
        "claude": ModelConfig("claude-sonnet-4-5", 200000, ["reasoning", "coding", "analysis"], 3.0),
        "gemini-megacontext": ModelConfig("gemini-1.5-pro", 2000000, ["large_context"], 7.0),
        "gemini-search": ModelConfig("gemini-grounded", 100000, ["web_search", "current_info"], 4.0),
        "gemini-media": ModelConfig("gemini-media", 100000, ["image_gen", "video_gen"], 5.0),
        "codex-auto": ModelConfig("gpt-5-codex", 128000, ["rapid_prototype", "auto_execute"], 2.0),
        "codex-reasoning": ModelConfig("gpt-5-codex-reasoning", 128000, ["reasoning", "alternative_view"], 3.5)
    }

    @classmethod
    def select_optimal_model(cls, task_requirements: Dict[str, Any]) -> str:
        """Select best model based on task requirements"""

        # Priority-based selection
        if task_requirements.get("large_context"):
            return "gemini-megacontext"
        elif task_requirements.get("current_info"):
            return "gemini-search"
        elif task_requirements.get("visual_output"):
            return "gemini-media"
        elif task_requirements.get("rapid_prototype"):
            return "codex-auto"
        elif task_requirements.get("alternative_view"):
            return "codex-reasoning"
        else:
            return "claude"  # Best overall

    @classmethod
    def get_model_info(cls, model_name: str) -> ModelConfig:
        """Get model configuration"""
        return cls.MODELS.get(model_name, cls.MODELS["claude"])


class MemoryManager:
    """Memory persistence across stages"""

    def __init__(self):
        self.memory_store: Dict[str, Any] = {}

    def write(self, key: str, value: Any):
        """Write to memory"""
        self.memory_store[key] = value
        logger.info(f"Memory write: {key}")

    def read(self, key: str) -> Optional[Any]:
        """Read from memory"""
        value = self.memory_store.get(key)
        logger.info(f"Memory read: {key} -> {value is not None}")
        return value

    def clear(self):
        """Clear memory"""
        self.memory_store.clear()
        logger.info("Memory cleared")


class CodexSandboxExecutor:
    """Codex sandbox iteration with auto-fix"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations

    async def execute_with_iteration(self, stage: Stage, test_suite: str) -> StageResult:
        """Execute tests with Codex auto-fix iteration"""
        logger.info(f"Codex sandbox iteration for stage: {stage.stage_id}")

        iterations = 0
        all_tests_passed = False

        while iterations < self.max_iterations and not all_tests_passed:
            iterations += 1
            logger.info(f"Iteration {iterations}/{self.max_iterations}")

            # Run tests
            test_results = await self._run_tests(test_suite)

            if test_results["all_passed"]:
                all_tests_passed = True
                logger.info(f"All tests passed on iteration {iterations}")
                break

            # Auto-fix failures
            failed_tests = test_results["failed"]
            for test in failed_tests:
                logger.info(f"Auto-fixing test: {test['name']}")
                fix_result = await self._codex_auto_fix(test)

                if fix_result["success"]:
                    logger.info(f"Fix applied for: {test['name']}")
                else:
                    logger.warning(f"Fix failed for: {test['name']}")

        return StageResult(
            stage_id=stage.stage_id,
            status=ExecutionStatus.SUCCESS if all_tests_passed else ExecutionStatus.FAILED,
            output={"iterations": iterations, "all_passed": all_tests_passed},
            iterations=iterations
        )

    async def _run_tests(self, test_suite: str) -> Dict[str, Any]:
        """Run test suite"""
        # Simulate test execution
        logger.info(f"Running test suite: {test_suite}")
        await asyncio.sleep(0.5)

        # Mock result
        return {
            "all_passed": False,
            "passed": 8,
            "failed": [
                {"name": "test_authentication", "error": "AssertionError: Expected 200, got 401"}
            ],
            "total": 9
        }

    async def _codex_auto_fix(self, test: Dict) -> Dict[str, Any]:
        """Use Codex to auto-fix test failure"""
        logger.info(f"Codex auto-fix: {test['name']}")

        # Simulate Codex fix
        await asyncio.sleep(0.3)

        # Mock fix application
        return {"success": True, "fix_applied": "Updated auth middleware"}


class SwarmCoordinator:
    """Swarm-based parallel execution"""

    def __init__(self, topology: str = "mesh", max_agents: int = 4):
        self.topology = topology
        self.max_agents = max_agents

    async def execute_parallel(self, stage: Stage) -> StageResult:
        """Execute stage with swarm coordination"""
        logger.info(f"Swarm parallel execution: {stage.name}")
        logger.info(f"Topology: {self.topology}, Max Agents: {self.max_agents}")

        # Initialize swarm via MCP
        await self._init_swarm()

        # Spawn agents for each skill
        tasks = []
        for skill in stage.skills:
            task = self._execute_skill_with_agent(skill)
            tasks.append(task)

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        aggregated = self._aggregate_results(results)

        return StageResult(
            stage_id=stage.stage_id,
            status=ExecutionStatus.SUCCESS if aggregated["success"] else ExecutionStatus.FAILED,
            output=aggregated
        )

    async def _init_swarm(self):
        """Initialize swarm topology"""
        logger.info(f"Initializing {self.topology} swarm")
        await asyncio.sleep(0.2)

    async def _execute_skill_with_agent(self, skill: str) -> Dict:
        """Execute skill with agent"""
        logger.info(f"Agent executing skill: {skill}")
        await asyncio.sleep(0.5)
        return {"skill": skill, "status": "success", "output": f"Result from {skill}"}

    def _aggregate_results(self, results: List) -> Dict:
        """Aggregate parallel results"""
        successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
        return {
            "success": len(successful) == len(results),
            "results": successful,
            "total": len(results),
            "successful": len(successful)
        }


class WorkflowExecutor:
    """Main workflow execution engine"""

    def __init__(self):
        self.memory = MemoryManager()
        self.model_selector = ModelSelector()
        self.codex_executor = CodexSandboxExecutor()
        self.swarm_coordinator = SwarmCoordinator()
        self.results: Dict[str, StageResult] = {}

    async def execute_cascade(self, cascade: Cascade) -> Dict[str, Any]:
        """Execute complete cascade workflow"""
        logger.info(f"Starting cascade execution: {cascade.name}")
        logger.info(f"Version: {cascade.version}")

        start_time = asyncio.get_event_loop().time()

        try:
            # Execute stages in order
            for stage in cascade.stages:
                # Check dependencies
                if not self._check_dependencies(stage):
                    logger.warning(f"Skipping stage {stage.stage_id}: dependencies not met")
                    self.results[stage.stage_id] = StageResult(
                        stage_id=stage.stage_id,
                        status=ExecutionStatus.SKIPPED
                    )
                    continue

                # Execute stage based on type
                result = await self._execute_stage(stage)
                self.results[stage.stage_id] = result

                # Store in memory if configured
                if stage.memory_config and stage.memory_config.get("write_keys"):
                    for key in stage.memory_config["write_keys"]:
                        self.memory.write(key, result.output)

                # Handle failures
                if result.status == ExecutionStatus.FAILED:
                    if not self._handle_failure(stage, result):
                        logger.error(f"Stage {stage.stage_id} failed and cannot recover")
                        break

            duration = asyncio.get_event_loop().time() - start_time

            return {
                "cascade": cascade.name,
                "status": self._get_overall_status(),
                "duration": duration,
                "stages": {k: v.__dict__ for k, v in self.results.items()},
                "memory_snapshot": self.memory.memory_store
            }

        except Exception as e:
            logger.error(f"Cascade execution error: {e}", exc_info=True)
            return {
                "cascade": cascade.name,
                "status": "error",
                "error": str(e)
            }

    async def _execute_stage(self, stage: Stage) -> StageResult:
        """Execute individual stage"""
        logger.info(f"Executing stage: {stage.name} ({stage.stage_type.value})")

        stage_start = asyncio.get_event_loop().time()

        try:
            # Select model if auto-select
            if stage.model == "auto-select":
                model = self.model_selector.select_optimal_model(stage.inputs)
                logger.info(f"Auto-selected model: {model}")
            else:
                model = stage.model

            # Execute based on stage type
            if stage.stage_type == StageType.CODEX_SANDBOX:
                result = await self.codex_executor.execute_with_iteration(stage, "test_suite")

            elif stage.stage_type == StageType.SWARM_PARALLEL:
                result = await self.swarm_coordinator.execute_parallel(stage)

            elif stage.stage_type == StageType.PARALLEL:
                result = await self._execute_parallel_stage(stage)

            elif stage.stage_type == StageType.CONDITIONAL:
                result = await self._execute_conditional_stage(stage)

            else:  # Sequential
                result = await self._execute_sequential_stage(stage)

            result.model_used = model
            result.duration = asyncio.get_event_loop().time() - stage_start

            logger.info(f"Stage {stage.stage_id} completed: {result.status.value}")
            return result

        except Exception as e:
            logger.error(f"Stage execution error: {e}", exc_info=True)
            return StageResult(
                stage_id=stage.stage_id,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )

    async def _execute_sequential_stage(self, stage: Stage) -> StageResult:
        """Execute sequential stage"""
        outputs = []

        for skill in stage.skills:
            logger.info(f"Executing skill: {skill}")
            output = await self._execute_skill(skill, stage.inputs)
            outputs.append(output)

        return StageResult(
            stage_id=stage.stage_id,
            status=ExecutionStatus.SUCCESS,
            output=outputs
        )

    async def _execute_parallel_stage(self, stage: Stage) -> StageResult:
        """Execute parallel stage"""
        tasks = [self._execute_skill(skill, stage.inputs) for skill in stage.skills]
        outputs = await asyncio.gather(*tasks)

        return StageResult(
            stage_id=stage.stage_id,
            status=ExecutionStatus.SUCCESS,
            output=list(outputs)
        )

    async def _execute_conditional_stage(self, stage: Stage) -> StageResult:
        """Execute conditional stage"""
        # Evaluate condition
        condition_met = self._evaluate_condition(stage.condition or "")

        logger.info(f"Condition {stage.condition}: {condition_met}")

        if condition_met:
            return await self._execute_sequential_stage(stage)
        else:
            return StageResult(
                stage_id=stage.stage_id,
                status=ExecutionStatus.SKIPPED,
                output="Condition not met"
            )

    async def _execute_skill(self, skill: str, inputs: Dict) -> Any:
        """Execute individual skill"""
        await asyncio.sleep(0.3)  # Simulate execution
        return f"Output from {skill}"

    def _check_dependencies(self, stage: Stage) -> bool:
        """Check if stage dependencies are satisfied"""
        for dep in stage.dependencies:
            if dep not in self.results or self.results[dep].status != ExecutionStatus.SUCCESS:
                return False
        return True

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate conditional expression"""
        # Simple evaluation - in production would use safe eval
        return True  # Mock

    def _handle_failure(self, stage: Stage, result: StageResult) -> bool:
        """Handle stage failure"""
        strategy = stage.error_handling.get("strategy", "fail")

        if strategy == "retry":
            max_retries = stage.error_handling.get("max_retries", 3)
            logger.info(f"Retrying stage {stage.stage_id}")
            return True

        elif strategy == "codex-auto-fix":
            logger.info(f"Attempting Codex auto-fix for {stage.stage_id}")
            return True

        elif strategy == "model-switch":
            logger.info(f"Switching model for {stage.stage_id}")
            return True

        return False

    def _get_overall_status(self) -> str:
        """Get overall cascade status"""
        if not self.results:
            return "no_stages"

        if all(r.status == ExecutionStatus.SUCCESS for r in self.results.values()):
            return "success"
        elif any(r.status == ExecutionStatus.FAILED for r in self.results.values()):
            return "failed"
        else:
            return "partial"


def load_cascade_definition(filepath: str) -> Cascade:
    """Load cascade from YAML/JSON file"""
    import yaml

    with open(filepath, 'r') as f:
        if filepath.endswith('.yaml') or filepath.endswith('.yml'):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    # Parse stages
    stages = []
    for stage_data in data.get("stages", []):
        stage = Stage(
            stage_id=stage_data["stage_id"],
            name=stage_data["name"],
            stage_type=StageType(stage_data.get("type", "sequential")),
            skills=stage_data.get("skills", []),
            inputs=stage_data.get("inputs", {}),
            model=stage_data.get("model", "auto-select"),
            codex_config=stage_data.get("codex_config"),
            swarm_config=stage_data.get("swarm_config"),
            memory_config=stage_data.get("memory"),
            error_handling=stage_data.get("error_handling", {}),
            condition=stage_data.get("condition"),
            dependencies=stage_data.get("dependencies", [])
        )
        stages.append(stage)

    return Cascade(
        name=data["cascade"]["name"],
        description=data["cascade"]["description"],
        version=data["cascade"].get("version", "1.0.0"),
        config=data["cascade"].get("config", {}),
        inputs=data.get("inputs", {}),
        stages=stages,
        memory=data.get("memory"),
        github_integration=data.get("github_integration")
    )


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Workflow Executor - Cascade Orchestration")
    parser.add_argument("cascade_file", help="Path to cascade definition file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Output file for results")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load cascade
    logger.info(f"Loading cascade from: {args.cascade_file}")
    cascade = load_cascade_definition(args.cascade_file)

    # Execute
    executor = WorkflowExecutor()
    results = await executor.execute_cascade(cascade)

    # Output results
    output_json = json.dumps(results, indent=2, default=str)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        logger.info(f"Results written to: {args.output}")
    else:
        print(output_json)

    # Exit code based on status
    sys.exit(0 if results["status"] == "success" else 1)


if __name__ == "__main__":
    asyncio.run(main())
