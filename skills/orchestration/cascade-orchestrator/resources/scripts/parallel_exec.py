#!/usr/bin/env python3
"""
Parallel Execution Engine
Handles concurrent stage execution with dependency management and result aggregation
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of parallel tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ParallelTask:
    """Task for parallel execution"""
    task_id: str
    name: str
    callable: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class TaskResult:
    """Result from task execution"""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    retries: int = 0


class DependencyGraph:
    """Manages task dependencies"""

    def __init__(self):
        self.graph: Dict[str, Set[str]] = {}
        self.reverse_graph: Dict[str, Set[str]] = {}

    def add_task(self, task_id: str, dependencies: Set[str]):
        """Add task with dependencies"""
        self.graph[task_id] = dependencies

        # Build reverse graph for efficient lookup
        if task_id not in self.reverse_graph:
            self.reverse_graph[task_id] = set()

        for dep in dependencies:
            if dep not in self.reverse_graph:
                self.reverse_graph[dep] = set()
            self.reverse_graph[dep].add(task_id)

    def get_ready_tasks(self, completed: Set[str], pending: Set[str]) -> Set[str]:
        """Get tasks ready to execute"""
        ready = set()

        for task_id in pending:
            deps = self.graph.get(task_id, set())
            if deps.issubset(completed):
                ready.add(task_id)

        return ready

    def get_dependents(self, task_id: str) -> Set[str]:
        """Get tasks that depend on given task"""
        return self.reverse_graph.get(task_id, set())

    def has_cycle(self) -> bool:
        """Check for circular dependencies"""
        visited = set()
        rec_stack = set()

        def visit(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.graph.get(node, set()):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self.graph:
            if node not in visited:
                if visit(node):
                    return True

        return False


class ParallelExecutor:
    """Executes tasks in parallel with dependency management"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.dep_graph = DependencyGraph()
        self.tasks: Dict[str, ParallelTask] = {}
        self.results: Dict[str, TaskResult] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def add_task(self, task: ParallelTask):
        """Add task to execution queue"""
        self.tasks[task.task_id] = task
        self.dep_graph.add_task(task.task_id, task.dependencies)
        logger.info(f"Added task: {task.name} (deps: {task.dependencies})")

    def execute_all(self) -> Dict[str, TaskResult]:
        """Execute all tasks with dependency management"""
        logger.info(f"Starting parallel execution of {len(self.tasks)} tasks")

        # Check for cycles
        if self.dep_graph.has_cycle():
            raise ValueError("Circular dependency detected in task graph")

        completed = set()
        pending = set(self.tasks.keys())
        futures = {}

        start_time = time.time()

        try:
            while pending or futures:
                # Get ready tasks
                ready = self.dep_graph.get_ready_tasks(completed, pending)

                # Submit ready tasks
                for task_id in ready:
                    task = self.tasks[task_id]
                    logger.info(f"Submitting task: {task.name}")

                    future = self.executor.submit(
                        self._execute_task_with_retry,
                        task
                    )
                    futures[future] = task_id
                    pending.remove(task_id)

                # Wait for at least one task to complete
                if futures:
                    done, _ = concurrent.futures.wait(
                        futures.keys(),
                        timeout=1.0,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        task_id = futures[future]
                        task = self.tasks[task_id]

                        try:
                            result = future.result()
                            self.results[task_id] = result

                            if result.status == TaskStatus.COMPLETED:
                                completed.add(task_id)
                                logger.info(f"Task completed: {task.name}")
                            else:
                                logger.error(f"Task failed: {task.name} - {result.error}")
                                # Cancel dependent tasks
                                self._cancel_dependents(task_id, pending)

                        except Exception as e:
                            logger.error(f"Task exception: {task.name} - {e}")
                            self.results[task_id] = TaskResult(
                                task_id=task_id,
                                status=TaskStatus.FAILED,
                                error=str(e)
                            )
                            self._cancel_dependents(task_id, pending)

                        finally:
                            del futures[future]

                # Small delay to prevent busy waiting
                if not ready and futures:
                    time.sleep(0.1)

            duration = time.time() - start_time
            logger.info(f"All tasks completed in {duration:.2f}s")

            return self.results

        finally:
            self.executor.shutdown(wait=False)

    def _execute_task_with_retry(self, task: ParallelTask) -> TaskResult:
        """Execute task with retry logic"""
        start_time = time.time()
        retries = 0

        while retries <= task.max_retries:
            try:
                logger.debug(f"Executing task: {task.name} (attempt {retries + 1})")

                # Execute with timeout if specified
                if task.timeout:
                    output = self._execute_with_timeout(
                        task.callable,
                        task.args,
                        task.kwargs,
                        task.timeout
                    )
                else:
                    output = task.callable(*task.args, **task.kwargs)

                duration = time.time() - start_time

                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    output=output,
                    duration=duration,
                    retries=retries
                )

            except Exception as e:
                retries += 1
                logger.warning(f"Task {task.name} failed (attempt {retries}): {e}")

                if retries > task.max_retries:
                    duration = time.time() - start_time
                    return TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        duration=duration,
                        retries=retries - 1
                    )

                # Exponential backoff
                time.sleep(2 ** retries * 0.1)

    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: Dict, timeout: float) -> Any:
        """Execute function with timeout"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Task execution exceeded {timeout}s")

        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            signal.alarm(0)
            raise

    def _cancel_dependents(self, task_id: str, pending: Set[str]):
        """Cancel tasks that depend on failed task"""
        dependents = self.dep_graph.get_dependents(task_id)

        for dep in dependents:
            if dep in pending:
                logger.warning(f"Cancelling task {self.tasks[dep].name} due to failed dependency")
                self.results[dep] = TaskResult(
                    task_id=dep,
                    status=TaskStatus.CANCELLED,
                    error=f"Dependency {task_id} failed"
                )
                pending.remove(dep)

                # Recursively cancel dependents
                self._cancel_dependents(dep, pending)


class AsyncParallelExecutor:
    """Async version of parallel executor"""

    def __init__(self, max_concurrency: int = 10):
        self.max_concurrency = max_concurrency
        self.dep_graph = DependencyGraph()
        self.tasks: Dict[str, ParallelTask] = {}
        self.results: Dict[str, TaskResult] = {}
        self.semaphore = asyncio.Semaphore(max_concurrency)

    def add_task(self, task: ParallelTask):
        """Add task to execution queue"""
        self.tasks[task.task_id] = task
        self.dep_graph.add_task(task.task_id, task.dependencies)

    async def execute_all(self) -> Dict[str, TaskResult]:
        """Execute all tasks asynchronously"""
        logger.info(f"Starting async parallel execution of {len(self.tasks)} tasks")

        if self.dep_graph.has_cycle():
            raise ValueError("Circular dependency detected")

        completed = set()
        pending = set(self.tasks.keys())
        running_tasks = {}

        start_time = time.time()

        while pending or running_tasks:
            # Get ready tasks
            ready = self.dep_graph.get_ready_tasks(completed, pending)

            # Submit ready tasks
            for task_id in ready:
                task = self.tasks[task_id]
                logger.info(f"Starting async task: {task.name}")

                async_task = asyncio.create_task(self._execute_task_async(task))
                running_tasks[async_task] = task_id
                pending.remove(task_id)

            # Wait for at least one task
            if running_tasks:
                done, _ = await asyncio.wait(
                    running_tasks.keys(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                for async_task in done:
                    task_id = running_tasks[async_task]
                    task = self.tasks[task_id]

                    try:
                        result = await async_task
                        self.results[task_id] = result

                        if result.status == TaskStatus.COMPLETED:
                            completed.add(task_id)
                            logger.info(f"Async task completed: {task.name}")
                        else:
                            logger.error(f"Async task failed: {task.name}")
                            self._cancel_dependents(task_id, pending)

                    except Exception as e:
                        logger.error(f"Async task exception: {e}")
                        self.results[task_id] = TaskResult(
                            task_id=task_id,
                            status=TaskStatus.FAILED,
                            error=str(e)
                        )

                    finally:
                        del running_tasks[async_task]

        duration = time.time() - start_time
        logger.info(f"All async tasks completed in {duration:.2f}s")

        return self.results

    async def _execute_task_async(self, task: ParallelTask) -> TaskResult:
        """Execute task asynchronously"""
        async with self.semaphore:
            start_time = time.time()

            try:
                # Execute callable
                if asyncio.iscoroutinefunction(task.callable):
                    output = await task.callable(*task.args, **task.kwargs)
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    output = await loop.run_in_executor(
                        None,
                        task.callable,
                        *task.args
                    )

                duration = time.time() - start_time

                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    output=output,
                    duration=duration
                )

            except Exception as e:
                duration = time.time() - start_time
                return TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    duration=duration
                )

    def _cancel_dependents(self, task_id: str, pending: Set[str]):
        """Cancel dependent tasks"""
        dependents = self.dep_graph.get_dependents(task_id)

        for dep in dependents:
            if dep in pending:
                self.results[dep] = TaskResult(
                    task_id=dep,
                    status=TaskStatus.CANCELLED,
                    error=f"Dependency {task_id} failed"
                )
                pending.remove(dep)
                self._cancel_dependents(dep, pending)


# Example usage
if __name__ == "__main__":
    import sys

    # Example tasks
    def task_a():
        time.sleep(1)
        return "Result A"

    def task_b():
        time.sleep(0.5)
        return "Result B"

    def task_c(a_result, b_result):
        time.sleep(0.3)
        return f"Result C (from {a_result} and {b_result})"

    # Create executor
    executor = ParallelExecutor(max_workers=3)

    # Add tasks with dependencies
    executor.add_task(ParallelTask("task_a", "Task A", task_a))
    executor.add_task(ParallelTask("task_b", "Task B", task_b))
    executor.add_task(ParallelTask(
        "task_c",
        "Task C",
        task_c,
        dependencies={"task_a", "task_b"}
    ))

    # Execute
    results = executor.execute_all()

    # Print results
    print("\nExecution Results:")
    print(json.dumps({k: v.__dict__ for k, v in results.items()}, indent=2, default=str))
