#!/usr/bin/env python3
"""
Queen Coordinator - Strategic Queen Implementation for Hive Mind
Implements high-level coordination, task delegation, and consensus building.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueenType(Enum):
    """Queen specialization types"""
    STRATEGIC = "strategic"  # Research, planning, analysis
    TACTICAL = "tactical"    # Implementation, execution
    ADAPTIVE = "adaptive"    # Optimization, dynamic adjustment


class ConsensusAlgorithm(Enum):
    """Consensus decision algorithms"""
    MAJORITY = "majority"    # Simple majority voting
    WEIGHTED = "weighted"    # Queen has 3x weight
    BYZANTINE = "byzantine"  # 2/3 supermajority required


@dataclass
class WorkerAgent:
    """Worker agent representation"""
    id: str
    type: str
    capabilities: List[str]
    status: str = "idle"
    tasks_completed: int = 0
    performance_score: float = 1.0


@dataclass
class Task:
    """Task representation"""
    id: str
    description: str
    priority: int
    assigned_to: Optional[str] = None
    status: str = "pending"
    created_at: str = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


@dataclass
class ConsensusDecision:
    """Consensus decision result"""
    topic: str
    decision: str
    confidence: float
    votes: Dict[str, str]
    timestamp: str


class QueenCoordinator:
    """Strategic queen coordinator for hive mind"""

    def __init__(
        self,
        queen_type: QueenType,
        objective: str,
        max_workers: int = 8,
        consensus_algorithm: ConsensusAlgorithm = ConsensusAlgorithm.BYZANTINE
    ):
        self.queen_type = queen_type
        self.objective = objective
        self.max_workers = max_workers
        self.consensus_algorithm = consensus_algorithm
        self.workers: Dict[str, WorkerAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.decisions: List[ConsensusDecision] = []

        logger.info(f"Queen coordinator initialized: {queen_type.value}")

    async def spawn_workers(self, worker_types: List[str]) -> List[WorkerAgent]:
        """Spawn worker agents based on objective needs"""
        spawned = []
        for i, worker_type in enumerate(worker_types[:self.max_workers]):
            worker = WorkerAgent(
                id=f"worker-{i+1}",
                type=worker_type,
                capabilities=self._get_worker_capabilities(worker_type)
            )
            self.workers[worker.id] = worker
            spawned.append(worker)
            logger.info(f"Spawned worker: {worker.id} ({worker.type})")

        return spawned

    def _get_worker_capabilities(self, worker_type: str) -> List[str]:
        """Get capabilities for worker type"""
        capability_map = {
            "researcher": ["analysis", "investigation", "data-gathering", "research"],
            "coder": ["implementation", "development", "coding", "refactoring"],
            "analyst": ["testing", "review", "quality-assurance", "metrics"],
            "optimizer": ["performance", "tuning", "benchmarking", "optimization"],
            "architect": ["design", "planning", "architecture", "system-design"],
            "tester": ["testing", "validation", "qa", "debugging"],
            "documenter": ["documentation", "writing", "API-docs", "guides"]
        }
        return capability_map.get(worker_type, ["general"])

    async def delegate_task(self, description: str, priority: int = 5) -> Task:
        """Intelligently delegate task to best worker"""
        task = Task(
            id=f"task-{len(self.tasks) + 1}",
            description=description,
            priority=priority
        )

        # Find best worker based on capabilities and performance
        best_worker = self._find_best_worker(description)
        if best_worker:
            task.assigned_to = best_worker.id
            best_worker.status = "busy"
            logger.info(f"Task {task.id} assigned to {best_worker.id}")
        else:
            logger.warning(f"No available worker for task {task.id}")

        self.tasks[task.id] = task
        return task

    def _find_best_worker(self, description: str) -> Optional[WorkerAgent]:
        """Find best worker based on task description and worker capabilities"""
        # Simple keyword matching + performance scoring
        keywords = description.lower().split()

        best_worker = None
        best_score = -1

        for worker in self.workers.values():
            if worker.status != "idle":
                continue

            # Score based on capability match + performance
            capability_score = sum(
                1 for cap in worker.capabilities
                if any(kw in cap for kw in keywords)
            )
            total_score = capability_score * worker.performance_score

            if total_score > best_score:
                best_score = total_score
                best_worker = worker

        return best_worker

    async def build_consensus(
        self,
        topic: str,
        options: List[str]
    ) -> ConsensusDecision:
        """Build consensus among workers on a decision"""
        logger.info(f"Building consensus on: {topic}")

        # Simulate worker voting (in production, this would query actual agents)
        votes = {}
        for worker_id, worker in self.workers.items():
            # Simplified voting logic (in reality, agents would reason about options)
            vote = options[hash(worker_id) % len(options)]
            votes[worker_id] = vote

        # Queen's vote (weighted if using weighted consensus)
        queen_vote = self._make_strategic_decision(topic, options)
        if self.consensus_algorithm == ConsensusAlgorithm.WEIGHTED:
            # Queen vote counts 3x
            votes["queen-1"] = queen_vote
            votes["queen-2"] = queen_vote
            votes["queen-3"] = queen_vote
        else:
            votes["queen"] = queen_vote

        # Calculate decision based on algorithm
        decision, confidence = self._calculate_consensus(votes, options)

        consensus = ConsensusDecision(
            topic=topic,
            decision=decision,
            confidence=confidence,
            votes=votes,
            timestamp=datetime.utcnow().isoformat()
        )

        self.decisions.append(consensus)
        logger.info(f"Consensus reached: {decision} ({confidence:.1%} confidence)")

        return consensus

    def _make_strategic_decision(self, topic: str, options: List[str]) -> str:
        """Queen makes strategic decision based on type and objective"""
        # Strategic queen prefers research and analysis
        if self.queen_type == QueenType.STRATEGIC:
            # Prefer options that involve planning/research
            for opt in options:
                if any(kw in opt.lower() for kw in ["research", "analyze", "plan"]):
                    return opt

        # Tactical queen prefers implementation
        elif self.queen_type == QueenType.TACTICAL:
            for opt in options:
                if any(kw in opt.lower() for kw in ["implement", "build", "execute"]):
                    return opt

        # Adaptive queen optimizes
        elif self.queen_type == QueenType.ADAPTIVE:
            for opt in options:
                if any(kw in opt.lower() for kw in ["optimize", "improve", "enhance"]):
                    return opt

        # Default to first option
        return options[0]

    def _calculate_consensus(
        self,
        votes: Dict[str, str],
        options: List[str]
    ) -> Tuple[str, float]:
        """Calculate consensus decision based on algorithm"""
        vote_counts = {opt: 0 for opt in options}
        for vote in votes.values():
            if vote in vote_counts:
                vote_counts[vote] += 1

        total_votes = len(votes)

        if self.consensus_algorithm == ConsensusAlgorithm.BYZANTINE:
            # Require 2/3 supermajority
            for opt, count in vote_counts.items():
                if count >= (2 * total_votes / 3):
                    return opt, count / total_votes
            # No supermajority - return majority
            winner = max(vote_counts.items(), key=lambda x: x[1])
            return winner[0], winner[1] / total_votes

        else:
            # Majority or weighted - return option with most votes
            winner = max(vote_counts.items(), key=lambda x: x[1])
            return winner[0], winner[1] / total_votes

    async def monitor_progress(self) -> Dict:
        """Monitor hive mind progress"""
        completed = sum(1 for t in self.tasks.values() if t.status == "completed")
        total = len(self.tasks)
        progress = (completed / total * 100) if total > 0 else 0

        return {
            "objective": self.objective,
            "queen_type": self.queen_type.value,
            "workers": len(self.workers),
            "tasks": {
                "total": total,
                "completed": completed,
                "pending": sum(1 for t in self.tasks.values() if t.status == "pending"),
                "in_progress": sum(1 for t in self.tasks.values() if t.status == "in_progress"),
                "progress": f"{progress:.1f}%"
            },
            "consensus_decisions": len(self.decisions),
            "timestamp": datetime.utcnow().isoformat()
        }

    def export_state(self) -> Dict:
        """Export hive mind state for persistence"""
        return {
            "queen_type": self.queen_type.value,
            "objective": self.objective,
            "max_workers": self.max_workers,
            "consensus_algorithm": self.consensus_algorithm.value,
            "workers": [asdict(w) for w in self.workers.values()],
            "tasks": [asdict(t) for t in self.tasks.values()],
            "decisions": [asdict(d) for d in self.decisions],
            "timestamp": datetime.utcnow().isoformat()
        }


async def main():
    """Demo queen coordinator"""
    # Initialize strategic queen
    queen = QueenCoordinator(
        queen_type=QueenType.STRATEGIC,
        objective="Build microservices architecture",
        max_workers=8,
        consensus_algorithm=ConsensusAlgorithm.BYZANTINE
    )

    # Spawn workers
    worker_types = ["researcher", "coder", "architect", "tester", "optimizer"]
    await queen.spawn_workers(worker_types)

    # Delegate tasks
    await queen.delegate_task("Research microservices patterns", priority=9)
    await queen.delegate_task("Design API architecture", priority=8)
    await queen.delegate_task("Implement authentication service", priority=7)

    # Build consensus on architecture decision
    decision = await queen.build_consensus(
        "API architecture pattern",
        ["REST", "GraphQL", "gRPC"]
    )
    print(f"\nConsensus Decision: {decision.decision} ({decision.confidence:.1%})")

    # Monitor progress
    status = await queen.monitor_progress()
    print(f"\nHive Mind Status:")
    print(json.dumps(status, indent=2))

    # Export state
    state = queen.export_state()
    print(f"\nExported state with {len(state['workers'])} workers, {len(state['tasks'])} tasks")


if __name__ == "__main__":
    asyncio.run(main())
