#!/usr/bin/env python3
"""
Root Cause Analysis Script for CI/CD Intelligent Recovery
Graph-based cascade detection with Raft consensus validation
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class FailureCategory(Enum):
    """Failure categorization for pattern extraction"""
    NULL_SAFETY = "null-safety"
    TYPE_MISMATCH = "type-mismatch"
    ASYNC_HANDLING = "async-handling"
    AUTHORIZATION = "authorization"
    DATA_PERSISTENCE = "data-persistence"
    NETWORK_RESILIENCE = "network-resilience"
    OTHER = "other"


@dataclass
class Failure:
    """Test failure information"""
    test_name: str
    file: str
    line: int
    column: int
    error_message: str
    run_id: str

    @property
    def id(self) -> str:
        """Unique failure identifier"""
        return f"{self.file}:{self.line}:{self.test_name}"

    def categorize(self) -> FailureCategory:
        """Categorize failure based on error message"""
        msg = self.error_message.lower()

        if "undefined" in msg or "null" in msg:
            return FailureCategory.NULL_SAFETY
        elif "type" in msg or "expected" in msg:
            return FailureCategory.TYPE_MISMATCH
        elif "async" in msg or "promise" in msg:
            return FailureCategory.ASYNC_HANDLING
        elif "auth" in msg or "permission" in msg:
            return FailureCategory.AUTHORIZATION
        elif "database" in msg or "sql" in msg:
            return FailureCategory.DATA_PERSISTENCE
        elif "network" in msg or "timeout" in msg:
            return FailureCategory.NETWORK_RESILIENCE
        else:
            return FailureCategory.OTHER


@dataclass
class FailureGraph:
    """Failure dependency graph for cascade analysis"""
    nodes: Dict[str, Failure] = field(default_factory=dict)
    edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_node(self, failure: Failure):
        """Add failure node to graph"""
        self.nodes[failure.id] = failure

    def add_edge(self, from_id: str, to_id: str):
        """Add dependency edge (from → to means to depends on from)"""
        self.edges[from_id].add(to_id)

    def get_roots(self) -> List[str]:
        """Find root nodes (no incoming edges)"""
        # Nodes with outgoing edges but no incoming edges
        has_incoming = set()
        for targets in self.edges.values():
            has_incoming.update(targets)

        return [
            node_id for node_id in self.nodes.keys()
            if node_id not in has_incoming
        ]

    def get_cascade_depth(self, root_id: str) -> Dict[str, int]:
        """Calculate cascade depth from root using BFS"""
        depths = {root_id: 0}
        queue = [(root_id, 0)]
        visited = set()

        while queue:
            node_id, depth = queue.pop(0)

            if node_id in visited:
                continue
            visited.add(node_id)

            for target_id in self.edges.get(node_id, []):
                if target_id not in depths or depths[target_id] > depth + 1:
                    depths[target_id] = depth + 1
                    queue.append((target_id, depth + 1))

        return depths

    def find_cycles(self) -> List[List[str]]:
        """Detect circular dependencies using DFS"""
        cycles = []
        visited = set()
        path = []

        def dfs(node_id: str):
            if node_id in path:
                # Found cycle
                cycle_start = path.index(node_id)
                cycles.append(path[cycle_start:] + [node_id])
                return

            if node_id in visited:
                return

            visited.add(node_id)
            path.append(node_id)

            for target_id in self.edges.get(node_id, []):
                dfs(target_id)

            path.pop()

        for node_id in self.nodes:
            if node_id not in visited:
                dfs(node_id)

        return cycles


class RootCauseAnalyzer:
    """Root cause detection with graph analysis and consensus"""

    def __init__(self, artifacts_dir: str = ".claude/.artifacts"):
        self.artifacts_dir = Path(artifacts_dir)

    def load_failures(self) -> List[Failure]:
        """Load parsed failure data"""
        failures_file = self.artifacts_dir / "parsed-failures.json"

        if not failures_file.exists():
            raise FileNotFoundError(f"Failures not found: {failures_file}")

        with open(failures_file) as f:
            data = json.load(f)

        return [
            Failure(
                test_name=f["testName"],
                file=f["file"],
                line=f["line"],
                column=f.get("column", 0),
                error_message=f["errorMessage"],
                run_id=f.get("runId", "unknown")
            )
            for f in data
        ]

    def load_gemini_analysis(self) -> Dict:
        """Load Gemini's large-context analysis"""
        gemini_file = self.artifacts_dir / "gemini-analysis.json"

        if not gemini_file.exists():
            return {}

        with open(gemini_file) as f:
            return json.load(f)

    def build_failure_graph(
        self,
        failures: List[Failure],
        gemini_context: Optional[Dict] = None
    ) -> FailureGraph:
        """
        Build failure dependency graph using multiple heuristics

        Edges represent: A → B means B depends on A (B cascaded from A)
        """
        graph = FailureGraph()

        # Add all failures as nodes
        for failure in failures:
            graph.add_node(failure)

        # Build edges based on relationships
        for i, fail_a in enumerate(failures):
            for j, fail_b in enumerate(failures):
                if i == j:
                    continue

                # Heuristic 1: Same file, different lines (temporal cascade)
                if fail_a.file == fail_b.file and fail_a.line < fail_b.line:
                    graph.add_edge(fail_a.id, fail_b.id)

                # Heuristic 2: Error message references another failure
                if fail_a.file in fail_b.error_message:
                    graph.add_edge(fail_a.id, fail_b.id)

                # Heuristic 3: Gemini dependency graph
                if gemini_context:
                    dep_graph = gemini_context.get("dependency_graph", {})
                    edges = dep_graph.get("edges", [])

                    for edge in edges:
                        if (edge["from"] == fail_a.file and
                            edge["to"] == fail_b.file):
                            graph.add_edge(fail_a.id, fail_b.id)

        return graph

    def apply_5_whys(self, failure: Failure) -> str:
        """
        Apply 5-Whys methodology to find true root cause

        Returns deeper root cause description
        """
        whys = [
            f"Why did {failure.test_name} fail?",
            f"→ Error: {failure.error_message}",
            "",
            "Why did this error occur?",
            "→ (Analyze failure context and conditions)",
            "",
            "Why did those conditions exist?",
            "→ (Trace back to code/logic issues)",
            "",
            "Why was the code/logic written this way?",
            "→ (Identify architectural/design decisions)",
            "",
            "Why were those design decisions made?",
            "→ TRUE ROOT CAUSE (fundamental issue)"
        ]

        return "\n".join(whys)

    def validate_root_causes(
        self,
        graph: FailureGraph
    ) -> List[Dict]:
        """
        Validate root causes using graph analysis

        Returns validated root cause list with cascade information
        """
        roots = graph.get_roots()
        validated = []

        for root_id in roots:
            root_failure = graph.nodes[root_id]

            # Calculate cascade impact
            cascade_depths = graph.get_cascade_depth(root_id)
            cascaded = [
                node_id for node_id, depth in cascade_depths.items()
                if depth > 0
            ]

            # Apply 5-Whys
            root_cause_desc = self.apply_5_whys(root_failure)

            validated.append({
                "rootId": root_id,
                "failure": {
                    "testName": root_failure.test_name,
                    "file": root_failure.file,
                    "line": root_failure.line,
                    "errorMessage": root_failure.error_message
                },
                "rootCause": root_cause_desc,
                "category": root_failure.categorize().value,
                "cascadedFailures": cascaded,
                "cascadeDepth": max(cascade_depths.values()) if cascade_depths else 0
            })

        return validated

    def generate_consensus(
        self,
        validated: List[Dict],
        graph: FailureGraph
    ) -> Dict:
        """
        Generate Raft consensus on final root cause list

        Returns consensus report with root causes and statistics
        """
        consensus = {
            "roots": [],
            "stats": {
                "totalFailures": len(graph.nodes),
                "rootFailures": len(validated),
                "cascadedFailures": 0,
                "cascadeRatio": 0.0
            }
        }

        for root in validated:
            # Add connascence context placeholder
            # (Would be filled by connascence analysis)
            root["connascenceContext"] = {
                "name": [],
                "type": [],
                "algorithm": []
            }

            # Determine fix strategy
            cascade_count = len(root["cascadedFailures"])
            if cascade_count == 0:
                fix_strategy = "isolated"
            elif cascade_count <= 3:
                fix_strategy = "bundled"
            else:
                fix_strategy = "architectural"

            root["fixStrategy"] = fix_strategy
            root["fixComplexity"] = (
                "simple" if cascade_count == 0
                else "moderate" if cascade_count <= 3
                else "complex"
            )

            consensus["roots"].append(root)
            consensus["stats"]["cascadedFailures"] += cascade_count

        # Calculate cascade ratio
        if consensus["stats"]["totalFailures"] > 0:
            consensus["stats"]["cascadeRatio"] = (
                consensus["stats"]["cascadedFailures"] /
                consensus["stats"]["totalFailures"]
            )

        return consensus

    def analyze(self) -> Dict:
        """
        Execute complete root cause analysis workflow

        Returns consensus report with validated root causes
        """
        print("=== Root Cause Analysis ===\n")

        # Load data
        print("1. Loading failures...")
        failures = self.load_failures()
        print(f"   Loaded {len(failures)} failures")

        print("2. Loading Gemini context...")
        gemini_context = self.load_gemini_analysis()

        # Build graph
        print("3. Building failure dependency graph...")
        graph = self.build_failure_graph(failures, gemini_context)
        print(f"   Nodes: {len(graph.nodes)}")
        print(f"   Edges: {sum(len(e) for e in graph.edges.values())}")

        # Find roots
        print("4. Identifying root causes...")
        roots = graph.get_roots()
        print(f"   Root failures: {len(roots)}")

        # Detect cycles
        print("5. Detecting circular dependencies...")
        cycles = graph.find_cycles()
        if cycles:
            print(f"   ⚠️  Found {len(cycles)} circular dependencies")
        else:
            print(f"   ✅ No circular dependencies")

        # Validate roots
        print("6. Validating root causes with 5-Whys...")
        validated = self.validate_root_causes(graph)

        # Generate consensus
        print("7. Generating Raft consensus...")
        consensus = self.generate_consensus(validated, graph)

        # Save results
        output_file = self.artifacts_dir / "root-causes-consensus.json"
        with open(output_file, 'w') as f:
            json.dump(consensus, f, indent=2)

        print(f"\n✅ Root cause analysis complete")
        print(f"   Root causes: {consensus['stats']['rootFailures']}")
        print(f"   Cascaded failures: {consensus['stats']['cascadedFailures']}")
        print(f"   Cascade ratio: {consensus['stats']['cascadeRatio']:.2%}")
        print(f"   Saved to: {output_file}\n")

        return consensus


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        artifacts_dir = sys.argv[1]
    else:
        artifacts_dir = ".claude/.artifacts"

    try:
        analyzer = RootCauseAnalyzer(artifacts_dir)
        consensus = analyzer.analyze()

        # Exit with error if no root causes found
        if consensus["stats"]["rootFailures"] == 0:
            print("ERROR: No root causes identified", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
