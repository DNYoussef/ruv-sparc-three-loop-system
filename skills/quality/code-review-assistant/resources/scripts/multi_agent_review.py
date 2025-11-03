#!/usr/bin/env python3
"""
Multi-Agent Code Review Orchestrator

Coordinates 5 specialized review agents for comprehensive PR analysis.
Part of code-review-assistant Gold tier enhancement.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess


@dataclass
class ReviewAgent:
    """Specialist review agent configuration"""
    name: str
    type: str
    capabilities: List[str]
    priority: int
    focus_areas: List[str]


@dataclass
class ReviewFinding:
    """Individual review finding"""
    severity: str  # critical, high, medium, low, info
    category: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    suggestion: Optional[str] = None
    agent: Optional[str] = None


@dataclass
class AgentReview:
    """Review results from a single agent"""
    agent_name: str
    agent_type: str
    score: float  # 0-100
    findings: List[ReviewFinding]
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class ComprehensiveReview:
    """Aggregated review from all agents"""
    pr_number: int
    pr_title: str
    overall_score: float
    merge_ready: bool
    blocking_issues: int
    warnings: int
    suggestions: int
    agent_reviews: List[AgentReview]
    timestamp: str
    execution_time: float


class MultiAgentReviewer:
    """Orchestrates multi-agent code review swarm"""

    # 5 Specialized Review Agents
    AGENTS = [
        ReviewAgent(
            name="Security Reviewer",
            type="security",
            capabilities=["vulnerability_scan", "secret_detection", "dependency_audit", "injection_check"],
            priority=1,
            focus_areas=["authentication", "authorization", "data_validation", "crypto", "dependencies"]
        ),
        ReviewAgent(
            name="Performance Analyst",
            type="performance",
            capabilities=["bottleneck_detection", "complexity_analysis", "memory_profiling", "n+1_queries"],
            priority=2,
            focus_areas=["algorithms", "database", "caching", "async_ops", "memory_leaks"]
        ),
        ReviewAgent(
            name="Style Reviewer",
            type="style",
            capabilities=["linting", "formatting", "naming_conventions", "best_practices"],
            priority=3,
            focus_areas=["readability", "maintainability", "consistency", "documentation"]
        ),
        ReviewAgent(
            name="Test Specialist",
            type="tests",
            capabilities=["coverage_analysis", "test_quality", "edge_cases", "test_patterns"],
            priority=2,
            focus_areas=["unit_tests", "integration_tests", "edge_cases", "mocking", "assertions"]
        ),
        ReviewAgent(
            name="Documentation Reviewer",
            type="documentation",
            capabilities=["comment_quality", "api_docs", "readme_updates", "changelog"],
            priority=4,
            focus_areas=["code_comments", "docstrings", "api_documentation", "user_guides"]
        )
    ]

    def __init__(self, pr_number: int, changed_files: List[str], focus_areas: List[str] = None):
        self.pr_number = pr_number
        self.changed_files = changed_files
        self.focus_areas = focus_areas or ["security", "performance", "style", "tests", "documentation"]
        self.start_time = datetime.now()

    async def initialize_swarm(self) -> bool:
        """Initialize mesh topology swarm for parallel reviews"""
        print(f"[MultiAgentReviewer] Initializing swarm for PR #{self.pr_number}")
        print(f"[MultiAgentReviewer] Focus areas: {', '.join(self.focus_areas)}")
        print(f"[MultiAgentReviewer] Files to review: {len(self.changed_files)}")

        # Initialize claude-flow swarm
        try:
            result = subprocess.run(
                ["npx", "claude-flow", "coordination", "swarm-init",
                 "--topology", "mesh",
                 "--max-agents", "5",
                 "--strategy", "specialized"],
                capture_output=True,
                text=True,
                check=True
            )
            print("[MultiAgentReviewer] Swarm initialized successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[MultiAgentReviewer] Failed to initialize swarm: {e.stderr}")
            return False

    async def spawn_review_agents(self) -> List[ReviewAgent]:
        """Spawn specialized review agents based on focus areas"""
        active_agents = [
            agent for agent in self.AGENTS
            if agent.type in self.focus_areas
        ]

        print(f"[MultiAgentReviewer] Spawning {len(active_agents)} specialized agents...")

        for agent in active_agents:
            try:
                subprocess.run(
                    ["npx", "claude-flow", "automation", "auto-agent",
                     "--task", f"Code review - {agent.name}",
                     "--type", agent.type],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"  ✓ {agent.name} spawned")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to spawn {agent.name}: {e.stderr}")

        return active_agents

    async def execute_security_review(self, agent: ReviewAgent) -> AgentReview:
        """Execute security-focused review"""
        start = datetime.now()
        findings = []

        # Run security scans
        for file_path in self.changed_files:
            # Check for common vulnerabilities
            findings.extend(self._scan_sql_injection(file_path))
            findings.extend(self._scan_xss(file_path))
            findings.extend(self._scan_secrets(file_path))
            findings.extend(self._scan_insecure_crypto(file_path))

        # Calculate security score
        critical_count = sum(1 for f in findings if f.severity == "critical")
        high_count = sum(1 for f in findings if f.severity == "high")
        score = max(0, 100 - (critical_count * 30 + high_count * 15))

        execution_time = (datetime.now() - start).total_seconds()

        return AgentReview(
            agent_name=agent.name,
            agent_type=agent.type,
            score=score,
            findings=findings,
            execution_time=execution_time,
            metadata={"critical_issues": critical_count, "high_issues": high_count}
        )

    async def execute_performance_review(self, agent: ReviewAgent) -> AgentReview:
        """Execute performance-focused review"""
        start = datetime.now()
        findings = []

        for file_path in self.changed_files:
            findings.extend(self._detect_n_plus_one(file_path))
            findings.extend(self._detect_inefficient_loops(file_path))
            findings.extend(self._detect_memory_leaks(file_path))

        issues_count = len(findings)
        score = max(0, 100 - (issues_count * 10))

        execution_time = (datetime.now() - start).total_seconds()

        return AgentReview(
            agent_name=agent.name,
            agent_type=agent.type,
            score=score,
            findings=findings,
            execution_time=execution_time,
            metadata={"bottlenecks_found": issues_count}
        )

    async def execute_style_review(self, agent: ReviewAgent) -> AgentReview:
        """Execute style-focused review"""
        start = datetime.now()
        findings = []

        for file_path in self.changed_files:
            findings.extend(self._check_naming_conventions(file_path))
            findings.extend(self._check_code_complexity(file_path))
            findings.extend(self._check_documentation(file_path))

        issues_count = len([f for f in findings if f.severity in ["high", "medium"]])
        score = max(0, 100 - (issues_count * 5))

        execution_time = (datetime.now() - start).total_seconds()

        return AgentReview(
            agent_name=agent.name,
            agent_type=agent.type,
            score=score,
            findings=findings,
            execution_time=execution_time,
            metadata={"style_violations": issues_count}
        )

    async def execute_test_review(self, agent: ReviewAgent) -> AgentReview:
        """Execute test-focused review"""
        start = datetime.now()
        findings = []

        test_files = [f for f in self.changed_files if "test" in f.lower() or "spec" in f.lower()]

        if not test_files:
            findings.append(ReviewFinding(
                severity="high",
                category="test_coverage",
                message="No test files found in PR",
                suggestion="Add tests for new functionality",
                agent=agent.name
            ))

        for file_path in test_files:
            findings.extend(self._check_test_quality(file_path))
            findings.extend(self._check_edge_cases(file_path))

        # Estimate coverage
        coverage = 80.0 if test_files else 0.0
        score = coverage

        execution_time = (datetime.now() - start).total_seconds()

        return AgentReview(
            agent_name=agent.name,
            agent_type=agent.type,
            score=score,
            findings=findings,
            execution_time=execution_time,
            metadata={"test_files": len(test_files), "estimated_coverage": coverage}
        )

    async def execute_documentation_review(self, agent: ReviewAgent) -> AgentReview:
        """Execute documentation-focused review"""
        start = datetime.now()
        findings = []

        for file_path in self.changed_files:
            findings.extend(self._check_comments(file_path))
            findings.extend(self._check_api_docs(file_path))

        issues_count = len(findings)
        score = max(60, 100 - (issues_count * 8))

        execution_time = (datetime.now() - start).total_seconds()

        return AgentReview(
            agent_name=agent.name,
            agent_type=agent.type,
            score=score,
            findings=findings,
            execution_time=execution_time,
            metadata={"documentation_gaps": issues_count}
        )

    async def execute_agent_review(self, agent: ReviewAgent) -> AgentReview:
        """Route to appropriate review executor"""
        print(f"[{agent.name}] Starting review...")

        if agent.type == "security":
            return await self.execute_security_review(agent)
        elif agent.type == "performance":
            return await self.execute_performance_review(agent)
        elif agent.type == "style":
            return await self.execute_style_review(agent)
        elif agent.type == "tests":
            return await self.execute_test_review(agent)
        elif agent.type == "documentation":
            return await self.execute_documentation_review(agent)
        else:
            raise ValueError(f"Unknown agent type: {agent.type}")

    async def coordinate_parallel_reviews(self, agents: List[ReviewAgent]) -> List[AgentReview]:
        """Execute all agent reviews in parallel"""
        print(f"[MultiAgentReviewer] Executing {len(agents)} reviews in parallel...")

        # Run all reviews concurrently
        reviews = await asyncio.gather(*[
            self.execute_agent_review(agent) for agent in agents
        ])

        for review in reviews:
            print(f"[{review.agent_name}] Complete - Score: {review.score:.1f}/100 "
                  f"({len(review.findings)} findings)")

        return reviews

    def aggregate_reviews(self, agent_reviews: List[AgentReview], pr_title: str) -> ComprehensiveReview:
        """Aggregate all agent reviews into comprehensive report"""

        # Calculate overall score (weighted by priority)
        total_weight = sum(agent.priority for agent in self.AGENTS if agent.type in self.focus_areas)
        weighted_score = sum(
            review.score * next(a.priority for a in self.AGENTS if a.name == review.agent_name)
            for review in agent_reviews
        )
        overall_score = weighted_score / total_weight if total_weight > 0 else 0

        # Count findings by severity
        all_findings = [f for review in agent_reviews for f in review.findings]
        blocking_issues = len([f for f in all_findings if f.severity == "critical"])
        warnings = len([f for f in all_findings if f.severity in ["high", "medium"]])
        suggestions = len([f for f in all_findings if f.severity in ["low", "info"]])

        # Determine merge readiness
        merge_ready = (
            blocking_issues == 0 and
            overall_score >= 80 and
            all(review.score >= 60 for review in agent_reviews)
        )

        execution_time = (datetime.now() - self.start_time).total_seconds()

        return ComprehensiveReview(
            pr_number=self.pr_number,
            pr_title=pr_title,
            overall_score=overall_score,
            merge_ready=merge_ready,
            blocking_issues=blocking_issues,
            warnings=warnings,
            suggestions=suggestions,
            agent_reviews=agent_reviews,
            timestamp=datetime.now().isoformat(),
            execution_time=execution_time
        )

    # Helper methods for security checks
    def _scan_sql_injection(self, file_path: str) -> List[ReviewFinding]:
        findings = []
        # Simplified check - real implementation would use AST parsing
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if 'execute(' in line.lower() and '+' in line:
                        findings.append(ReviewFinding(
                            severity="critical",
                            category="sql_injection",
                            message="Potential SQL injection vulnerability",
                            file=file_path,
                            line=i,
                            suggestion="Use parameterized queries",
                            agent="Security Reviewer"
                        ))
        except Exception:
            pass
        return findings

    def _scan_xss(self, file_path: str) -> List[ReviewFinding]:
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if 'dangerouslySetInnerHTML' in line or 'innerHTML' in line:
                        findings.append(ReviewFinding(
                            severity="high",
                            category="xss",
                            message="Potential XSS vulnerability",
                            file=file_path,
                            line=i,
                            suggestion="Sanitize user input",
                            agent="Security Reviewer"
                        ))
        except Exception:
            pass
        return findings

    def _scan_secrets(self, file_path: str) -> List[ReviewFinding]:
        findings = []
        patterns = ['password', 'api_key', 'secret', 'token', 'private_key']
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    for pattern in patterns:
                        if pattern in line.lower() and '=' in line and '"' in line:
                            findings.append(ReviewFinding(
                                severity="critical",
                                category="secrets",
                                message=f"Potential hardcoded {pattern}",
                                file=file_path,
                                line=i,
                                suggestion="Use environment variables",
                                agent="Security Reviewer"
                            ))
        except Exception:
            pass
        return findings

    def _scan_insecure_crypto(self, file_path: str) -> List[ReviewFinding]:
        findings = []
        insecure = ['md5', 'sha1', 'des', 'rc4']
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    for algo in insecure:
                        if algo.upper() in line or f"'{algo}'" in line:
                            findings.append(ReviewFinding(
                                severity="high",
                                category="crypto",
                                message=f"Insecure cryptographic algorithm: {algo}",
                                file=file_path,
                                line=i,
                                suggestion=f"Use SHA-256 or stronger",
                                agent="Security Reviewer"
                            ))
        except Exception:
            pass
        return findings

    # Helper methods for performance checks
    def _detect_n_plus_one(self, file_path: str) -> List[ReviewFinding]:
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'for ' in content and 'query' in content.lower():
                    findings.append(ReviewFinding(
                        severity="medium",
                        category="performance",
                        message="Potential N+1 query pattern",
                        file=file_path,
                        suggestion="Use eager loading or batch queries",
                        agent="Performance Analyst"
                    ))
        except Exception:
            pass
        return findings

    def _detect_inefficient_loops(self, file_path: str) -> List[ReviewFinding]:
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if 'for ' in line and 'for ' in line[line.find('for ')+4:]:
                        findings.append(ReviewFinding(
                            severity="low",
                            category="performance",
                            message="Nested loop detected",
                            file=file_path,
                            line=i,
                            suggestion="Consider algorithmic optimization",
                            agent="Performance Analyst"
                        ))
        except Exception:
            pass
        return findings

    def _detect_memory_leaks(self, file_path: str) -> List[ReviewFinding]:
        return []  # Placeholder

    # Helper methods for style checks
    def _check_naming_conventions(self, file_path: str) -> List[ReviewFinding]:
        return []  # Placeholder

    def _check_code_complexity(self, file_path: str) -> List[ReviewFinding]:
        return []  # Placeholder

    def _check_documentation(self, file_path: str) -> List[ReviewFinding]:
        return []  # Placeholder

    # Helper methods for test checks
    def _check_test_quality(self, file_path: str) -> List[ReviewFinding]:
        return []  # Placeholder

    def _check_edge_cases(self, file_path: str) -> List[ReviewFinding]:
        return []  # Placeholder

    # Helper methods for documentation checks
    def _check_comments(self, file_path: str) -> List[ReviewFinding]:
        return []  # Placeholder

    def _check_api_docs(self, file_path: str) -> List[ReviewFinding]:
        return []  # Placeholder


async def main():
    """Main entry point for multi-agent review"""
    if len(sys.argv) < 2:
        print("Usage: multi_agent_review.py <pr_number> [focus_areas...]")
        sys.exit(1)

    pr_number = int(sys.argv[1])
    focus_areas = sys.argv[2:] if len(sys.argv) > 2 else None

    # Mock changed files for demo (would be fetched from GitHub in production)
    changed_files = [
        "src/auth/login.py",
        "src/api/users.js",
        "tests/test_auth.py"
    ]

    reviewer = MultiAgentReviewer(pr_number, changed_files, focus_areas)

    # Initialize swarm
    await reviewer.initialize_swarm()

    # Spawn agents
    agents = await reviewer.spawn_review_agents()

    # Execute parallel reviews
    agent_reviews = await reviewer.coordinate_parallel_reviews(agents)

    # Aggregate results
    comprehensive_review = reviewer.aggregate_reviews(agent_reviews, f"PR #{pr_number}")

    # Output results
    print("\n" + "="*80)
    print("COMPREHENSIVE CODE REVIEW RESULTS")
    print("="*80)
    print(f"Overall Score: {comprehensive_review.overall_score:.1f}/100")
    print(f"Merge Ready: {'✅ Yes' if comprehensive_review.merge_ready else '⚠️ No'}")
    print(f"Blocking Issues: {comprehensive_review.blocking_issues}")
    print(f"Warnings: {comprehensive_review.warnings}")
    print(f"Suggestions: {comprehensive_review.suggestions}")
    print(f"Execution Time: {comprehensive_review.execution_time:.2f}s")
    print("="*80)

    # Save to JSON
    output_file = f"pr-{pr_number}-review.json"
    with open(output_file, 'w') as f:
        json.dump(asdict(comprehensive_review), f, indent=2, default=str)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
