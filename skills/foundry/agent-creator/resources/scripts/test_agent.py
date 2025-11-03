#!/usr/bin/env python3
"""
Agent Testing Framework
Test agent system prompts against typical cases, edge cases, and integration scenarios

Usage:
    python test_agent.py --agent marketing-specialist --test-suite basic
    python test_agent.py --agent backend-dev --test-suite comprehensive
    python test_agent.py --agent all --generate-report
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess

class AgentTester:
    """Test framework for agent system prompts."""

    def __init__(self, agent_name: str, prompt_file: Path):
        self.agent_name = agent_name
        self.prompt_file = prompt_file
        self.test_results = {
            "agent_name": agent_name,
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }

    def run_test_suite(self, suite: str = "basic") -> Dict[str, Any]:
        """Run specified test suite."""
        print(f"\n{'='*70}")
        print(f"AGENT TESTING FRAMEWORK")
        print(f"Agent: {self.agent_name}")
        print(f"Suite: {suite}")
        print(f"{'='*70}\n")

        if suite == "basic":
            self._run_basic_tests()
        elif suite == "comprehensive":
            self._run_comprehensive_tests()
        elif suite == "integration":
            self._run_integration_tests()
        else:
            print(f"Unknown test suite: {suite}")
            return self.test_results

        self._calculate_summary()
        return self.test_results

    def _run_basic_tests(self):
        """Run basic validation tests."""
        print("1. BASIC VALIDATION TESTS")
        print("-" * 70)

        # Test 1: Identity Consistency
        self._test_identity_consistency()

        # Test 2: Command Coverage
        self._test_command_coverage()

        # Test 3: Evidence-Based Patterns
        self._test_evidence_patterns()

        # Test 4: Structural Quality
        self._test_structural_quality()

    def _run_comprehensive_tests(self):
        """Run comprehensive test suite."""
        self._run_basic_tests()

        print("\n2. EDGE CASE TESTS")
        print("-" * 70)

        # Test 5: Edge Case Handling
        self._test_edge_cases()

        # Test 6: Error Handling
        self._test_error_handling()

        # Test 7: Workflow Completeness
        self._test_workflow_completeness()

    def _run_integration_tests(self):
        """Run integration tests."""
        print("\n3. INTEGRATION TESTS")
        print("-" * 70)

        # Test 8: MCP Integration
        self._test_mcp_integration()

        # Test 9: Cross-Agent Coordination
        self._test_cross_agent_coordination()

        # Test 10: Memory Patterns
        self._test_memory_patterns()

    # Individual test methods

    def _test_identity_consistency(self):
        """Test that agent maintains consistent role identity."""
        test_name = "identity_consistency"
        print(f"\nTest: {test_name}")

        # Load prompt
        with open(self.prompt_file) as f:
            content = f.read()

        checks = {
            "has_core_identity": "## 🎭 CORE IDENTITY" in content,
            "has_role_statement": "I am a **" in content,
            "has_expertise_domains": "- **" in content and "** -" in content,
            "mentions_purpose": "My purpose is to" in content
        }

        passed = all(checks.values())
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "checks": checks,
            "score": sum(checks.values()) / len(checks) * 100
        }

        result = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {result}")
        for check, status in checks.items():
            print(f"    {check}: {'✓' if status else '✗'}")

    def _test_command_coverage(self):
        """Test that essential commands are documented."""
        test_name = "command_coverage"
        print(f"\nTest: {test_name}")

        with open(self.prompt_file) as f:
            content = f.read()

        required_commands = [
            "/file-read", "/file-write",
            "/git-status", "/git-commit",
            "/memory-store", "/memory-retrieve"
        ]

        found_commands = sum(1 for cmd in required_commands if cmd in content)
        coverage_percent = (found_commands / len(required_commands)) * 100

        passed = coverage_percent >= 80
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "found_commands": found_commands,
            "total_commands": len(required_commands),
            "coverage_percent": coverage_percent
        }

        result = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {result}")
        print(f"  Coverage: {found_commands}/{len(required_commands)} ({coverage_percent:.1f}%)")

    def _test_evidence_patterns(self):
        """Test for evidence-based prompting patterns."""
        test_name = "evidence_patterns"
        print(f"\nTest: {test_name}")

        with open(self.prompt_file) as f:
            content = f.read()

        patterns = {
            "self_consistency": "Self-Consistency" in content,
            "program_of_thought": "Program-of-Thought" in content or "Decomposition" in content,
            "plan_and_solve": "Plan-and-Solve" in content,
            "guardrails": "## 🚧 GUARDRAILS" in content,
            "success_criteria": "## ✅ SUCCESS CRITERIA" in content
        }

        passed = sum(patterns.values()) >= 3
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "patterns_found": patterns,
            "score": sum(patterns.values()) / len(patterns) * 100
        }

        result = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {result}")
        for pattern, found in patterns.items():
            print(f"    {pattern}: {'✓' if found else '✗'}")

    def _test_structural_quality(self):
        """Test structural quality and organization."""
        test_name = "structural_quality"
        print(f"\nTest: {test_name}")

        with open(self.prompt_file) as f:
            lines = f.readlines()

        # Count sections
        section_headers = [line for line in lines if line.startswith("##")]

        # Check for examples
        has_workflow_examples = any("WORKFLOW EXAMPLES" in line for line in lines)
        example_count = sum(1 for line in lines if line.startswith("###"))

        # Check for code blocks
        code_blocks = sum(1 for line in lines if line.strip() == "```")

        quality_checks = {
            "has_multiple_sections": len(section_headers) >= 5,
            "has_workflow_examples": has_workflow_examples,
            "has_sufficient_examples": example_count >= 2,
            "has_code_examples": code_blocks >= 2
        }

        passed = sum(quality_checks.values()) >= 3
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "quality_checks": quality_checks,
            "section_count": len(section_headers),
            "example_count": example_count,
            "code_block_count": code_blocks // 2
        }

        result = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {result}")
        print(f"  Sections: {len(section_headers)}")
        print(f"  Examples: {example_count}")
        print(f"  Code blocks: {code_blocks // 2}")

    def _test_edge_cases(self):
        """Test edge case documentation."""
        test_name = "edge_case_handling"
        print(f"\nTest: {test_name}")

        with open(self.prompt_file) as f:
            content = f.read().lower()

        edge_case_indicators = [
            "edge case", "failure mode", "error handling",
            "fallback", "when to escalate", "boundary condition"
        ]

        found = sum(1 for indicator in edge_case_indicators if indicator in content)
        passed = found >= 2

        self.test_results["tests"][test_name] = {
            "passed": passed,
            "indicators_found": found,
            "total_indicators": len(edge_case_indicators)
        }

        result = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {result}")
        print(f"  Edge case indicators: {found}/{len(edge_case_indicators)}")

    def _test_error_handling(self):
        """Test error handling patterns."""
        test_name = "error_handling"
        print(f"\nTest: {test_name}")

        with open(self.prompt_file) as f:
            content = f.read()

        error_patterns = {
            "has_never_statements": "❌ NEVER:" in content,
            "has_validation": "VALIDATE" in content or "verify" in content.lower(),
            "has_escalation": "escalate" in content.lower(),
            "has_retry_logic": "retry" in content.lower() or "fallback" in content.lower()
        }

        passed = sum(error_patterns.values()) >= 2
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "error_patterns": error_patterns
        }

        result = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {result}")
        for pattern, found in error_patterns.items():
            print(f"    {pattern}: {'✓' if found else '✗'}")

    def _test_workflow_completeness(self):
        """Test workflow documentation completeness."""
        test_name = "workflow_completeness"
        print(f"\nTest: {test_name}")

        with open(self.prompt_file) as f:
            content = f.read()

        workflow_elements = {
            "has_workflow_section": "## 📖 WORKFLOW EXAMPLES" in content,
            "has_workflow_names": "### Workflow" in content,
            "has_step_by_step": "Step 1:" in content or "1." in content,
            "has_commands": "COMMANDS:" in content or "```" in content,
            "has_validation_steps": "VALIDATION:" in content or "verify" in content.lower()
        }

        passed = sum(workflow_elements.values()) >= 4
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "workflow_elements": workflow_elements
        }

        result = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {result}")
        for element, found in workflow_elements.items():
            print(f"    {element}: {'✓' if found else '✗'}")

    def _test_mcp_integration(self):
        """Test MCP tool integration patterns."""
        test_name = "mcp_integration"
        print(f"\nTest: {test_name}")

        with open(self.prompt_file) as f:
            content = f.read()

        mcp_checks = {
            "has_mcp_section": "## 🔧 MCP SERVER TOOLS" in content,
            "uses_claude_flow": "mcp__claude-flow__" in content,
            "documents_when_how": "WHEN:" in content and "HOW:" in content,
            "has_namespace_pattern": self.agent_name in content or "/task-id/" in content
        }

        passed = sum(mcp_checks.values()) >= 3
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "mcp_checks": mcp_checks
        }

        result = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {result}")
        for check, status in mcp_checks.items():
            print(f"    {check}: {'✓' if status else '✗'}")

    def _test_cross_agent_coordination(self):
        """Test cross-agent coordination patterns."""
        test_name = "cross_agent_coordination"
        print(f"\nTest: {test_name}")

        with open(self.prompt_file) as f:
            content = f.read().lower()

        coordination_patterns = [
            "agent-delegate", "agent-escalate", "memory-store",
            "coordination", "collaborate", "notify"
        ]

        found = sum(1 for pattern in coordination_patterns if pattern in content)
        passed = found >= 3

        self.test_results["tests"][test_name] = {
            "passed": passed,
            "patterns_found": found,
            "total_patterns": len(coordination_patterns)
        }

        result = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {result}")
        print(f"  Coordination patterns: {found}/{len(coordination_patterns)}")

    def _test_memory_patterns(self):
        """Test memory usage patterns."""
        test_name = "memory_patterns"
        print(f"\nTest: {test_name}")

        with open(self.prompt_file) as f:
            content = f.read()

        memory_checks = {
            "has_memory_commands": "/memory-store" in content or "/memory-retrieve" in content,
            "has_namespace_pattern": "/" in content and "task-id" in content.lower(),
            "documents_memory_usage": "memory" in content.lower() and ("when:" in content.lower() or "how:" in content.lower()),
            "stores_results": "store" in content.lower() and ("memory" in content.lower() or "results" in content.lower())
        }

        passed = sum(memory_checks.values()) >= 3
        self.test_results["tests"][test_name] = {
            "passed": passed,
            "memory_checks": memory_checks
        }

        result = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Result: {result}")
        for check, status in memory_checks.items():
            print(f"    {check}: {'✓' if status else '✗'}")

    def _calculate_summary(self):
        """Calculate overall test summary."""
        total_tests = len(self.test_results["tests"])
        passed_tests = sum(1 for test in self.test_results["tests"].values() if test["passed"])

        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }

    def save_report(self, output_file: Path):
        """Save test report to file."""
        with open(output_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        print(f"\n✓ Test report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Agent Testing Framework")
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--prompt-file", type=Path, help="Path to system prompt file")
    parser.add_argument("--test-suite", choices=["basic", "comprehensive", "integration"],
                       default="basic", help="Test suite to run")
    parser.add_argument("--output", type=Path, help="Output file for test report")

    args = parser.parse_args()

    # Determine prompt file
    if args.prompt_file is None:
        # Try to find prompt file in standard location
        prompt_file = Path(f"./agent-outputs/{args.agent}/{args.agent}-base-prompt-v1.md")
        if not prompt_file.exists():
            prompt_file = Path(f"./agent-outputs/{args.agent}/{args.agent}-enhanced-prompt-v2.md")
        if not prompt_file.exists():
            print(f"Error: Could not find prompt file for agent '{args.agent}'")
            print("Please specify --prompt-file")
            sys.exit(1)
    else:
        prompt_file = args.prompt_file

    if not prompt_file.exists():
        print(f"Error: Prompt file not found: {prompt_file}")
        sys.exit(1)

    # Run tests
    tester = AgentTester(args.agent, prompt_file)
    results = tester.run_test_suite(args.test_suite)

    # Print summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1f}%")

    # Save report
    if args.output:
        output_file = args.output
    else:
        output_file = Path(f"./agent-outputs/{args.agent}/{args.agent}-test-report.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

    tester.save_report(output_file)

    # Exit with appropriate code
    if results['summary']['success_rate'] >= 80:
        print("\n✓ TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
