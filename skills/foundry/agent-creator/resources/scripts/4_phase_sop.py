#!/usr/bin/env python3
"""
4-Phase Agent Creation SOP Script
Automates the official 4-phase agent creation methodology from Desktop .claude-flow

Usage:
    python 4_phase_sop.py --agent-name "marketing-specialist" --mode interactive
    python 4_phase_sop.py --agent-name "backend-dev" --mode batch --input spec.yaml
    python 4_phase_sop.py --phase 1 --output phase1-analysis.json
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# ============================================================================
# PHASE 1: INITIAL ANALYSIS & INTENT DECODING (30-60 minutes)
# ============================================================================

class Phase1Analyzer:
    """Deep domain understanding through systematic research."""

    def __init__(self, agent_name: str, output_dir: Path):
        self.agent_name = agent_name
        self.output_dir = output_dir
        self.validation_checklist = {
            "domain_description": False,
            "key_challenges": False,
            "tech_stack": False,
            "integrations": False
        }

    def domain_breakdown(self) -> Dict[str, Any]:
        """Conduct deep domain analysis."""
        print(f"\n{'='*70}")
        print(f"PHASE 1: INITIAL ANALYSIS & INTENT DECODING")
        print(f"Agent: {self.agent_name}")
        print(f"{'='*70}\n")

        analysis = {
            "agent_name": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "phase": 1,
            "domain_breakdown": {},
            "technology_stack": {},
            "integration_points": {},
            "validation_status": {}
        }

        # Domain Breakdown
        print("1. DOMAIN BREAKDOWN")
        print("-" * 70)
        analysis["domain_breakdown"]["problem_solved"] = input(
            "What problem does this agent solve?\n> "
        )

        challenges = []
        print("\nWhat are the key challenges in this domain?")
        for i in range(5):
            challenge = input(f"  Challenge {i+1}: ")
            if challenge:
                challenges.append(challenge)
        analysis["domain_breakdown"]["key_challenges"] = challenges

        patterns = []
        print("\nWhat patterns do human experts use?")
        for i in range(3):
            pattern = input(f"  Pattern {i+1}: ")
            if pattern:
                patterns.append(pattern)
        analysis["domain_breakdown"]["expert_patterns"] = patterns

        failure_modes = []
        print("\nWhat are common failure modes?")
        for i in range(3):
            mode = input(f"  Failure mode {i+1}: ")
            if mode:
                failure_modes.append(mode)
        analysis["domain_breakdown"]["failure_modes"] = failure_modes

        # Technology Stack Mapping
        print("\n2. TECHNOLOGY STACK MAPPING")
        print("-" * 70)
        analysis["technology_stack"]["tools_frameworks"] = input(
            "What tools, frameworks, libraries are used?\n> "
        ).split(", ")

        analysis["technology_stack"]["file_types"] = input(
            "What file types, formats, protocols?\n> "
        ).split(", ")

        analysis["technology_stack"]["integrations"] = input(
            "What integrations or APIs?\n> "
        ).split(", ")

        analysis["technology_stack"]["config_patterns"] = input(
            "What configuration patterns?\n> "
        ).split(", ")

        # Integration Points
        print("\n3. INTEGRATION POINTS")
        print("-" * 70)
        analysis["integration_points"]["mcp_servers"] = input(
            "What MCP servers will this agent use?\n> "
        ).split(", ")

        analysis["integration_points"]["coordinating_agents"] = input(
            "What other agents will it coordinate with?\n> "
        ).split(", ")

        analysis["integration_points"]["data_flows"] = input(
            "What data flows in/out?\n> "
        )

        analysis["integration_points"]["memory_patterns"] = input(
            "What memory patterns needed?\n> "
        )

        # Validation Gate
        self._validate_phase1(analysis)
        analysis["validation_status"] = self.validation_checklist

        return analysis

    def _validate_phase1(self, analysis: Dict) -> bool:
        """Validate Phase 1 completion."""
        print("\n4. VALIDATION GATE")
        print("-" * 70)

        # Check domain description
        if analysis["domain_breakdown"]["problem_solved"]:
            self.validation_checklist["domain_description"] = True
            print("✓ Can describe domain in specific, technical terms")
        else:
            print("✗ Domain description incomplete")

        # Check key challenges
        if len(analysis["domain_breakdown"]["key_challenges"]) >= 5:
            self.validation_checklist["key_challenges"] = True
            print("✓ Identified 5+ key challenges")
        else:
            print(f"✗ Only {len(analysis['domain_breakdown']['key_challenges'])} challenges identified (need 5+)")

        # Check tech stack
        if len(analysis["technology_stack"]["tools_frameworks"]) >= 3:
            self.validation_checklist["tech_stack"] = True
            print("✓ Mapped technology stack comprehensively")
        else:
            print("✗ Technology stack incomplete")

        # Check integrations
        if analysis["integration_points"]["mcp_servers"]:
            self.validation_checklist["integrations"] = True
            print("✓ Clear on integration requirements")
        else:
            print("✗ Integration requirements unclear")

        return all(self.validation_checklist.values())

    def save_output(self, analysis: Dict):
        """Save Phase 1 analysis to file."""
        output_file = self.output_dir / f"{self.agent_name}-phase1-analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✓ Phase 1 analysis saved to: {output_file}")
        return output_file


# ============================================================================
# PHASE 2: META-COGNITIVE EXTRACTION (30-45 minutes)
# ============================================================================

class Phase2Extractor:
    """Identify cognitive expertise domains and create agent specification."""

    def __init__(self, agent_name: str, phase1_data: Dict, output_dir: Path):
        self.agent_name = agent_name
        self.phase1_data = phase1_data
        self.output_dir = output_dir
        self.validation_checklist = {
            "expertise_domains": False,
            "decision_heuristics": False,
            "agent_spec": False,
            "examples": False
        }

    def extract_expertise(self) -> Dict[str, Any]:
        """Extract meta-cognitive expertise domains."""
        print(f"\n{'='*70}")
        print(f"PHASE 2: META-COGNITIVE EXTRACTION")
        print(f"Agent: {self.agent_name}")
        print(f"{'='*70}\n")

        extraction = {
            "agent_name": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "phase": 2,
            "expertise_domains": [],
            "decision_frameworks": [],
            "agent_specification": {},
            "examples": {},
            "validation_status": {}
        }

        # Expertise Domain Identification
        print("1. EXPERTISE DOMAIN IDENTIFICATION")
        print("-" * 70)
        print("What knowledge domains are activated when you think about this role?")
        for i in range(5):
            domain = input(f"  Domain {i+1}: ")
            if domain:
                extraction["expertise_domains"].append(domain)

        # Decision Frameworks
        print("\n2. DECISION FRAMEWORKS")
        print("-" * 70)
        heuristics = []
        print("What heuristics, patterns, rules-of-thumb?")
        for i in range(5):
            heuristic = input(f"  Heuristic {i+1} (When X, do Y because Z): ")
            if heuristic:
                heuristics.append(heuristic)
        extraction["decision_frameworks"] = heuristics

        # Agent Specification Creation
        print("\n3. AGENT SPECIFICATION")
        print("-" * 70)
        spec = {
            "role": input("Primary role (Specific title): "),
            "expertise_domains": extraction["expertise_domains"],
            "cognitive_patterns": heuristics,
            "core_capabilities": [],
            "quality_standards": {}
        }

        print("\nCore Capabilities (with specific examples):")
        for i in range(5):
            capability = input(f"  Capability {i+1}: ")
            if capability:
                spec["core_capabilities"].append(capability)

        spec["quality_standards"]["output_criteria"] = input(
            "\nOutput must meet (criteria): "
        )
        spec["quality_standards"]["performance_metrics"] = input(
            "Performance measured by (metrics): "
        )

        failure_prevention = []
        print("\nFailure modes to prevent:")
        for i in range(3):
            mode = input(f"  Failure mode {i+1}: ")
            if mode:
                failure_prevention.append(mode)
        spec["quality_standards"]["failure_prevention"] = failure_prevention

        extraction["agent_specification"] = spec

        # Supporting Artifacts
        print("\n4. SUPPORTING ARTIFACTS")
        print("-" * 70)
        extraction["examples"]["good_output"] = input("Example of good output: ")
        extraction["examples"]["bad_output"] = input("Example of bad output: ")

        edge_cases = []
        print("\nEdge cases to document:")
        for i in range(3):
            case = input(f"  Edge case {i+1}: ")
            if case:
                edge_cases.append(case)
        extraction["examples"]["edge_cases"] = edge_cases

        # Validation Gate
        self._validate_phase2(extraction)
        extraction["validation_status"] = self.validation_checklist

        return extraction

    def _validate_phase2(self, extraction: Dict) -> bool:
        """Validate Phase 2 completion."""
        print("\n5. VALIDATION GATE")
        print("-" * 70)

        if len(extraction["expertise_domains"]) >= 3:
            self.validation_checklist["expertise_domains"] = True
            print("✓ Identified 3+ expertise domains")
        else:
            print(f"✗ Only {len(extraction['expertise_domains'])} domains (need 3+)")

        if len(extraction["decision_frameworks"]) >= 5:
            self.validation_checklist["decision_heuristics"] = True
            print("✓ Documented 5+ decision heuristics")
        else:
            print(f"✗ Only {len(extraction['decision_frameworks'])} heuristics (need 5+)")

        if extraction["agent_specification"]["role"]:
            self.validation_checklist["agent_spec"] = True
            print("✓ Created complete agent specification")
        else:
            print("✗ Agent specification incomplete")

        if extraction["examples"]["good_output"] and extraction["examples"]["bad_output"]:
            self.validation_checklist["examples"] = True
            print("✓ Examples demonstrate quality standards")
        else:
            print("✗ Examples incomplete")

        return all(self.validation_checklist.values())

    def save_output(self, extraction: Dict):
        """Save Phase 2 extraction to file."""
        output_file = self.output_dir / f"{self.agent_name}-phase2-extraction.json"
        with open(output_file, 'w') as f:
            json.dump(extraction, f, indent=2)
        print(f"\n✓ Phase 2 extraction saved to: {output_file}")

        # Also create markdown spec
        spec_file = self.output_dir / f"{self.agent_name}-specification.md"
        self._create_spec_markdown(extraction, spec_file)
        return output_file

    def _create_spec_markdown(self, extraction: Dict, output_file: Path):
        """Create markdown agent specification."""
        spec = extraction["agent_specification"]
        content = f"""# Agent Specification: {self.agent_name}

## Role & Expertise
- **Primary role**: {spec['role']}
- **Expertise domains**: {', '.join(spec['expertise_domains'])}
- **Cognitive patterns**:
{chr(10).join(f"  - {h}" for h in spec['cognitive_patterns'])}

## Core Capabilities
{chr(10).join(f"{i+1}. {cap}" for i, cap in enumerate(spec['core_capabilities']))}

## Decision Frameworks
{chr(10).join(f"- {fw}" for fw in extraction['decision_frameworks'])}

## Quality Standards
- **Output criteria**: {spec['quality_standards']['output_criteria']}
- **Performance metrics**: {spec['quality_standards']['performance_metrics']}
- **Failure modes to prevent**:
{chr(10).join(f"  - {mode}" for mode in spec['quality_standards']['failure_prevention'])}

## Examples

### Good Output
{extraction['examples']['good_output']}

### Bad Output
{extraction['examples']['bad_output']}

### Edge Cases
{chr(10).join(f"- {case}" for case in extraction['examples']['edge_cases'])}
"""
        with open(output_file, 'w') as f:
            f.write(content)
        print(f"✓ Agent specification saved to: {output_file}")


# ============================================================================
# PHASE 3: AGENT ARCHITECTURE DESIGN (45-60 minutes)
# ============================================================================

class Phase3Architect:
    """Transform specification into production-ready base system prompt."""

    def __init__(self, agent_name: str, phase2_data: Dict, output_dir: Path):
        self.agent_name = agent_name
        self.phase2_data = phase2_data
        self.output_dir = output_dir

    def design_architecture(self) -> Dict[str, Any]:
        """Design system prompt architecture."""
        print(f"\n{'='*70}")
        print(f"PHASE 3: AGENT ARCHITECTURE DESIGN")
        print(f"Agent: {self.agent_name}")
        print(f"{'='*70}\n")

        spec = self.phase2_data["agent_specification"]

        print("Generating base system prompt v1.0...")
        print("This uses the template from Phase 3 of the 4-phase SOP.")
        print("\nPress Enter to continue...")
        input()

        # Generate base prompt structure
        base_prompt = self._generate_base_prompt(spec)

        architecture = {
            "agent_name": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "phase": 3,
            "base_prompt_v1": base_prompt,
            "cognitive_framework": self._extract_cognitive_framework(spec),
            "guardrails": self._extract_guardrails(),
            "validation_status": {}
        }

        return architecture

    def _generate_base_prompt(self, spec: Dict) -> str:
        """Generate base system prompt from specification."""
        role = spec["role"]
        domains = spec["expertise_domains"]
        capabilities = spec["core_capabilities"]

        prompt = f"""# {self.agent_name.upper().replace('-', ' ')} - SYSTEM PROMPT v1.0

## 🎭 CORE IDENTITY

I am a **{role}** with comprehensive, deeply-ingrained knowledge of {', '.join(domains[:2])}. Through systematic reverse engineering and domain expertise, I possess precision-level understanding of:

{chr(10).join(f'- **{domain}** - {cap}' for domain, cap in zip(domains, capabilities))}

My purpose is to {spec.get('quality_standards', {}).get('output_criteria', 'deliver high-quality outputs')} by leveraging {domains[0] if domains else 'domain expertise'}.

## 📋 UNIVERSAL COMMANDS I USE

**File Operations**:
- /file-read, /file-write, /glob-search, /grep-search
WHEN: Analyzing or modifying codebase, documentation, or configuration files
HOW: Use glob for pattern matching, grep for content search, read before write

**Git Operations**:
- /git-status, /git-commit, /git-push
WHEN: Tracking changes, committing work, synchronizing with remote
HOW: Always check status first, commit with descriptive messages

**Communication & Coordination**:
- /memory-store, /memory-retrieve
- /agent-delegate, /agent-escalate
WHEN: Cross-agent data sharing, task delegation, complex coordination
HOW: Namespace pattern: {self.agent_name}/task-id/data-type

## 🎯 MY SPECIALIST COMMANDS

[TODO: Add role-specific commands during Phase 4 technical enhancement]

## 🔧 MCP SERVER TOOLS I USE

**Claude Flow MCP**:
- mcp__claude-flow__agent_spawn
  WHEN: Need to coordinate with other agents for parallel work
  HOW: Specify agent type, capabilities, and coordination pattern

- mcp__claude-flow__memory_store
  WHEN: Cross-agent data sharing and persistent state
  HOW: Namespace: {self.agent_name}/task-id/data-type

[TODO: Add domain-specific MCP servers during Phase 4]

## 🧠 COGNITIVE FRAMEWORK

### Self-Consistency Validation
Before finalizing deliverables, I validate from multiple angles:
1. Technical correctness against domain standards
2. Completeness of requirements coverage
3. Quality criteria satisfaction

### Program-of-Thought Decomposition
For complex tasks, I decompose BEFORE execution:
1. Break down into logical subtasks
2. Identify dependencies between steps
3. Assess risks and edge cases

### Plan-and-Solve Execution
My standard workflow:
1. PLAN: Analyze requirements and design approach
2. VALIDATE: Check approach against constraints
3. EXECUTE: Implement solution systematically
4. VERIFY: Test and validate results
5. DOCUMENT: Store outcomes in memory

## 🚧 GUARDRAILS - WHAT I NEVER DO

[TODO: Add specific failure modes and guardrails from Phase 2 during Phase 4]

**General Principles**:
❌ NEVER: Skip validation steps
WHY: Leads to errors that cascade through workflow

❌ NEVER: Make assumptions without clarification
WHY: Wastes time implementing wrong solution

## ✅ SUCCESS CRITERIA

Task complete when:
- [ ] Requirements fully satisfied
- [ ] Quality standards met
- [ ] Results validated and tested
- [ ] Outcomes stored in memory
- [ ] Relevant agents notified

## 📖 WORKFLOW EXAMPLES

[TODO: Add 2+ workflow examples with exact commands during Phase 4]

---
Generated: {datetime.now().isoformat()}
Version: 1.0 (Base Architecture)
Next: Phase 4 Technical Enhancement
"""
        return prompt

    def _extract_cognitive_framework(self, spec: Dict) -> Dict[str, Any]:
        """Extract cognitive framework specification."""
        return {
            "self_consistency": [
                "Technical correctness validation",
                "Requirements coverage check",
                "Quality criteria verification"
            ],
            "program_of_thought": [
                "Task decomposition",
                "Dependency analysis",
                "Risk assessment"
            ],
            "plan_and_solve": [
                "PLAN: Requirements analysis",
                "VALIDATE: Constraint checking",
                "EXECUTE: Systematic implementation",
                "VERIFY: Testing and validation",
                "DOCUMENT: Memory storage"
            ]
        }

    def _extract_guardrails(self) -> Dict[str, List[str]]:
        """Extract guardrails from Phase 2 failure modes."""
        failure_modes = self.phase2_data["agent_specification"]["quality_standards"]["failure_prevention"]

        return {
            "never_skip": ["validation steps", "requirement clarification"],
            "never_assume": ["requirements without confirmation", "edge case handling"],
            "always_validate": ["outputs against quality criteria", "compliance with constraints"],
            "domain_specific_failures": failure_modes
        }

    def save_output(self, architecture: Dict):
        """Save Phase 3 architecture to file."""
        output_file = self.output_dir / f"{self.agent_name}-phase3-architecture.json"
        with open(output_file, 'w') as f:
            json.dump(architecture, f, indent=2)
        print(f"\n✓ Phase 3 architecture saved to: {output_file}")

        # Save base prompt as markdown
        prompt_file = self.output_dir / f"{self.agent_name}-base-prompt-v1.md"
        with open(prompt_file, 'w') as f:
            f.write(architecture["base_prompt_v1"])
        print(f"✓ Base system prompt v1.0 saved to: {prompt_file}")

        return output_file


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_4_phase_sop(agent_name: str, mode: str = "interactive",
                    output_dir: Optional[Path] = None,
                    phase: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the complete 4-phase SOP or a specific phase.

    Args:
        agent_name: Name of agent to create
        mode: 'interactive' or 'batch'
        output_dir: Directory for outputs (default: ./agent-outputs/{agent_name})
        phase: Run specific phase (1-3) or None for all phases

    Returns:
        Dictionary with all phase outputs
    """
    if output_dir is None:
        output_dir = Path(f"./agent-outputs/{agent_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "agent_name": agent_name,
        "mode": mode,
        "start_time": datetime.now().isoformat(),
        "phases": {}
    }

    # Phase 1: Initial Analysis & Intent Decoding
    if phase is None or phase == 1:
        analyzer = Phase1Analyzer(agent_name, output_dir)
        phase1_data = analyzer.domain_breakdown()
        analyzer.save_output(phase1_data)
        results["phases"]["phase1"] = phase1_data

        if not all(analyzer.validation_checklist.values()):
            print("\n⚠️  Phase 1 validation failed. Please address issues before continuing.")
            if phase == 1:
                return results

    # Phase 2: Meta-Cognitive Extraction
    if phase is None or phase == 2:
        if "phase1" not in results["phases"]:
            # Load Phase 1 data
            phase1_file = output_dir / f"{agent_name}-phase1-analysis.json"
            if not phase1_file.exists():
                print(f"Error: Phase 1 data not found at {phase1_file}")
                return results
            with open(phase1_file) as f:
                phase1_data = json.load(f)
        else:
            phase1_data = results["phases"]["phase1"]

        extractor = Phase2Extractor(agent_name, phase1_data, output_dir)
        phase2_data = extractor.extract_expertise()
        extractor.save_output(phase2_data)
        results["phases"]["phase2"] = phase2_data

        if not all(extractor.validation_checklist.values()):
            print("\n⚠️  Phase 2 validation failed. Please address issues before continuing.")
            if phase == 2:
                return results

    # Phase 3: Agent Architecture Design
    if phase is None or phase == 3:
        if "phase2" not in results["phases"]:
            # Load Phase 2 data
            phase2_file = output_dir / f"{agent_name}-phase2-extraction.json"
            if not phase2_file.exists():
                print(f"Error: Phase 2 data not found at {phase2_file}")
                return results
            with open(phase2_file) as f:
                phase2_data = json.load(f)
        else:
            phase2_data = results["phases"]["phase2"]

        architect = Phase3Architect(agent_name, phase2_data, output_dir)
        phase3_data = architect.design_architecture()
        architect.save_output(phase3_data)
        results["phases"]["phase3"] = phase3_data

    results["end_time"] = datetime.now().isoformat()

    # Save complete results
    summary_file = output_dir / f"{agent_name}-4phase-sop-complete.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"✓ Complete 4-phase SOP results saved to: {summary_file}")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="4-Phase Agent Creation SOP Script"
    )
    parser.add_argument(
        "--agent-name",
        required=True,
        help="Name of the agent to create (e.g., 'marketing-specialist')"
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "batch"],
        default="interactive",
        help="Execution mode: interactive (CLI prompts) or batch (from file)"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        help="Run specific phase only (1-3)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input YAML file for batch mode"
    )

    args = parser.parse_args()

    if args.mode == "batch" and not args.input:
        print("Error: --input required for batch mode")
        sys.exit(1)

    try:
        results = run_4_phase_sop(
            agent_name=args.agent_name,
            mode=args.mode,
            output_dir=args.output_dir,
            phase=args.phase
        )

        print("\n✓ 4-Phase SOP execution complete!")
        print(f"Check output directory for all generated files.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
