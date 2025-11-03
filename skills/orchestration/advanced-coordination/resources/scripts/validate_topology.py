#!/usr/bin/env python3
"""
Swarm Topology Validator

Validates swarm topology configuration files against schema and best practices.
Ensures topology files are valid before deployment.

Usage:
    python validate_topology.py <topology-file.yaml>
    python validate_topology.py <topology-file.yaml> --json
"""

import sys
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple


class TopologyValidator:
    """Validates swarm topology configurations."""

    VALID_TOPOLOGIES = {"mesh", "hierarchical", "ring", "star"}
    VALID_STRATEGIES = {"balanced", "specialized", "adaptive"}
    VALID_CONSENSUS = {"byzantine", "raft", "gossip", "proof-of-learning"}

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.errors = []
        self.warnings = []
        self.config = None

    def load_config(self) -> bool:
        """Load YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            return True
        except FileNotFoundError:
            self.errors.append(f"Configuration file not found: {self.config_path}")
            return False
        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML syntax: {e}")
            return False

    def validate_structure(self) -> bool:
        """Validate required configuration structure."""
        if not isinstance(self.config, dict):
            self.errors.append("Configuration must be a YAML dictionary")
            return False

        required_fields = {"topology", "maxAgents"}
        missing_fields = required_fields - set(self.config.keys())

        if missing_fields:
            self.errors.append(f"Missing required fields: {missing_fields}")
            return False

        return True

    def validate_topology(self) -> bool:
        """Validate topology type."""
        topology = self.config.get("topology")

        if topology not in self.VALID_TOPOLOGIES:
            self.errors.append(
                f"Invalid topology '{topology}'. Must be one of: {self.VALID_TOPOLOGIES}"
            )
            return False

        return True

    def validate_agent_count(self) -> bool:
        """Validate agent count constraints."""
        max_agents = self.config.get("maxAgents")

        if not isinstance(max_agents, int):
            self.errors.append("maxAgents must be an integer")
            return False

        if max_agents < 2:
            self.errors.append("maxAgents must be at least 2 for coordination")
            return False

        if max_agents > 100:
            self.warnings.append(
                f"maxAgents={max_agents} is very high. Consider scaling gradually."
            )

        # Topology-specific constraints
        topology = self.config.get("topology")
        if topology == "hierarchical" and max_agents < 3:
            self.errors.append("Hierarchical topology requires at least 3 agents")
            return False

        if topology == "ring" and max_agents < 3:
            self.errors.append("Ring topology requires at least 3 agents")
            return False

        return True

    def validate_strategy(self) -> bool:
        """Validate distribution strategy."""
        strategy = self.config.get("strategy", "balanced")

        if strategy not in self.VALID_STRATEGIES:
            self.errors.append(
                f"Invalid strategy '{strategy}'. Must be one of: {self.VALID_STRATEGIES}"
            )
            return False

        return True

    def validate_consensus(self) -> bool:
        """Validate consensus mechanism if specified."""
        consensus = self.config.get("consensus")

        if consensus is None:
            return True  # Consensus is optional

        if consensus not in self.VALID_CONSENSUS:
            self.errors.append(
                f"Invalid consensus '{consensus}'. Must be one of: {self.VALID_CONSENSUS}"
            )
            return False

        # Consensus-specific warnings
        if consensus == "byzantine" and self.config.get("maxAgents", 0) < 4:
            self.warnings.append(
                "Byzantine consensus requires at least 4 agents for fault tolerance"
            )

        return True

    def validate_all(self) -> Tuple[bool, Dict]:
        """Run all validations and return results."""
        if not self.load_config():
            return False, self._get_results()

        validations = [
            self.validate_structure(),
            self.validate_topology(),
            self.validate_agent_count(),
            self.validate_strategy(),
            self.validate_consensus(),
        ]

        is_valid = all(validations)
        return is_valid, self._get_results()

    def _get_results(self) -> Dict:
        """Get validation results."""
        return {
            "valid": len(self.errors) == 0,
            "config_file": str(self.config_path),
            "errors": self.errors,
            "warnings": self.warnings,
            "config": self.config,
        }


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validate_topology.py <topology-file.yaml> [--json]")
        sys.exit(1)

    config_file = sys.argv[1]
    json_output = "--json" in sys.argv

    validator = TopologyValidator(config_file)
    is_valid, results = validator.validate_all()

    if json_output:
        print(json.dumps(results, indent=2))
    else:
        # Human-readable output
        print(f"\n{'='*60}")
        print(f"Topology Validation: {config_file}")
        print(f"{'='*60}\n")

        if results["errors"]:
            print("[!] ERRORS:")
            for error in results["errors"]:
                print(f"  - {error}")
            print()

        if results["warnings"]:
            print("[!] WARNINGS:")
            for warning in results["warnings"]:
                print(f"  - {warning}")
            print()

        if results["valid"]:
            print("[PASS] Validation PASSED")
            print(f"\nTopology: {results['config'].get('topology')}")
            print(f"Max Agents: {results['config'].get('maxAgents')}")
            print(f"Strategy: {results['config'].get('strategy', 'balanced')}")
            if results['config'].get('consensus'):
                print(f"Consensus: {results['config']['consensus']}")
        else:
            print("[FAIL] Validation FAILED")

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
