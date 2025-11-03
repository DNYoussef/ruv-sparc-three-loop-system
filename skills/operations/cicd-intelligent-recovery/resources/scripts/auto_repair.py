#!/usr/bin/env python3
"""
Auto-Repair Script for CI/CD Intelligent Recovery
Automated fix generation with connascence-aware context bundling
"""

import json
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class FixStrategy(Enum):
    """Fix strategies based on connascence analysis"""
    ISOLATED = "isolated"  # Single file fix
    BUNDLED = "bundled"    # Multiple related files
    ARCHITECTURAL = "architectural"  # Requires system redesign


@dataclass
class ConnascenceContext:
    """Connascence coupling context for fix strategy"""
    name: List[str]  # Files sharing symbols (CoN)
    type: List[str]  # Files sharing types (CoT)
    algorithm: List[str]  # Files sharing algorithms (CoA)

    @property
    def total_coupling(self) -> int:
        """Total connascence coupling count"""
        return len(self.name) + len(self.type) + len(self.algorithm)

    @property
    def is_high_coupling(self) -> bool:
        """High coupling requires bundled fixes"""
        return self.total_coupling > 5


@dataclass
class RootCause:
    """Root cause failure information"""
    failure_id: str
    file: str
    line: int
    error_message: str
    root_cause_description: str
    cascaded_failures: List[str]
    connascence: ConnascenceContext
    fix_strategy: FixStrategy


class AutoRepair:
    """Automated repair engine with intelligent fix generation"""

    def __init__(self, artifacts_dir: str = ".claude/.artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.fixes_dir = self.artifacts_dir / "fixes"
        self.fixes_dir.mkdir(parents=True, exist_ok=True)

    def load_root_causes(self) -> List[RootCause]:
        """Load root causes from consensus"""
        consensus_file = self.artifacts_dir / "root-causes-consensus.json"

        if not consensus_file.exists():
            raise FileNotFoundError(f"Root cause consensus not found: {consensus_file}")

        with open(consensus_file) as f:
            data = json.load(f)

        root_causes = []
        for root in data.get("roots", []):
            failure = root["failure"]
            conn_context = root.get("connascenceContext", {})

            root_causes.append(RootCause(
                failure_id=failure.get("testName", "unknown"),
                file=failure.get("file", "unknown"),
                line=failure.get("line", 0),
                error_message=failure.get("errorMessage", ""),
                root_cause_description=root.get("rootCause", ""),
                cascaded_failures=root.get("cascadedFailures", []),
                connascence=ConnascenceContext(
                    name=conn_context.get("name", []),
                    type=conn_context.get("type", []),
                    algorithm=conn_context.get("algorithm", [])
                ),
                fix_strategy=FixStrategy(root.get("fixStrategy", "isolated"))
            ))

        return root_causes

    def generate_fix_plan(self, root_cause: RootCause) -> Dict[str, Any]:
        """
        Generate detailed fix plan using program-of-thought structure

        Returns fix plan with:
        - Affected files (primary + connascence)
        - Minimal changes description
        - Predicted side effects
        - Validation approach
        """
        plan = {
            "rootCause": root_cause.root_cause_description,
            "fixStrategy": root_cause.fix_strategy.value,
            "files": [],
            "minimalChanges": "",
            "predictedSideEffects": [],
            "validationPlan": {
                "mustPass": [],
                "mightFail": [],
                "newTests": []
            },
            "reasoning": []
        }

        # Step 1: Identify affected files
        plan["files"].append({
            "path": root_cause.file,
            "reason": "primary failure location",
            "changes": f"Fix root cause: {root_cause.root_cause_description}"
        })

        # Add connascence-related files
        for conn_file in root_cause.connascence.name:
            plan["files"].append({
                "path": conn_file,
                "reason": "connascence of name (shared symbols)",
                "changes": "Update symbol references to match fix"
            })

        for conn_file in root_cause.connascence.type:
            plan["files"].append({
                "path": conn_file,
                "reason": "connascence of type (type dependencies)",
                "changes": "Update type definitions/usage"
            })

        for conn_file in root_cause.connascence.algorithm:
            plan["files"].append({
                "path": conn_file,
                "reason": "connascence of algorithm (shared logic)",
                "changes": "Update algorithm implementation"
            })

        # Step 2: Design minimal fix
        if root_cause.fix_strategy == FixStrategy.ISOLATED:
            plan["minimalChanges"] = "Single file fix - isolated change with no cascading updates"
        elif root_cause.fix_strategy == FixStrategy.BUNDLED:
            plan["minimalChanges"] = f"Bundled fix across {len(plan['files'])} files - atomic changes to preserve consistency"
        else:  # ARCHITECTURAL
            plan["minimalChanges"] = "Architectural refactor required - reduces coupling and prevents future cascades"

        # Step 3: Predict side effects
        if root_cause.cascaded_failures:
            plan["predictedSideEffects"].append(
                f"Auto-resolves {len(root_cause.cascaded_failures)} cascaded failures"
            )

        if root_cause.connascence.is_high_coupling:
            plan["predictedSideEffects"].append(
                f"High coupling detected ({root_cause.connascence.total_coupling} files) - verify all references"
            )

        # Step 4: Plan validation
        plan["validationPlan"]["mustPass"].append(root_cause.failure_id)

        for cascaded in root_cause.cascaded_failures:
            plan["validationPlan"]["mustPass"].append(cascaded)

        # Step 5: Reasoning
        plan["reasoning"] = [
            f"Root cause: {root_cause.root_cause_description}",
            f"Strategy: {root_cause.fix_strategy.value} ({len(plan['files'])} files)",
            f"Connascence coupling: {root_cause.connascence.total_coupling} files",
            f"Expected cascade resolution: {len(root_cause.cascaded_failures)} tests"
        ]

        return plan

    def execute_fix(self, root_cause: RootCause, fix_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute fix implementation based on plan

        Returns fix implementation with:
        - Patch in git diff format
        - Files changed
        - Change descriptions with reasoning
        - Commit message
        """
        impl = {
            "patch": "",
            "filesChanged": [f["path"] for f in fix_plan["files"]],
            "changes": [],
            "commitMessage": ""
        }

        # Generate changes for each file
        for file_info in fix_plan["files"]:
            impl["changes"].append({
                "file": file_info["path"],
                "what": file_info["changes"],
                "why": f"Root cause fix: {root_cause.root_cause_description}",
                "reasoning": file_info["reason"]
            })

        # Generate commit message
        impl["commitMessage"] = self._generate_commit_message(
            root_cause, fix_plan, impl
        )

        # Note: Actual code patching would happen here
        # This is a template showing the structure
        impl["patch"] = "# Git patch would be generated here\n"
        impl["patch"] += f"# Files: {', '.join(impl['filesChanged'])}\n"
        impl["patch"] += f"# Strategy: {fix_plan['fixStrategy']}\n"

        return impl

    def _generate_commit_message(
        self,
        root_cause: RootCause,
        fix_plan: Dict[str, Any],
        impl: Dict[str, Any]
    ) -> str:
        """Generate descriptive commit message with reasoning"""
        msg = f"fix: {root_cause.failure_id}\n\n"
        msg += f"Root Cause: {root_cause.root_cause_description}\n\n"
        msg += f"Fix Strategy: {fix_plan['fixStrategy']}\n"
        msg += f"Files Changed: {len(impl['filesChanged'])}\n\n"

        msg += "Changes:\n"
        for change in impl["changes"]:
            msg += f"- {change['file']}: {change['what']}\n"

        if root_cause.cascaded_failures:
            msg += f"\nResolves {len(root_cause.cascaded_failures)} cascaded failures\n"

        msg += f"\nConnascence Context: {root_cause.connascence.total_coupling} coupled files\n"
        msg += "\nProgram-of-Thought: Plan → Execute → Validate → Approve\n"

        return msg

    def validate_fix(self, root_cause: RootCause, impl: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate fix quality before approval

        Returns validation report with:
        - Fix applied status
        - Test results
        - Theater check
        - Side effects verification
        """
        validation = {
            "fixApplied": True,
            "originalTestPassed": False,
            "allTestsResult": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "failedTests": []
            },
            "rootCauseResolved": False,
            "cascadeResolved": False,
            "newFailures": [],
            "sideEffects": {
                "predicted": [],
                "unexpected": []
            },
            "verdict": "PENDING",
            "reasoning": ""
        }

        # Note: Actual validation would run tests here
        # This shows the validation structure

        validation["reasoning"] = (
            f"Validating fix for {root_cause.failure_id} "
            f"with {len(impl['filesChanged'])} file changes"
        )

        return validation

    def repair_all(self) -> Dict[str, Any]:
        """
        Execute complete repair workflow

        Returns summary of all repairs with approval decisions
        """
        root_causes = self.load_root_causes()

        results = {
            "total": len(root_causes),
            "approved": 0,
            "rejected": 0,
            "fixes": []
        }

        for root_cause in root_causes:
            print(f"\n=== Fixing: {root_cause.failure_id} ===")

            # Phase 1: Planning
            print(f"  1. Generating fix plan...")
            fix_plan = self.generate_fix_plan(root_cause)

            # Save plan
            plan_file = self.artifacts_dir / f"fix-plan-{root_cause.failure_id}.json"
            with open(plan_file, 'w') as f:
                json.dump(fix_plan, f, indent=2)

            # Phase 2: Execution
            print(f"  2. Executing fix...")
            impl = self.execute_fix(root_cause, fix_plan)

            # Save implementation
            impl_file = self.artifacts_dir / f"fix-impl-{root_cause.failure_id}.json"
            with open(impl_file, 'w') as f:
                json.dump(impl, f, indent=2)

            # Save patch
            patch_file = self.fixes_dir / f"{root_cause.failure_id}.patch"
            with open(patch_file, 'w') as f:
                f.write(impl["patch"])

            # Phase 3: Validation
            print(f"  3. Validating fix...")
            validation = self.validate_fix(root_cause, impl)

            # Save validation
            val_file = self.artifacts_dir / f"fix-validation-sandbox-{root_cause.failure_id}.json"
            with open(val_file, 'w') as f:
                json.dump(validation, f, indent=2)

            # Phase 4: Approval decision
            if validation["verdict"] == "PASS":
                print(f"  ✅ Fix approved")
                results["approved"] += 1
                decision = "APPROVED"
            else:
                print(f"  ❌ Fix rejected")
                results["rejected"] += 1
                decision = "REJECTED"

            results["fixes"].append({
                "failureId": root_cause.failure_id,
                "decision": decision,
                "strategy": fix_plan["fixStrategy"],
                "filesChanged": len(impl["filesChanged"])
            })

        # Save summary
        summary_file = self.artifacts_dir / "auto-repair-summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Auto-Repair Complete: {results['approved']}/{results['total']} approved")
        print(f"{'='*60}\n")

        return results


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        artifacts_dir = sys.argv[1]
    else:
        artifacts_dir = ".claude/.artifacts"

    try:
        repairer = AutoRepair(artifacts_dir)
        results = repairer.repair_all()

        # Exit with error if no fixes approved
        if results["approved"] == 0:
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
