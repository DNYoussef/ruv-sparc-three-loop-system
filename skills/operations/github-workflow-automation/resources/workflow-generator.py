#!/usr/bin/env python3
"""
GitHub Actions Workflow Generator
Intelligent workflow generation with swarm coordination and language detection
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set


class WorkflowGenerator:
    """Generate optimized GitHub Actions workflows based on repository analysis."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.languages: Set[str] = set()
        self.frameworks: Set[str] = set()
        self.package_managers: Set[str] = set()
        self.workflow_config: Dict = {}

    def detect_languages(self) -> Set[str]:
        """Detect programming languages used in repository."""
        language_patterns = {
            "python": ["*.py", "requirements.txt", "setup.py", "pyproject.toml"],
            "javascript": ["*.js", "package.json", "*.jsx"],
            "typescript": ["*.ts", "tsconfig.json", "*.tsx"],
            "go": ["*.go", "go.mod"],
            "rust": ["*.rs", "Cargo.toml"],
            "java": ["*.java", "pom.xml", "build.gradle"],
            "ruby": ["*.rb", "Gemfile"],
            "php": ["*.php", "composer.json"],
            "csharp": ["*.cs", "*.csproj"],
            "cpp": ["*.cpp", "*.h", "CMakeLists.txt"],
        }

        detected = set()
        for lang, patterns in language_patterns.items():
            for pattern in patterns:
                if list(self.repo_path.rglob(pattern)):
                    detected.add(lang)
                    break

        self.languages = detected
        return detected

    def detect_frameworks(self) -> Set[str]:
        """Detect frameworks and their configurations."""
        framework_markers = {
            "react": ["package.json", "src/App.jsx", "src/App.tsx"],
            "vue": ["package.json", "vue.config.js"],
            "angular": ["angular.json"],
            "django": ["manage.py", "settings.py"],
            "flask": ["app.py", "wsgi.py"],
            "express": ["package.json", "app.js", "server.js"],
            "spring": ["pom.xml", "build.gradle"],
            "rails": ["Gemfile", "config/routes.rb"],
            "nextjs": ["next.config.js", "package.json"],
            "gatsby": ["gatsby-config.js"],
        }

        detected = set()
        for framework, markers in framework_markers.items():
            if any(list(self.repo_path.rglob(marker)) for marker in markers):
                detected.add(framework)

        self.frameworks = detected
        return detected

    def detect_package_managers(self) -> Set[str]:
        """Detect package managers in use."""
        pm_files = {
            "npm": "package.json",
            "yarn": "yarn.lock",
            "pnpm": "pnpm-lock.yaml",
            "pip": "requirements.txt",
            "poetry": "pyproject.toml",
            "go": "go.mod",
            "cargo": "Cargo.toml",
            "maven": "pom.xml",
            "gradle": "build.gradle",
            "composer": "composer.json",
            "bundler": "Gemfile",
        }

        detected = set()
        for pm, file in pm_files.items():
            if (self.repo_path / file).exists():
                detected.add(pm)

        self.package_managers = detected
        return detected

    def generate_ci_workflow(self) -> str:
        """Generate CI workflow based on detected technologies."""
        workflow = {
            "name": "Intelligent CI Pipeline",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {},
            },
            "jobs": {},
        }

        # Add language-specific jobs
        if "python" in self.languages:
            workflow["jobs"]["python-ci"] = self._python_job()

        if any(lang in self.languages for lang in ["javascript", "typescript"]):
            workflow["jobs"]["node-ci"] = self._node_job()

        if "go" in self.languages:
            workflow["jobs"]["go-ci"] = self._go_job()

        if "rust" in self.languages:
            workflow["jobs"]["rust-ci"] = self._rust_job()

        # Add security scanning
        workflow["jobs"]["security"] = self._security_job()

        # Add swarm coordination
        if len(workflow["jobs"]) > 2:
            workflow["jobs"]["swarm-coordinator"] = self._swarm_coordinator_job()

        return self._workflow_to_yaml(workflow)

    def _python_job(self) -> Dict:
        """Generate Python CI job."""
        return {
            "runs-on": "ubuntu-latest",
            "strategy": {
                "matrix": {
                    "python-version": ["3.9", "3.10", "3.11"],
                }
            },
            "steps": [
                {"uses": "actions/checkout@v3"},
                {
                    "name": "Set up Python ${{ matrix.python-version }}",
                    "uses": "actions/setup-python@v4",
                    "with": {"python-version": "${{ matrix.python-version }}"},
                },
                {
                    "name": "Cache pip packages",
                    "uses": "actions/cache@v3",
                    "with": {
                        "path": "~/.cache/pip",
                        "key": "${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}",
                    },
                },
                {
                    "name": "Install dependencies",
                    "run": "pip install -r requirements.txt",
                },
                {
                    "name": "Run tests",
                    "run": "pytest --cov --cov-report=xml",
                },
                {
                    "name": "Upload coverage",
                    "uses": "codecov/codecov-action@v3",
                },
            ],
        }

    def _node_job(self) -> Dict:
        """Generate Node.js CI job."""
        pm = "npm"
        if "yarn" in self.package_managers:
            pm = "yarn"
        elif "pnpm" in self.package_managers:
            pm = "pnpm"

        install_cmd = {
            "npm": "npm ci",
            "yarn": "yarn install --frozen-lockfile",
            "pnpm": "pnpm install --frozen-lockfile",
        }

        return {
            "runs-on": "ubuntu-latest",
            "strategy": {
                "matrix": {
                    "node-version": ["16.x", "18.x", "20.x"],
                }
            },
            "steps": [
                {"uses": "actions/checkout@v3"},
                {
                    "name": "Use Node.js ${{ matrix.node-version }}",
                    "uses": "actions/setup-node@v3",
                    "with": {"node-version": "${{ matrix.node-version }}"},
                },
                {
                    "name": "Cache dependencies",
                    "uses": "actions/cache@v3",
                    "with": {
                        "path": "node_modules",
                        "key": "${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}",
                    },
                },
                {
                    "name": "Install dependencies",
                    "run": install_cmd[pm],
                },
                {
                    "name": "Run tests",
                    "run": f"{pm} test",
                },
                {
                    "name": "Build",
                    "run": f"{pm} run build",
                },
            ],
        }

    def _go_job(self) -> Dict:
        """Generate Go CI job."""
        return {
            "runs-on": "ubuntu-latest",
            "strategy": {
                "matrix": {
                    "go-version": ["1.19", "1.20", "1.21"],
                }
            },
            "steps": [
                {"uses": "actions/checkout@v3"},
                {
                    "name": "Set up Go",
                    "uses": "actions/setup-go@v4",
                    "with": {"go-version": "${{ matrix.go-version }}"},
                },
                {
                    "name": "Cache Go modules",
                    "uses": "actions/cache@v3",
                    "with": {
                        "path": "~/go/pkg/mod",
                        "key": "${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}",
                    },
                },
                {
                    "name": "Download dependencies",
                    "run": "go mod download",
                },
                {
                    "name": "Run tests",
                    "run": "go test -v -race -coverprofile=coverage.txt ./...",
                },
                {
                    "name": "Build",
                    "run": "go build -v ./...",
                },
            ],
        }

    def _rust_job(self) -> Dict:
        """Generate Rust CI job."""
        return {
            "runs-on": "ubuntu-latest",
            "steps": [
                {"uses": "actions/checkout@v3"},
                {
                    "name": "Install Rust",
                    "uses": "actions-rs/toolchain@v1",
                    "with": {
                        "toolchain": "stable",
                        "override": True,
                        "components": "rustfmt, clippy",
                    },
                },
                {
                    "name": "Cache cargo",
                    "uses": "actions/cache@v3",
                    "with": {
                        "path": "~/.cargo",
                        "key": "${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}",
                    },
                },
                {
                    "name": "Run tests",
                    "run": "cargo test --verbose",
                },
                {
                    "name": "Run clippy",
                    "run": "cargo clippy -- -D warnings",
                },
                {
                    "name": "Check formatting",
                    "run": "cargo fmt -- --check",
                },
            ],
        }

    def _security_job(self) -> Dict:
        """Generate security scanning job."""
        return {
            "runs-on": "ubuntu-latest",
            "steps": [
                {"uses": "actions/checkout@v3"},
                {
                    "name": "Run Trivy vulnerability scanner",
                    "uses": "aquasecurity/trivy-action@master",
                    "with": {
                        "scan-type": "fs",
                        "scan-ref": ".",
                        "format": "sarif",
                        "output": "trivy-results.sarif",
                    },
                },
                {
                    "name": "Upload Trivy results",
                    "uses": "github/codeql-action/upload-sarif@v2",
                    "with": {"sarif_file": "trivy-results.sarif"},
                },
            ],
        }

    def _swarm_coordinator_job(self) -> Dict:
        """Generate swarm coordination job."""
        return {
            "needs": ["python-ci", "node-ci", "security"],
            "runs-on": "ubuntu-latest",
            "steps": [
                {"uses": "actions/checkout@v3"},
                {
                    "name": "Initialize Swarm Coordination",
                    "run": "npx ruv-swarm init --topology mesh --max-agents 6",
                },
                {
                    "name": "Coordinate Results",
                    "run": "npx ruv-swarm actions coordinate --analyze-results --generate-report",
                },
            ],
        }

    def _workflow_to_yaml(self, workflow: Dict) -> str:
        """Convert workflow dict to YAML string."""
        import yaml
        try:
            return yaml.dump(workflow, default_flow_style=False, sort_keys=False)
        except ImportError:
            # Fallback to JSON if PyYAML not available
            return json.dumps(workflow, indent=2)

    def analyze_repository(self) -> Dict:
        """Analyze repository and return comprehensive report."""
        self.detect_languages()
        self.detect_frameworks()
        self.detect_package_managers()

        return {
            "languages": list(self.languages),
            "frameworks": list(self.frameworks),
            "package_managers": list(self.package_managers),
            "recommended_workflow": "ci.yml",
            "complexity": len(self.languages) + len(self.frameworks),
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate optimized GitHub Actions workflows"
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to repository (default: current directory)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze repository and show technologies",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate CI workflow",
    )
    parser.add_argument(
        "--output",
        help="Output file path for generated workflow",
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format",
    )

    args = parser.parse_args()

    generator = WorkflowGenerator(args.repo_path)

    if args.analyze:
        analysis = generator.analyze_repository()
        print(json.dumps(analysis, indent=2))

    if args.generate:
        workflow = generator.generate_ci_workflow()
        if args.output:
            with open(args.output, "w") as f:
                f.write(workflow)
            print(f"✅ Workflow generated: {args.output}")
        else:
            print(workflow)


if __name__ == "__main__":
    main()
