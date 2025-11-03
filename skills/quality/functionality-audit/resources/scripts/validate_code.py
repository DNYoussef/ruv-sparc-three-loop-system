#!/usr/bin/env python3
"""
Code Validation Orchestrator for functionality-audit
Creates sandbox, runs tests, analyzes results, reports findings

Usage:
    python validate_code.py --code-path ./src --test-cases ./tests
    python validate_code.py --code-path ./app.py --auto-generate-tests
    python validate_code.py --config validation-config.json
"""
import argparse
import subprocess
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil


class CodeValidator:
    """Main orchestrator for code validation through sandbox testing"""

    def __init__(self, code_path: str, test_cases: Optional[str] = None,
                 auto_generate: bool = False, sandbox_type: str = "e2b"):
        """
        Initialize code validator

        Args:
            code_path: Path to code to validate
            test_cases: Path to existing test cases (optional)
            auto_generate: Auto-generate test cases if True
            sandbox_type: Sandbox type (e2b, docker, or local)
        """
        self.code_path = Path(code_path).resolve()
        self.test_cases = Path(test_cases).resolve() if test_cases else None
        self.auto_generate = auto_generate
        self.sandbox_type = sandbox_type

        # Validation state
        self.sandbox_id = None
        self.workspace = None
        self.test_results = {}
        self.failures = []
        self.recommendations = []

        # Validate inputs
        if not self.code_path.exists():
            raise FileNotFoundError(f"Code path not found: {self.code_path}")

        if test_cases and not self.test_cases.exists():
            raise FileNotFoundError(f"Test cases not found: {self.test_cases}")

    def create_sandbox(self) -> str:
        """
        Create isolated test environment

        Returns:
            sandbox_id: Identifier for the sandbox
        """
        print(f"[1/5] Creating {self.sandbox_type} sandbox...")

        try:
            if self.sandbox_type == "e2b":
                return self._create_e2b_sandbox()
            elif self.sandbox_type == "docker":
                return self._create_docker_sandbox()
            else:
                return self._create_local_sandbox()
        except Exception as e:
            print(f"ERROR: Failed to create sandbox: {e}")
            raise

    def _create_e2b_sandbox(self) -> str:
        """Create E2B sandbox using flow-nexus"""
        config_path = Path(__file__).parent.parent / "templates" / "sandbox-config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Sandbox config not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        # Detect language and adjust template
        if self.code_path.suffix in ['.py', '.pyw']:
            config['template'] = 'python'
            config['install_packages'].extend(['pytest', 'coverage', 'pytest-cov'])
        elif self.code_path.suffix in ['.js', '.ts', '.jsx', '.tsx']:
            config['template'] = 'node'
            config['install_packages'].extend(['jest', 'mocha', 'chai'])

        # Create sandbox via CLI
        cmd = [
            'npx', 'flow-nexus@latest', 'sandbox', 'create',
            '--template', config['template'],
            '--timeout', str(config['timeout']),
            '--output', 'json'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create E2B sandbox: {result.stderr}")

        response = json.loads(result.stdout)
        self.sandbox_id = response.get('sandbox_id')
        self.workspace = Path(config['working_dir'])

        print(f"✓ E2B sandbox created: {self.sandbox_id}")
        return self.sandbox_id

    def _create_docker_sandbox(self) -> str:
        """Create Docker container sandbox"""
        container_name = f"validator-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Detect language and choose image
        if self.code_path.suffix in ['.py', '.pyw']:
            image = 'python:3.11-slim'
        elif self.code_path.suffix in ['.js', '.ts']:
            image = 'node:20-slim'
        else:
            image = 'ubuntu:22.04'

        cmd = [
            'docker', 'run', '-d',
            '--name', container_name,
            '--network', 'none',  # Isolate network
            '-v', f"{self.code_path.parent}:/workspace",
            image,
            'tail', '-f', '/dev/null'  # Keep container alive
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create Docker sandbox: {result.stderr}")

        self.sandbox_id = container_name
        self.workspace = Path('/workspace')

        print(f"✓ Docker sandbox created: {container_name}")
        return container_name

    def _create_local_sandbox(self) -> str:
        """Create local temporary directory sandbox"""
        temp_dir = tempfile.mkdtemp(prefix='validator-')

        # Copy code to temp directory
        if self.code_path.is_dir():
            shutil.copytree(self.code_path, Path(temp_dir) / 'code', dirs_exist_ok=True)
        else:
            shutil.copy2(self.code_path, temp_dir)

        self.sandbox_id = temp_dir
        self.workspace = Path(temp_dir)

        print(f"✓ Local sandbox created: {temp_dir}")
        return temp_dir

    def generate_tests(self) -> Path:
        """
        Generate comprehensive test cases

        Returns:
            Path to generated test file
        """
        print("[2/5] Generating test cases...")

        if not self.auto_generate and self.test_cases:
            print(f"✓ Using existing tests: {self.test_cases}")
            return self.test_cases

        # Call test generator
        generator_path = Path(__file__).parent / "test_generator.py"

        if not generator_path.exists():
            raise FileNotFoundError(f"Test generator not found: {generator_path}")

        output_path = self.workspace / "generated_tests.py"

        cmd = [
            sys.executable,
            str(generator_path),
            '--code-path', str(self.code_path),
            '--output', str(output_path),
            '--include-edge-cases',
            '--include-boundaries'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise RuntimeError(f"Test generation failed: {result.stderr}")

        print(f"✓ Generated tests: {output_path}")
        return output_path

    def execute_tests(self, test_path: Path) -> Dict:
        """
        Run tests with monitoring

        Args:
            test_path: Path to test file

        Returns:
            Dict with test results
        """
        print("[3/5] Executing tests...")

        try:
            if self.sandbox_type == "e2b":
                return self._execute_tests_e2b(test_path)
            elif self.sandbox_type == "docker":
                return self._execute_tests_docker(test_path)
            else:
                return self._execute_tests_local(test_path)
        except Exception as e:
            print(f"ERROR: Test execution failed: {e}")
            raise

    def _execute_tests_e2b(self, test_path: Path) -> Dict:
        """Execute tests in E2B sandbox"""
        # Upload code and tests
        cmd = [
            'npx', 'flow-nexus@latest', 'sandbox', 'upload',
            '--sandbox-id', self.sandbox_id,
            '--file', str(self.code_path),
            '--destination', str(self.workspace / 'code.py')
        ]
        subprocess.run(cmd, check=True, timeout=30)

        cmd = [
            'npx', 'flow-nexus@latest', 'sandbox', 'upload',
            '--sandbox-id', self.sandbox_id,
            '--file', str(test_path),
            '--destination', str(self.workspace / 'tests.py')
        ]
        subprocess.run(cmd, check=True, timeout=30)

        # Run pytest with coverage
        cmd = [
            'npx', 'flow-nexus@latest', 'sandbox', 'execute',
            '--sandbox-id', self.sandbox_id,
            '--command', 'pytest tests.py --json-report --json-report-file=results.json --cov=code --cov-report=json',
            '--output', 'json'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        response = json.loads(result.stdout)

        # Parse results
        return self._parse_pytest_results(response.get('stdout', ''))

    def _execute_tests_docker(self, test_path: Path) -> Dict:
        """Execute tests in Docker container"""
        # Install dependencies
        if self.code_path.suffix in ['.py', '.pyw']:
            cmd = ['docker', 'exec', self.sandbox_id, 'pip', 'install', 'pytest', 'pytest-cov', 'pytest-json-report']
            subprocess.run(cmd, check=True, timeout=60)

            # Run tests
            cmd = [
                'docker', 'exec', self.sandbox_id,
                'pytest', str(self.workspace / test_path.name),
                '--json-report', '--json-report-file=/workspace/results.json',
                '--cov', str(self.workspace / self.code_path.name),
                '--cov-report=json:/workspace/coverage.json'
            ]
        else:
            cmd = ['docker', 'exec', self.sandbox_id, 'npm', 'install', '-g', 'jest']
            subprocess.run(cmd, check=True, timeout=60)

            cmd = ['docker', 'exec', self.sandbox_id, 'jest', '--json', '--outputFile=/workspace/results.json']

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Read results
        results_path = self.code_path.parent / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)

        return self._parse_pytest_results(result.stdout)

    def _execute_tests_local(self, test_path: Path) -> Dict:
        """Execute tests in local sandbox"""
        # Install pytest if needed
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest', 'pytest-cov', 'pytest-json-report'],
                      capture_output=True, timeout=60)

        # Run tests
        cmd = [
            sys.executable, '-m', 'pytest',
            str(test_path),
            '--json-report', '--json-report-file', str(self.workspace / 'results.json'),
            '--cov', str(self.code_path),
            '--cov-report=json:' + str(self.workspace / 'coverage.json'),
            '-v'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=self.workspace)

        # Load JSON results
        results_path = self.workspace / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)

        return self._parse_pytest_results(result.stdout)

    def _parse_pytest_results(self, output: str) -> Dict:
        """Parse pytest output into structured results"""
        # Simplified parsing - in production, use pytest-json-report
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'duration': 0.0,
            'tests': []
        }

        lines = output.split('\n')
        for line in lines:
            if 'passed' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'passed' in part.lower() and i > 0:
                        results['passed'] = int(parts[i-1])
            if 'failed' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'failed' in part.lower() and i > 0:
                        results['failed'] = int(parts[i-1])

        results['total_tests'] = results['passed'] + results['failed']
        return results

    def analyze_results(self, test_results: Dict) -> Tuple[List[Dict], List[str]]:
        """
        Analyze test results and identify failures

        Args:
            test_results: Dict with test execution results

        Returns:
            Tuple of (failures, recommendations)
        """
        print("[4/5] Analyzing results...")

        failures = []
        recommendations = []

        # Analyze test failures
        if test_results.get('failed', 0) > 0:
            for test in test_results.get('tests', []):
                if test.get('outcome') == 'failed':
                    failure = {
                        'test_name': test.get('nodeid', 'Unknown'),
                        'expected': test.get('call', {}).get('longrepr', 'N/A'),
                        'actual': test.get('call', {}).get('outcome', 'failed'),
                        'root_cause': self._identify_root_cause(test)
                    }
                    failures.append(failure)

        # Generate recommendations
        pass_rate = test_results.get('passed', 0) / max(test_results.get('total_tests', 1), 1)

        if pass_rate < 0.5:
            recommendations.append("CRITICAL: Less than 50% of tests passing. Major functionality issues detected.")
        elif pass_rate < 0.8:
            recommendations.append("WARNING: Less than 80% of tests passing. Significant issues found.")
        else:
            recommendations.append("PASS: Most tests passing. Minor issues may need attention.")

        # Coverage recommendations
        coverage_path = self.workspace / 'coverage.json'
        if coverage_path.exists():
            with open(coverage_path) as f:
                coverage = json.load(f)
                total_coverage = coverage.get('totals', {}).get('percent_covered', 0)

                if total_coverage < 70:
                    recommendations.append(f"Low test coverage ({total_coverage:.1f}%). Add more test cases.")
                elif total_coverage < 90:
                    recommendations.append(f"Good test coverage ({total_coverage:.1f}%). Consider edge cases.")
                else:
                    recommendations.append(f"Excellent test coverage ({total_coverage:.1f}%).")

        print(f"✓ Found {len(failures)} failures, generated {len(recommendations)} recommendations")
        return failures, recommendations

    def _identify_root_cause(self, test: Dict) -> str:
        """Identify root cause of test failure"""
        longrepr = test.get('call', {}).get('longrepr', '')

        if 'AssertionError' in longrepr:
            return "Assertion failed - expected behavior not met"
        elif 'TypeError' in longrepr:
            return "Type mismatch - incorrect data types"
        elif 'ValueError' in longrepr:
            return "Invalid value - input validation failed"
        elif 'AttributeError' in longrepr:
            return "Missing attribute/method - incomplete implementation"
        elif 'ImportError' in longrepr:
            return "Missing dependency - import failed"
        elif 'Timeout' in longrepr:
            return "Performance issue - operation too slow"
        else:
            return "Unknown error - manual investigation needed"

    def report_findings(self, test_results: Dict, failures: List[Dict],
                       recommendations: List[str]) -> Path:
        """
        Generate validation report

        Args:
            test_results: Test execution results
            failures: List of test failures
            recommendations: List of recommendations

        Returns:
            Path to generated report
        """
        print("[5/5] Generating report...")

        report = {
            'validation_report': {
                'skill_name': 'functionality-audit',
                'timestamp': datetime.now().isoformat(),
                'code_path': str(self.code_path),
                'sandbox_id': self.sandbox_id,
                'sandbox_type': self.sandbox_type,
                'test_results': {
                    'total_tests': test_results.get('total_tests', 0),
                    'passed': test_results.get('passed', 0),
                    'failed': test_results.get('failed', 0),
                    'skipped': test_results.get('skipped', 0),
                    'duration': test_results.get('duration', 0.0)
                },
                'failures': failures,
                'recommendations': recommendations,
                'verdict': self._generate_verdict(test_results, failures)
            }
        }

        # Write YAML report
        output_path = Path.cwd() / f"validation-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.yaml"

        try:
            import yaml
            with open(output_path, 'w') as f:
                yaml.dump(report, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            # Fallback to JSON if PyYAML not available
            output_path = output_path.with_suffix('.json')
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        print(f"✓ Report generated: {output_path}")
        return output_path

    def _generate_verdict(self, test_results: Dict, failures: List[Dict]) -> str:
        """Generate final validation verdict"""
        pass_rate = test_results.get('passed', 0) / max(test_results.get('total_tests', 1), 1)

        if pass_rate >= 0.95 and len(failures) == 0:
            return "APPROVED - Code is production-ready"
        elif pass_rate >= 0.8:
            return "CONDITIONAL - Minor issues, review recommended"
        elif pass_rate >= 0.5:
            return "NEEDS WORK - Significant issues, refactoring required"
        else:
            return "REJECTED - Major failures, complete rewrite needed"

    def cleanup(self):
        """Clean up sandbox resources"""
        print("Cleaning up sandbox...")

        try:
            if self.sandbox_type == "e2b" and self.sandbox_id:
                cmd = ['npx', 'flow-nexus@latest', 'sandbox', 'delete', '--sandbox-id', self.sandbox_id]
                subprocess.run(cmd, timeout=30)
            elif self.sandbox_type == "docker" and self.sandbox_id:
                subprocess.run(['docker', 'stop', self.sandbox_id], timeout=30)
                subprocess.run(['docker', 'rm', self.sandbox_id], timeout=30)
            elif self.sandbox_type == "local" and self.sandbox_id:
                shutil.rmtree(self.sandbox_id, ignore_errors=True)

            print("✓ Sandbox cleaned up")
        except Exception as e:
            print(f"WARNING: Cleanup failed: {e}")

    def run(self) -> Path:
        """
        Execute complete validation workflow

        Returns:
            Path to validation report
        """
        try:
            # Step 1: Create sandbox
            self.create_sandbox()

            # Step 2: Generate/load tests
            test_path = self.generate_tests()

            # Step 3: Execute tests
            test_results = self.execute_tests(test_path)
            self.test_results = test_results

            # Step 4: Analyze results
            failures, recommendations = self.analyze_results(test_results)
            self.failures = failures
            self.recommendations = recommendations

            # Step 5: Generate report
            report_path = self.report_findings(test_results, failures, recommendations)

            return report_path

        finally:
            # Always cleanup
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Code validation orchestrator for functionality-audit')
    parser.add_argument('--code-path', required=True, help='Path to code to validate')
    parser.add_argument('--test-cases', help='Path to existing test cases (optional)')
    parser.add_argument('--auto-generate-tests', action='store_true',
                       help='Auto-generate test cases')
    parser.add_argument('--sandbox-type', choices=['e2b', 'docker', 'local'],
                       default='local', help='Sandbox type (default: local)')
    parser.add_argument('--config', help='Path to configuration JSON file')

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
            args.code_path = config.get('code_path', args.code_path)
            args.test_cases = config.get('test_cases', args.test_cases)
            args.auto_generate_tests = config.get('auto_generate_tests', args.auto_generate_tests)
            args.sandbox_type = config.get('sandbox_type', args.sandbox_type)

    # Create validator
    validator = CodeValidator(
        code_path=args.code_path,
        test_cases=args.test_cases,
        auto_generate=args.auto_generate_tests,
        sandbox_type=args.sandbox_type
    )

    # Run validation
    print("=" * 60)
    print("FUNCTIONALITY-AUDIT CODE VALIDATOR")
    print("=" * 60)

    try:
        report_path = validator.run()

        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        print(f"Report: {report_path}")
        print(f"Tests: {validator.test_results.get('passed', 0)}/{validator.test_results.get('total_tests', 0)} passed")
        print(f"Failures: {len(validator.failures)}")
        print(f"Verdict: {validator._generate_verdict(validator.test_results, validator.failures)}")

        sys.exit(0 if validator.test_results.get('failed', 0) == 0 else 1)

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    main()
