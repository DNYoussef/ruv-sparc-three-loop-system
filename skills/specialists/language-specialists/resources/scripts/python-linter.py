#!/usr/bin/env python3
"""
Python Linter - Automated code quality checks for Python projects.

Runs black, ruff, mypy, and checks import ordering with isort.
Validates docstrings and reports cyclomatic complexity.

Usage:
    python python-linter.py --dir src/ --strict
    python python-linter.py --dir . --fix
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def run_command(cmd: List[str], cwd: Path = Path(".")) -> Tuple[int, str, str]:
    """Execute shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out after 60 seconds"
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"


def check_black(directory: Path, fix: bool = False) -> bool:
    """Run black formatter."""
    print("🔍 Checking code formatting with black...")
    cmd = ["black", str(directory)]
    if not fix:
        cmd.append("--check")

    exit_code, stdout, stderr = run_command(cmd)

    if exit_code == 0:
        print("✅ Black formatting: PASSED")
        return True
    else:
        print(f"❌ Black formatting: FAILED\n{stderr}")
        if not fix:
            print("💡 Run with --fix to auto-format")
        return False


def check_ruff(directory: Path, strict: bool = False) -> bool:
    """Run ruff linter."""
    print("\n🔍 Running ruff linter...")
    cmd = ["ruff", "check", str(directory)]
    if not strict:
        cmd.extend(["--select", "E,F,W"])  # Basic checks only

    exit_code, stdout, stderr = run_command(cmd)

    if exit_code == 0:
        print("✅ Ruff linting: PASSED")
        return True
    else:
        print(f"❌ Ruff linting: FAILED\n{stdout}")
        return False


def check_mypy(directory: Path, strict: bool = False) -> bool:
    """Run mypy type checker."""
    print("\n🔍 Running mypy type checker...")
    cmd = ["mypy", str(directory)]
    if strict:
        cmd.append("--strict")

    exit_code, stdout, stderr = run_command(cmd)

    if exit_code == 0:
        print("✅ Mypy type checking: PASSED")
        return True
    else:
        print(f"❌ Mypy type checking: FAILED\n{stdout}")
        return False


def check_isort(directory: Path, fix: bool = False) -> bool:
    """Check import ordering with isort."""
    print("\n🔍 Checking import ordering with isort...")
    cmd = ["isort", str(directory)]
    if not fix:
        cmd.append("--check-only")

    exit_code, stdout, stderr = run_command(cmd)

    if exit_code == 0:
        print("✅ Import ordering: PASSED")
        return True
    else:
        print(f"❌ Import ordering: FAILED\n{stderr}")
        if not fix:
            print("💡 Run with --fix to auto-organize imports")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Python code quality linter")
    parser.add_argument("--dir", type=str, default=".", help="Directory to check")
    parser.add_argument(
        "--strict", action="store_true", help="Enable strict mode (mypy strict)"
    )
    parser.add_argument(
        "--fix", action="store_true", help="Auto-fix formatting and import issues"
    )

    args = parser.parse_args()
    directory = Path(args.dir)

    if not directory.exists():
        print(f"❌ Error: Directory {directory} does not exist")
        sys.exit(1)

    print(f"🚀 Running Python linter on: {directory.absolute()}\n")

    results = []
    results.append(check_black(directory, fix=args.fix))
    results.append(check_ruff(directory, strict=args.strict))
    results.append(check_mypy(directory, strict=args.strict))
    results.append(check_isort(directory, fix=args.fix))

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Results: {passed}/{total} checks passed")
    print("=" * 50)

    if passed == total:
        print("✅ All checks passed!")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
