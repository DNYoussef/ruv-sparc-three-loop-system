#!/usr/bin/env python3
"""
Deployment Verifier
Validates deployment configuration and infrastructure readiness
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class VerificationResult:
    """Verification check result"""
    category: str
    check: str
    passed: bool
    severity: str  # critical, high, medium, low
    message: str
    recommendation: str = ""


class DeploymentVerifier:
    """Comprehensive deployment configuration verification"""

    def __init__(self, target_path: str, environment: str = "production"):
        self.target_path = Path(target_path)
        self.environment = environment
        self.results: List[VerificationResult] = []

    def verify_all(self) -> Dict:
        """Run all verification checks"""
        print(f"\n{'='*60}")
        print(f"Deployment Verification - {self.environment.upper()}")
        print(f"{'='*60}\n")

        self.verify_environment_config()
        self.verify_dependencies()
        self.verify_build_process()
        self.verify_database_config()
        self.verify_monitoring()
        self.verify_logging()
        self.verify_error_handling()
        self.verify_health_checks()
        self.verify_scalability()
        self.verify_backup_strategy()

        return self.generate_report()

    def verify_environment_config(self):
        """Check environment variable configuration"""
        print("[1/10] Verifying environment configuration...")

        # Check for .env.example
        env_example = self.target_path / ".env.example"
        env_file = self.target_path / ".env"

        if env_file.exists() and not env_example.exists():
            self.add_result(
                "Environment",
                "Missing .env.example",
                False,
                "medium",
                ".env exists but .env.example is missing",
                "Create .env.example with all required variables (without values)"
            )

        # Check for required environment variables
        required_vars = self._get_required_env_vars()
        if env_example.exists():
            with open(env_example) as f:
                example_content = f.read()
                missing_vars = [var for var in required_vars if var not in example_content]

                if missing_vars:
                    self.add_result(
                        "Environment",
                        "Missing required variables",
                        False,
                        "high",
                        f"Missing variables in .env.example: {', '.join(missing_vars)}",
                        "Add all required environment variables to .env.example"
                    )
                else:
                    self.add_result(
                        "Environment",
                        "Environment variables",
                        True,
                        "info",
                        "All required variables documented"
                    )

        # Check for environment-specific configs
        config_files = list(self.target_path.glob("config/*.js")) + list(self.target_path.glob("config/*.json"))
        has_env_configs = any(env in str(f) for f in config_files for env in ["production", "staging", "development"])

        if not has_env_configs:
            self.add_result(
                "Environment",
                "Environment-specific configs",
                False,
                "medium",
                "No environment-specific configuration files found",
                "Create separate configs for production/staging/development"
            )

    def verify_dependencies(self):
        """Verify dependency management"""
        print("[2/10] Verifying dependencies...")

        package_json = self.target_path / "package.json"
        if not package_json.exists():
            return

        with open(package_json) as f:
            pkg = json.load(f)

        # Check for lockfile
        has_lockfile = (self.target_path / "package-lock.json").exists() or \
                      (self.target_path / "yarn.lock").exists() or \
                      (self.target_path / "pnpm-lock.yaml").exists()

        if not has_lockfile:
            self.add_result(
                "Dependencies",
                "Missing lockfile",
                False,
                "critical",
                "No package lockfile found",
                "Commit package-lock.json/yarn.lock to ensure deterministic builds"
            )
        else:
            self.add_result(
                "Dependencies",
                "Lockfile present",
                True,
                "info",
                "Dependency lockfile exists"
            )

        # Check for production dependencies
        deps = pkg.get("dependencies", {})
        dev_deps = pkg.get("devDependencies", {})

        # Common dev dependencies that should NOT be in production
        dev_only = ["nodemon", "ts-node", "tsx", "@types/"]
        misplaced = [dep for dep in deps if any(d in dep for d in dev_only)]

        if misplaced:
            self.add_result(
                "Dependencies",
                "Misplaced dev dependencies",
                False,
                "medium",
                f"Dev dependencies in production: {', '.join(misplaced)}",
                "Move dev-only dependencies to devDependencies"
            )

    def verify_build_process(self):
        """Verify build configuration"""
        print("[3/10] Verifying build process...")

        package_json = self.target_path / "package.json"
        if not package_json.exists():
            return

        with open(package_json) as f:
            pkg = json.load(f)

        scripts = pkg.get("scripts", {})

        # Check for required scripts
        required_scripts = {
            "build": "Production build script",
            "start": "Production start script",
        }

        for script, desc in required_scripts.items():
            if script not in scripts:
                self.add_result(
                    "Build",
                    f"Missing {script} script",
                    False,
                    "high",
                    f"No {script} script found",
                    f"Add '{script}' script to package.json for {desc.lower()}"
                )

        # Check for TypeScript compilation
        tsconfig = self.target_path / "tsconfig.json"
        if tsconfig.exists():
            if "build" in scripts and "tsc" not in scripts["build"]:
                self.add_result(
                    "Build",
                    "TypeScript compilation",
                    False,
                    "high",
                    "TypeScript project but build doesn't run tsc",
                    "Ensure build script compiles TypeScript (tsc or alternative bundler)"
                )

        # Check for build output directory
        dist_dirs = ["dist", "build", "out", ".next"]
        has_dist = any((self.target_path / d).exists() for d in dist_dirs)

        if not has_dist and "build" in scripts:
            self.add_result(
                "Build",
                "Build output",
                False,
                "medium",
                "No build output directory found",
                "Run build script to generate production build"
            )

    def verify_database_config(self):
        """Verify database configuration"""
        print("[4/10] Verifying database configuration...")

        # Check for database connection pooling
        has_pooling = self._file_contains_pattern(
            [".js", ".ts"],
            ["pool", "maxConnections", "connectionLimit"]
        )

        if not has_pooling:
            self.add_result(
                "Database",
                "Connection pooling",
                False,
                "high",
                "No database connection pooling detected",
                "Configure connection pooling for production database"
            )

        # Check for migration files
        migration_dirs = ["migrations", "prisma/migrations", "db/migrations"]
        has_migrations = any((self.target_path / d).exists() for d in migration_dirs)

        if not has_migrations:
            self.add_result(
                "Database",
                "Database migrations",
                False,
                "medium",
                "No migration directory found",
                "Use database migrations for schema changes"
            )

        # Check for connection timeout configuration
        has_timeout = self._file_contains_pattern(
            [".js", ".ts"],
            ["connectTimeout", "connectionTimeout", "acquireTimeout"]
        )

        if not has_timeout:
            self.add_result(
                "Database",
                "Connection timeouts",
                False,
                "medium",
                "No connection timeout configuration found",
                "Configure connection timeouts to prevent hanging connections"
            )

    def verify_monitoring(self):
        """Verify monitoring configuration"""
        print("[5/10] Verifying monitoring setup...")

        # Check for APM/monitoring tools
        monitoring_tools = [
            "newrelic", "datadog", "sentry", "@sentry/node",
            "prometheus", "prom-client", "opentelemetry"
        ]

        package_json = self.target_path / "package.json"
        has_monitoring = False

        if package_json.exists():
            with open(package_json) as f:
                pkg = json.load(f)
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                has_monitoring = any(tool in deps for tool in monitoring_tools)

        if not has_monitoring:
            self.add_result(
                "Monitoring",
                "APM/monitoring",
                False,
                "high",
                "No monitoring/APM tool detected",
                "Install monitoring tool (Sentry, Datadog, New Relic, etc.)"
            )
        else:
            self.add_result(
                "Monitoring",
                "APM/monitoring",
                True,
                "info",
                "Monitoring tool detected"
            )

        # Check for metrics collection
        has_metrics = self._file_contains_pattern(
            [".js", ".ts"],
            ["metrics", "gauge", "counter", "histogram"]
        )

        if not has_metrics:
            self.add_result(
                "Monitoring",
                "Metrics collection",
                False,
                "medium",
                "No metrics collection detected",
                "Implement metrics collection for monitoring"
            )

    def verify_logging(self):
        """Verify logging configuration"""
        print("[6/10] Verifying logging setup...")

        # Check for structured logging
        logging_libs = ["winston", "pino", "bunyan", "log4js"]

        package_json = self.target_path / "package.json"
        has_logger = False

        if package_json.exists():
            with open(package_json) as f:
                pkg = json.load(f)
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                has_logger = any(lib in deps for lib in logging_libs)

        if not has_logger:
            self.add_result(
                "Logging",
                "Structured logging",
                False,
                "high",
                "No structured logging library detected",
                "Use winston, pino, or bunyan for production logging"
            )

        # Check for log levels
        has_log_levels = self._file_contains_pattern(
            [".js", ".ts"],
            ["logger.error", "logger.warn", "logger.info", "logger.debug"]
        )

        if not has_log_levels:
            self.add_result(
                "Logging",
                "Log levels",
                False,
                "medium",
                "No log level usage detected",
                "Use appropriate log levels (error, warn, info, debug)"
            )

    def verify_error_handling(self):
        """Verify error handling"""
        print("[7/10] Verifying error handling...")

        # Check for global error handlers
        has_error_middleware = self._file_contains_pattern(
            [".js", ".ts"],
            ["app.use.*error", "errorHandler", "errorMiddleware"]
        )

        if not has_error_middleware:
            self.add_result(
                "Error Handling",
                "Global error handler",
                False,
                "high",
                "No global error handling middleware detected",
                "Implement global error handling middleware"
            )

        # Check for uncaught exception handlers
        has_uncaught_handlers = self._file_contains_pattern(
            [".js", ".ts"],
            ["uncaughtException", "unhandledRejection"]
        )

        if not has_uncaught_handlers:
            self.add_result(
                "Error Handling",
                "Uncaught exception handlers",
                False,
                "critical",
                "No uncaught exception/rejection handlers",
                "Add process.on('uncaughtException') and process.on('unhandledRejection') handlers"
            )

    def verify_health_checks(self):
        """Verify health check endpoints"""
        print("[8/10] Verifying health checks...")

        # Check for health/readiness endpoints
        has_health = self._file_contains_pattern(
            [".js", ".ts"],
            ["/health", "/healthz", "/ready", "/readiness", "/liveness"]
        )

        if not has_health:
            self.add_result(
                "Health Checks",
                "Health endpoints",
                False,
                "high",
                "No health check endpoints detected",
                "Implement /health and /ready endpoints for orchestration"
            )
        else:
            self.add_result(
                "Health Checks",
                "Health endpoints",
                True,
                "info",
                "Health check endpoints detected"
            )

    def verify_scalability(self):
        """Verify scalability configuration"""
        print("[9/10] Verifying scalability...")

        # Check for clustering/multi-process setup
        has_clustering = self._file_contains_pattern(
            [".js", ".ts"],
            ["cluster", "pm2", "throng"]
        )

        if not has_clustering and self.environment == "production":
            self.add_result(
                "Scalability",
                "Process clustering",
                False,
                "medium",
                "No clustering/multi-process setup detected",
                "Use PM2, cluster module, or container orchestration for multi-process"
            )

        # Check for stateless design (no file-based sessions)
        has_file_sessions = self._file_contains_pattern(
            [".js", ".ts"],
            ["session.*FileStore", "session.*file"]
        )

        if has_file_sessions:
            self.add_result(
                "Scalability",
                "Stateless sessions",
                False,
                "high",
                "File-based sessions detected (not stateless)",
                "Use Redis or database for session storage in production"
            )

    def verify_backup_strategy(self):
        """Verify backup and recovery strategy"""
        print("[10/10] Verifying backup strategy...")

        # Check for backup documentation
        backup_docs = ["BACKUP.md", "docs/backup.md", "DISASTER-RECOVERY.md"]
        has_backup_docs = any((self.target_path / d).exists() for d in backup_docs)

        if not has_backup_docs:
            self.add_result(
                "Backup",
                "Backup documentation",
                False,
                "high",
                "No backup/disaster recovery documentation",
                "Document backup strategy and recovery procedures"
            )

        # Check for database backup scripts
        backup_scripts = list(self.target_path.glob("scripts/*backup*")) + \
                        list(self.target_path.glob("scripts/*restore*"))

        if not backup_scripts:
            self.add_result(
                "Backup",
                "Backup scripts",
                False,
                "medium",
                "No backup scripts found",
                "Create automated backup scripts"
            )

    def add_result(self, category: str, check: str, passed: bool, severity: str, message: str, recommendation: str = ""):
        """Add verification result"""
        self.results.append(
            VerificationResult(category, check, passed, severity, message, recommendation)
        )

    def _get_required_env_vars(self) -> List[str]:
        """Get list of required environment variables"""
        return [
            "NODE_ENV",
            "PORT",
            "DATABASE_URL",
            "LOG_LEVEL",
        ]

    def _file_contains_pattern(self, extensions: List[str], patterns: List[str]) -> bool:
        """Check if any file contains any of the patterns"""
        for ext in extensions:
            for file_path in self.target_path.rglob(f"*{ext}"):
                if "node_modules" in str(file_path) or "test" in str(file_path):
                    continue

                try:
                    with open(file_path) as f:
                        content = f.read()
                        if any(pattern in content for pattern in patterns):
                            return True
                except:
                    pass

        return False

    def generate_report(self) -> Dict:
        """Generate verification report"""
        # Count by severity
        critical = [r for r in self.results if r.severity == "critical" and not r.passed]
        high = [r for r in self.results if r.severity == "high" and not r.passed]
        medium = [r for r in self.results if r.severity == "medium" and not r.passed]
        low = [r for r in self.results if r.severity == "low" and not r.passed]

        passed = len([r for r in self.results if r.passed])
        total = len(self.results)

        ready = len(critical) == 0 and len(high) == 0

        report = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "ready_for_deployment": ready,
            "checks_passed": passed,
            "checks_total": total,
            "severity_counts": {
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "low": len(low),
            },
            "results": [asdict(r) for r in self.results],
        }

        self.print_report(report)
        return report

    def print_report(self, report: Dict):
        """Print formatted report"""
        print(f"\n{'='*60}")
        print("Deployment Verification Report")
        print(f"{'='*60}\n")

        print(f"Environment: {report['environment']}")
        print(f"Checks Passed: {report['checks_passed']}/{report['checks_total']}\n")

        print("Severity Breakdown:")
        print(f"  Critical: {report['severity_counts']['critical']}")
        print(f"  High:     {report['severity_counts']['high']}")
        print(f"  Medium:   {report['severity_counts']['medium']}")
        print(f"  Low:      {report['severity_counts']['low']}\n")

        # Group by category
        by_category = {}
        for result in self.results:
            if result.category not in by_category:
                by_category[result.category] = []
            by_category[result.category].append(result)

        for category, results in by_category.items():
            print(f"{category}:")
            for result in results:
                status = "✅" if result.passed else "❌"
                print(f"  {status} {result.check}: {result.message}")
                if result.recommendation and not result.passed:
                    print(f"      → {result.recommendation}")
            print()

        print(f"{'='*60}")
        if report["ready_for_deployment"]:
            print("✅ DEPLOYMENT VERIFICATION PASSED")
        else:
            print("❌ DEPLOYMENT VERIFICATION FAILED")
            print("\nBlocking Issues:")
            for result in self.results:
                if not result.passed and result.severity in ["critical", "high"]:
                    print(f"  - {result.category}: {result.message}")
        print(f"{'='*60}\n")


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: deployment-verifier.py <target_path> [environment]")
        print("\nEnvironment: staging | production (default: production)")
        sys.exit(1)

    target_path = sys.argv[1]
    environment = sys.argv[2] if len(sys.argv) > 2 else "production"

    verifier = DeploymentVerifier(target_path, environment)
    report = verifier.verify_all()

    # Save report
    output_path = f"deployment-verification-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to: {output_path}")

    sys.exit(0 if report["ready_for_deployment"] else 1)


if __name__ == "__main__":
    main()
