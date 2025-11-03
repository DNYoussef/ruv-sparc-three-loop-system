#!/usr/bin/env python3
"""
Failure Detection Script for CI/CD Intelligent Recovery
Real-time failure monitoring with pattern recognition
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class FailurePattern:
    """Detected failure pattern"""
    pattern_type: str
    occurrences: int
    files: List[str]
    severity: str  # critical, high, medium, low
    description: str
    first_seen: str
    last_seen: str


@dataclass
class FailureMonitor:
    """Real-time failure monitoring and pattern detection"""
    artifacts_dir: Path
    patterns: Dict[str, FailurePattern] = field(default_factory=dict)
    failure_history: List[Dict] = field(default_factory=list)

    def __init__(self, artifacts_dir: str = ".claude/.artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Load existing patterns if available
        self.load_patterns()

    def load_patterns(self):
        """Load historical failure patterns"""
        patterns_file = self.artifacts_dir / "failure-patterns-history.json"

        if patterns_file.exists():
            with open(patterns_file) as f:
                data = json.load(f)

            for pattern_data in data.get("patterns", []):
                pattern = FailurePattern(**pattern_data)
                self.patterns[pattern.pattern_type] = pattern

    def save_patterns(self):
        """Save failure patterns to disk"""
        patterns_file = self.artifacts_dir / "failure-patterns-history.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "patterns": [
                {
                    "pattern_type": p.pattern_type,
                    "occurrences": p.occurrences,
                    "files": p.files,
                    "severity": p.severity,
                    "description": p.description,
                    "first_seen": p.first_seen,
                    "last_seen": p.last_seen
                }
                for p in self.patterns.values()
            ]
        }

        with open(patterns_file, 'w') as f:
            json.dump(data, f, indent=2)

    def detect_pattern_type(self, error_message: str) -> str:
        """Classify error message into pattern type"""
        msg_lower = error_message.lower()

        # Pattern detection rules
        patterns = {
            "null_pointer": [r"\bnull\b", r"\bundefined\b", r"cannot read property"],
            "type_error": [r"type error", r"expected .+, got", r"invalid type"],
            "async_error": [r"promise", r"async", r"await", r"callback"],
            "import_error": [r"cannot find module", r"import error", r"module not found"],
            "assertion_error": [r"expected .+ to (be|equal)", r"assertion failed"],
            "timeout_error": [r"timeout", r"timed out", r"exceeded"],
            "network_error": [r"network", r"econnrefused", r"fetch failed"],
            "database_error": [r"database", r"sql", r"connection refused"],
            "permission_error": [r"permission", r"access denied", r"forbidden"],
            "syntax_error": [r"syntax error", r"unexpected token", r"parse error"]
        }

        for pattern_type, regexes in patterns.items():
            for regex in regexes:
                if re.search(regex, msg_lower):
                    return pattern_type

        return "unknown"

    def calculate_severity(self, failure: Dict) -> str:
        """Calculate failure severity based on impact"""
        pattern_type = self.detect_pattern_type(failure.get("errorMessage", ""))

        # Critical patterns
        if pattern_type in ["database_error", "permission_error"]:
            return "critical"

        # High severity
        if pattern_type in ["null_pointer", "syntax_error"]:
            return "high"

        # Medium severity
        if pattern_type in ["type_error", "async_error", "import_error"]:
            return "medium"

        # Low severity
        return "low"

    def update_pattern(self, failure: Dict):
        """Update or create failure pattern"""
        pattern_type = self.detect_pattern_type(failure.get("errorMessage", ""))
        severity = self.calculate_severity(failure)
        timestamp = datetime.now().isoformat()

        if pattern_type in self.patterns:
            # Update existing pattern
            pattern = self.patterns[pattern_type]
            pattern.occurrences += 1
            pattern.last_seen = timestamp

            # Add file if not already tracked
            file_path = failure.get("file", "")
            if file_path and file_path not in pattern.files:
                pattern.files.append(file_path)

            # Escalate severity if needed
            severity_order = ["low", "medium", "high", "critical"]
            if severity_order.index(severity) > severity_order.index(pattern.severity):
                pattern.severity = severity
        else:
            # Create new pattern
            self.patterns[pattern_type] = FailurePattern(
                pattern_type=pattern_type,
                occurrences=1,
                files=[failure.get("file", "")],
                severity=severity,
                description=self._generate_description(pattern_type),
                first_seen=timestamp,
                last_seen=timestamp
            )

    def _generate_description(self, pattern_type: str) -> str:
        """Generate human-readable pattern description"""
        descriptions = {
            "null_pointer": "Null or undefined value access",
            "type_error": "Type mismatch or invalid type usage",
            "async_error": "Asynchronous operation handling issue",
            "import_error": "Module import or dependency issue",
            "assertion_error": "Test assertion failure",
            "timeout_error": "Operation timeout or performance issue",
            "network_error": "Network connectivity or API issue",
            "database_error": "Database connection or query issue",
            "permission_error": "Authorization or access control issue",
            "syntax_error": "Code syntax error",
            "unknown": "Unclassified failure pattern"
        }
        return descriptions.get(pattern_type, "Unknown failure pattern")

    def detect_trends(self) -> List[Dict]:
        """Detect trending failure patterns"""
        trends = []

        for pattern in self.patterns.values():
            # High occurrence patterns
            if pattern.occurrences >= 5:
                trends.append({
                    "type": "high_frequency",
                    "pattern": pattern.pattern_type,
                    "occurrences": pattern.occurrences,
                    "severity": pattern.severity,
                    "recommendation": f"Investigate {pattern.pattern_type}: {pattern.occurrences} occurrences"
                })

            # Critical severity patterns
            if pattern.severity == "critical":
                trends.append({
                    "type": "critical_severity",
                    "pattern": pattern.pattern_type,
                    "files": pattern.files,
                    "recommendation": f"Immediate attention required: {pattern.description}"
                })

            # Widespread patterns (multiple files)
            if len(pattern.files) >= 3:
                trends.append({
                    "type": "widespread",
                    "pattern": pattern.pattern_type,
                    "file_count": len(pattern.files),
                    "recommendation": f"Systemic issue: affects {len(pattern.files)} files"
                })

        return trends

    def generate_alerts(self) -> List[Dict]:
        """Generate actionable alerts for detected patterns"""
        alerts = []

        for pattern in self.patterns.values():
            # Alert for critical issues
            if pattern.severity == "critical":
                alerts.append({
                    "level": "critical",
                    "pattern": pattern.pattern_type,
                    "message": f"CRITICAL: {pattern.description} in {len(pattern.files)} files",
                    "action": "Immediate investigation required",
                    "files": pattern.files
                })

            # Alert for high frequency
            if pattern.occurrences >= 10:
                alerts.append({
                    "level": "warning",
                    "pattern": pattern.pattern_type,
                    "message": f"High frequency: {pattern.occurrences} occurrences of {pattern.pattern_type}",
                    "action": "Consider systematic fix",
                    "files": pattern.files
                })

        return alerts

    def analyze_failures(self, failures: List[Dict]) -> Dict:
        """
        Analyze failures and detect patterns

        Returns comprehensive analysis with patterns, trends, and alerts
        """
        print("=== Failure Detection Analysis ===\n")

        # Update patterns for each failure
        print(f"Analyzing {len(failures)} failures...")
        for failure in failures:
            self.update_pattern(failure)
            self.failure_history.append(failure)

        # Detect trends
        print("Detecting failure trends...")
        trends = self.detect_trends()

        # Generate alerts
        print("Generating alerts...")
        alerts = self.generate_alerts()

        # Build summary
        pattern_summary = [
            {
                "type": p.pattern_type,
                "occurrences": p.occurrences,
                "severity": p.severity,
                "files": len(p.files),
                "description": p.description
            }
            for p in sorted(
                self.patterns.values(),
                key=lambda x: (x.severity != "critical", x.severity != "high", -x.occurrences)
            )
        ]

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_failures": len(failures),
                "unique_patterns": len(self.patterns),
                "critical_patterns": sum(1 for p in self.patterns.values() if p.severity == "critical"),
                "high_severity": sum(1 for p in self.patterns.values() if p.severity == "high")
            },
            "patterns": pattern_summary,
            "trends": trends,
            "alerts": alerts
        }

        # Save analysis
        output_file = self.artifacts_dir / "failure-detection-analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        # Save patterns history
        self.save_patterns()

        # Print summary
        print(f"\n✅ Analysis complete")
        print(f"   Total failures: {analysis['summary']['total_failures']}")
        print(f"   Unique patterns: {analysis['summary']['unique_patterns']}")
        print(f"   Critical: {analysis['summary']['critical_patterns']}")
        print(f"   High severity: {analysis['summary']['high_severity']}")

        if alerts:
            print(f"\n⚠️  {len(alerts)} alerts generated:")
            for alert in alerts[:5]:  # Show first 5
                print(f"   [{alert['level'].upper()}] {alert['message']}")

        print(f"\nSaved to: {output_file}\n")

        return analysis


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        artifacts_dir = sys.argv[1]
    else:
        artifacts_dir = ".claude/.artifacts"

    try:
        # Load parsed failures
        failures_file = Path(artifacts_dir) / "parsed-failures.json"

        if not failures_file.exists():
            print(f"ERROR: Failures file not found: {failures_file}", file=sys.stderr)
            sys.exit(1)

        with open(failures_file) as f:
            failures = json.load(f)

        # Analyze
        monitor = FailureMonitor(artifacts_dir)
        analysis = monitor.analyze_failures(failures)

        # Exit with warning if critical issues found
        if analysis["summary"]["critical_patterns"] > 0:
            sys.exit(2)  # Warning exit code

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
