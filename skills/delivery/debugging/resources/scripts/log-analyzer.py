#!/usr/bin/env python3

"""
Log Analyzer Script

Analyzes application logs to identify patterns, errors, and anomalies.
Supports multiple log formats and provides statistical analysis.

Usage:
    python log-analyzer.py --file <logfile> --format <format> [--output <report.json>]
    python log-analyzer.py --dir <logdir> --pattern "*.log" [--time-range "2024-01-01,2024-01-31"]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LogAnalyzer:
    """Comprehensive log file analysis tool"""

    # Common log patterns
    LOG_PATTERNS = {
        'apache': r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" (?P<status>\d+) (?P<size>\S+)',
        'nginx': r'(?P<ip>\S+) - \S+ \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" (?P<status>\d+) (?P<size>\d+)',
        'syslog': r'(?P<timestamp>\w+ \d+ \d+:\d+:\d+) (?P<host>\S+) (?P<process>\S+)\[(?P<pid>\d+)\]: (?P<message>.*)',
        'json': None,  # JSON logs handled separately
        'custom': r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?P<level>\w+)\] (?P<message>.*)'
    }

    # Error keywords for pattern matching
    ERROR_KEYWORDS = [
        'error', 'exception', 'fatal', 'critical', 'fail', 'timeout',
        'null pointer', 'segfault', 'core dump', 'stack trace', 'panic'
    ]

    def __init__(self, log_format: str = 'custom'):
        self.log_format = log_format
        self.pattern = self.LOG_PATTERNS.get(log_format)
        self.entries: List[Dict] = []
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []
        self.statistics = defaultdict(int)

    def parse_log_file(self, filepath: Path) -> None:
        """Parse a single log file"""
        print(f"📖 Parsing log file: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        entry = self._parse_line(line, line_num)
                        if entry:
                            self.entries.append(entry)
                            self._classify_entry(entry)
                    except Exception as e:
                        print(f"⚠️  Error parsing line {line_num}: {e}", file=sys.stderr)

            print(f"✅ Parsed {len(self.entries)} log entries")

        except Exception as e:
            print(f"❌ Error reading file {filepath}: {e}", file=sys.stderr)
            raise

    def _parse_line(self, line: str, line_num: int) -> Optional[Dict]:
        """Parse a single log line"""
        line = line.strip()
        if not line:
            return None

        # Handle JSON logs
        if self.log_format == 'json':
            try:
                entry = json.loads(line)
                entry['_line_num'] = line_num
                return entry
            except json.JSONDecodeError:
                return None

        # Handle regex-based logs
        if self.pattern:
            match = re.search(self.pattern, line)
            if match:
                entry = match.groupdict()
                entry['_line_num'] = line_num
                entry['_raw'] = line
                return entry

        # Fallback: treat as plain text
        return {
            '_line_num': line_num,
            '_raw': line,
            'message': line
        }

    def _classify_entry(self, entry: Dict) -> None:
        """Classify log entry by severity"""
        message = str(entry.get('message', entry.get('_raw', ''))).lower()
        level = str(entry.get('level', '')).lower()

        # Check for errors
        if 'error' in level or any(kw in message for kw in ['error', 'exception', 'fatal', 'critical']):
            self.errors.append(entry)
            self.statistics['errors'] += 1

        # Check for warnings
        elif 'warn' in level or 'warning' in message:
            self.warnings.append(entry)
            self.statistics['warnings'] += 1

        # Track HTTP status codes if present
        if 'status' in entry:
            status = int(entry['status'])
            if status >= 500:
                self.statistics['5xx_errors'] += 1
            elif status >= 400:
                self.statistics['4xx_errors'] += 1
            elif status >= 200 and status < 300:
                self.statistics['2xx_success'] += 1

    def analyze_patterns(self) -> Dict:
        """Analyze log patterns and extract insights"""
        print("\n🔍 Analyzing patterns...")

        analysis = {
            'total_entries': len(self.entries),
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'statistics': dict(self.statistics),
            'error_patterns': self._extract_error_patterns(),
            'temporal_distribution': self._analyze_temporal_distribution(),
            'top_errors': self._get_top_errors(10),
            'anomalies': self._detect_anomalies()
        }

        return analysis

    def _extract_error_patterns(self) -> List[Dict]:
        """Extract common error patterns"""
        error_messages = [
            str(e.get('message', e.get('_raw', '')))
            for e in self.errors
        ]

        # Extract stack traces
        stack_traces = []
        for msg in error_messages:
            if 'traceback' in msg.lower() or 'stack trace' in msg.lower():
                stack_traces.append(msg)

        # Find common error messages
        error_counter = Counter(error_messages)

        return [
            {'message': msg, 'count': count, 'percentage': count / len(self.errors) * 100}
            for msg, count in error_counter.most_common(10)
        ]

    def _analyze_temporal_distribution(self) -> Dict:
        """Analyze error distribution over time"""
        timestamps = []
        for entry in self.errors:
            ts = entry.get('timestamp')
            if ts:
                timestamps.append(ts)

        # Group by hour
        hourly_distribution = defaultdict(int)
        for ts in timestamps:
            try:
                # Try to parse timestamp
                hour = self._extract_hour(ts)
                if hour is not None:
                    hourly_distribution[hour] += 1
            except Exception:
                continue

        return dict(hourly_distribution)

    def _extract_hour(self, timestamp: str) -> Optional[int]:
        """Extract hour from various timestamp formats"""
        # Try common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%d/%b/%Y:%H:%M:%S',
            '%b %d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S'
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp.split('.')[0], fmt)
                return dt.hour
            except ValueError:
                continue

        return None

    def _get_top_errors(self, limit: int = 10) -> List[Dict]:
        """Get most frequent errors with context"""
        error_types = defaultdict(list)

        for error in self.errors:
            error_msg = str(error.get('message', error.get('_raw', '')))
            # Extract first line as error type
            error_type = error_msg.split('\n')[0][:100]
            error_types[error_type].append(error)

        top_errors = sorted(
            error_types.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:limit]

        return [
            {
                'type': error_type,
                'count': len(occurrences),
                'sample_lines': [e['_line_num'] for e in occurrences[:3]],
                'first_occurrence': occurrences[0]['_line_num'],
                'last_occurrence': occurrences[-1]['_line_num']
            }
            for error_type, occurrences in top_errors
        ]

    def _detect_anomalies(self) -> List[Dict]:
        """Detect anomalous patterns in logs"""
        anomalies = []

        # Detect error bursts (>10 errors in short timeframe)
        if len(self.errors) > 10:
            error_lines = [e['_line_num'] for e in self.errors]
            for i in range(len(error_lines) - 10):
                if error_lines[i + 10] - error_lines[i] < 100:
                    anomalies.append({
                        'type': 'error_burst',
                        'description': f'10+ errors in lines {error_lines[i]}-{error_lines[i+10]}',
                        'severity': 'high'
                    })

        # Detect repeating errors
        consecutive_errors = defaultdict(int)
        prev_msg = None
        for error in self.errors:
            msg = str(error.get('message', ''))[:50]
            if msg == prev_msg:
                consecutive_errors[msg] += 1
            prev_msg = msg

        for msg, count in consecutive_errors.items():
            if count > 5:
                anomalies.append({
                    'type': 'repeating_error',
                    'description': f'Error repeated {count} times: {msg}',
                    'severity': 'medium'
                })

        return anomalies

    def generate_report(self, output_path: Optional[Path] = None) -> Dict:
        """Generate comprehensive analysis report"""
        print("\n📊 Generating analysis report...")

        report = {
            'metadata': {
                'analyzed_at': datetime.now().isoformat(),
                'log_format': self.log_format,
                'total_entries': len(self.entries)
            },
            'summary': {
                'errors': len(self.errors),
                'warnings': len(self.warnings),
                'error_rate': len(self.errors) / len(self.entries) * 100 if self.entries else 0
            },
            'analysis': self.analyze_patterns()
        }

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Report saved to: {output_path}")

        return report

    def print_summary(self) -> None:
        """Print human-readable summary"""
        print("\n" + "="*60)
        print("📋 LOG ANALYSIS SUMMARY")
        print("="*60)

        print(f"\n📈 Statistics:")
        print(f"  Total entries: {len(self.entries)}")
        print(f"  Errors: {len(self.errors)} ({len(self.errors)/len(self.entries)*100:.2f}%)")
        print(f"  Warnings: {len(self.warnings)} ({len(self.warnings)/len(self.entries)*100:.2f}%)")

        if self.statistics:
            print(f"\n📊 Details:")
            for key, value in sorted(self.statistics.items()):
                print(f"  {key}: {value}")

        analysis = self.analyze_patterns()

        if analysis['top_errors']:
            print(f"\n🔝 Top Errors:")
            for i, error in enumerate(analysis['top_errors'][:5], 1):
                print(f"  {i}. [{error['count']}x] {error['type'][:80]}")

        if analysis['anomalies']:
            print(f"\n⚠️  Anomalies Detected:")
            for anomaly in analysis['anomalies']:
                print(f"  [{anomaly['severity'].upper()}] {anomaly['description']}")


def main():
    parser = argparse.ArgumentParser(description='Analyze log files for errors and patterns')
    parser.add_argument('--file', type=Path, help='Log file to analyze')
    parser.add_argument('--format', choices=['apache', 'nginx', 'syslog', 'json', 'custom'],
                        default='custom', help='Log format')
    parser.add_argument('--output', type=Path, help='Output report path (JSON)')

    args = parser.parse_args()

    if not args.file:
        print("❌ Error: --file is required", file=sys.stderr)
        sys.exit(1)

    if not args.file.exists():
        print(f"❌ Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    analyzer = LogAnalyzer(log_format=args.format)
    analyzer.parse_log_file(args.file)
    analyzer.generate_report(args.output)
    analyzer.print_summary()

    print("\n✅ Log analysis complete!")


if __name__ == '__main__':
    main()
