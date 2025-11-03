#!/usr/bin/env python3

"""
Stack Trace Analyzer

Analyzes stack traces to identify root causes, common failure points,
and provides intelligent recommendations for debugging.

Usage:
    python stack-trace-analyzer.py --input <trace-file>
    python stack-trace-analyzer.py --clipboard
    cat error.log | python stack-trace-analyzer.py --stdin
"""

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Set


@dataclass
class StackFrame:
    """Represents a single frame in a stack trace"""
    file: str
    function: str
    line: Optional[int]
    code: Optional[str]
    is_third_party: bool = False


@dataclass
class StackTrace:
    """Represents a complete stack trace"""
    exception_type: str
    message: str
    frames: List[StackFrame]
    raw_text: str


class StackTraceAnalyzer:
    """Intelligent stack trace analysis"""

    # Common third-party library patterns
    THIRD_PARTY_PATTERNS = [
        r'node_modules',
        r'site-packages',
        r'vendor',
        r'dist-packages',
        r'\.gem',
        r'@[a-zA-Z0-9_-]+/',  # npm scoped packages
    ]

    # Exception type patterns for different languages
    EXCEPTION_PATTERNS = {
        'python': r'(\w+(?:\.\w+)*Error|Exception): (.+)',
        'javascript': r'(\w+(?:Error|Exception)): (.+)',
        'java': r'(\w+(?:\.\w+)*Exception): (.+)',
        'ruby': r'(\w+(?:::\w+)*Error): (.+)',
    }

    # Stack frame patterns for different languages
    FRAME_PATTERNS = {
        'python': r'\s+File "([^"]+)", line (\d+), in (\w+)\s*(?:\n\s+(.+))?',
        'javascript': r'\s+at (?:(\w+) \()?([^:]+):(\d+):(\d+)\)?',
        'java': r'\s+at ([\w.$]+)\.(\w+)\(([^:]+):(\d+)\)',
        'ruby': r'\s+from ([^:]+):(\d+):in `([^\']+)\'',
    }

    def __init__(self):
        self.traces: List[StackTrace] = []
        self.language: Optional[str] = None

    def parse_input(self, text: str) -> None:
        """Parse stack trace(s) from input text"""
        # Detect language
        self.language = self._detect_language(text)
        print(f"🔍 Detected language: {self.language or 'unknown'}")

        # Split into individual traces (some logs may contain multiple)
        trace_blocks = self._split_traces(text)
        print(f"📚 Found {len(trace_blocks)} stack trace(s)")

        for i, block in enumerate(trace_blocks, 1):
            try:
                trace = self._parse_trace(block)
                if trace:
                    self.traces.append(trace)
                    print(f"✅ Parsed trace {i}: {trace.exception_type}")
            except Exception as e:
                print(f"⚠️  Error parsing trace {i}: {e}", file=sys.stderr)

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect programming language from stack trace"""
        patterns = {
            'python': r'File ".*\.py", line \d+',
            'javascript': r'at.*\.js:\d+:\d+',
            'java': r'at [\w.$]+\.\w+\([^:]+\.java:\d+\)',
            'ruby': r'from .*\.rb:\d+:in',
        }

        for lang, pattern in patterns.items():
            if re.search(pattern, text):
                return lang

        return None

    def _split_traces(self, text: str) -> List[str]:
        """Split input into individual stack traces"""
        # Look for exception headers
        exception_starts = []

        for lang, pattern in self.EXCEPTION_PATTERNS.items():
            for match in re.finditer(pattern, text, re.MULTILINE):
                exception_starts.append(match.start())

        if not exception_starts:
            # No clear exception headers, treat as single trace
            return [text]

        # Split at exception headers
        exception_starts.sort()
        traces = []
        for i, start in enumerate(exception_starts):
            end = exception_starts[i + 1] if i + 1 < len(exception_starts) else len(text)
            traces.append(text[start:end].strip())

        return traces

    def _parse_trace(self, text: str) -> Optional[StackTrace]:
        """Parse a single stack trace"""
        # Extract exception info
        exception_type = "Unknown"
        message = ""

        if self.language:
            pattern = self.EXCEPTION_PATTERNS.get(self.language)
            if pattern:
                match = re.search(pattern, text)
                if match:
                    exception_type = match.group(1)
                    message = match.group(2).strip()

        # Extract frames
        frames = self._parse_frames(text)

        if not frames:
            return None

        return StackTrace(
            exception_type=exception_type,
            message=message,
            frames=frames,
            raw_text=text
        )

    def _parse_frames(self, text: str) -> List[StackFrame]:
        """Parse stack frames from trace"""
        frames = []

        if not self.language:
            return frames

        pattern = self.FRAME_PATTERNS.get(self.language)
        if not pattern:
            return frames

        if self.language == 'python':
            for match in re.finditer(pattern, text, re.MULTILINE):
                file_path = match.group(1)
                line_num = int(match.group(2))
                function = match.group(3)
                code = match.group(4) if match.lastindex >= 4 else None

                frames.append(StackFrame(
                    file=file_path,
                    function=function,
                    line=line_num,
                    code=code.strip() if code else None,
                    is_third_party=self._is_third_party(file_path)
                ))

        elif self.language == 'javascript':
            for match in re.finditer(pattern, text, re.MULTILINE):
                function = match.group(1) or '<anonymous>'
                file_path = match.group(2)
                line_num = int(match.group(3))

                frames.append(StackFrame(
                    file=file_path,
                    function=function,
                    line=line_num,
                    code=None,
                    is_third_party=self._is_third_party(file_path)
                ))

        # Similar parsing for other languages...

        return frames

    def _is_third_party(self, file_path: str) -> bool:
        """Check if file is from third-party library"""
        for pattern in self.THIRD_PARTY_PATTERNS:
            if re.search(pattern, file_path):
                return True
        return False

    def analyze(self) -> Dict:
        """Perform comprehensive analysis of stack traces"""
        print("\n🔬 Analyzing stack traces...")

        analysis = {
            'total_traces': len(self.traces),
            'exception_types': self._analyze_exception_types(),
            'failure_points': self._identify_failure_points(),
            'root_causes': self._identify_root_causes(),
            'patterns': self._detect_patterns(),
            'recommendations': self._generate_recommendations()
        }

        return analysis

    def _analyze_exception_types(self) -> Dict:
        """Analyze distribution of exception types"""
        exception_counts = Counter(trace.exception_type for trace in self.traces)

        return {
            'distribution': dict(exception_counts),
            'most_common': exception_counts.most_common(5)
        }

    def _identify_failure_points(self) -> List[Dict]:
        """Identify most common failure points in code"""
        failure_points = defaultdict(list)

        for trace in self.traces:
            if trace.frames:
                # Get first non-third-party frame (likely the actual failure point)
                for frame in trace.frames:
                    if not frame.is_third_party:
                        key = f"{frame.file}:{frame.function}:{frame.line}"
                        failure_points[key].append(trace.exception_type)
                        break

        # Sort by frequency
        sorted_points = sorted(
            failure_points.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        return [
            {
                'location': location,
                'count': len(exceptions),
                'exception_types': list(set(exceptions))
            }
            for location, exceptions in sorted_points[:10]
        ]

    def _identify_root_causes(self) -> List[Dict]:
        """Identify likely root causes based on patterns"""
        root_causes = []

        # Analyze common exception messages
        message_patterns = defaultdict(int)
        for trace in self.traces:
            # Extract key parts of error message
            if 'null' in trace.message.lower() or 'none' in trace.message.lower():
                message_patterns['null_pointer'] += 1
            elif 'index' in trace.message.lower() or 'bounds' in trace.message.lower():
                message_patterns['index_out_of_bounds'] += 1
            elif 'timeout' in trace.message.lower():
                message_patterns['timeout'] += 1
            elif 'permission' in trace.message.lower() or 'denied' in trace.message.lower():
                message_patterns['permission_denied'] += 1
            elif 'file' in trace.message.lower() or 'directory' in trace.message.lower():
                message_patterns['file_system'] += 1

        for pattern, count in sorted(message_patterns.items(), key=lambda x: x[1], reverse=True):
            root_causes.append({
                'category': pattern,
                'count': count,
                'percentage': count / len(self.traces) * 100
            })

        return root_causes

    def _detect_patterns(self) -> List[str]:
        """Detect patterns across multiple stack traces"""
        patterns = []

        # Check for recurring call chains
        call_chains = Counter()
        for trace in self.traces:
            if len(trace.frames) >= 3:
                # Create signature from top 3 frames
                chain = " -> ".join(f.function for f in trace.frames[:3])
                call_chains[chain] += 1

        for chain, count in call_chains.items():
            if count > 1:
                patterns.append(f"Recurring call chain ({count}x): {chain}")

        # Check for similar error messages
        if len(self.traces) > 1:
            messages = [trace.message for trace in self.traces]
            # Simple similarity check (could be enhanced)
            unique_messages = len(set(messages))
            if unique_messages < len(messages) / 2:
                patterns.append(f"High message similarity: {unique_messages} unique out of {len(messages)} total")

        return patterns

    def _generate_recommendations(self) -> List[str]:
        """Generate debugging recommendations based on analysis"""
        recommendations = []

        if not self.traces:
            return ["No stack traces to analyze"]

        # Based on exception types
        exception_types = [trace.exception_type for trace in self.traces]

        if 'NullPointerException' in exception_types or 'NoneType' in str(exception_types):
            recommendations.append("🔍 Add null/None checks before accessing object properties")
            recommendations.append("💡 Use optional chaining or safe navigation operators")

        if 'IndexError' in exception_types or 'ArrayIndexOutOfBounds' in exception_types:
            recommendations.append("🔍 Validate array/list indices before access")
            recommendations.append("💡 Use length checks or bounds validation")

        if 'TimeoutError' in exception_types or 'timeout' in str(exception_types).lower():
            recommendations.append("🔍 Increase timeout values or optimize slow operations")
            recommendations.append("💡 Add retry logic with exponential backoff")

        # Based on failure points
        failure_points = self._identify_failure_points()
        if failure_points and failure_points[0]['count'] > len(self.traces) / 2:
            recommendations.append(f"🎯 Focus debugging on: {failure_points[0]['location']}")

        # General recommendations
        recommendations.append("📝 Add comprehensive logging at failure points")
        recommendations.append("🧪 Create unit tests to reproduce the error")
        recommendations.append("🔄 Consider adding error handling and graceful degradation")

        return recommendations

    def print_summary(self) -> None:
        """Print human-readable summary"""
        analysis = self.analyze()

        print("\n" + "="*70)
        print("📊 STACK TRACE ANALYSIS SUMMARY")
        print("="*70)

        print(f"\n📈 Overview:")
        print(f"  Total traces analyzed: {analysis['total_traces']}")
        print(f"  Detected language: {self.language or 'unknown'}")

        if analysis['exception_types']['most_common']:
            print(f"\n🔝 Most Common Exceptions:")
            for exc_type, count in analysis['exception_types']['most_common']:
                print(f"  {exc_type}: {count} ({count/analysis['total_traces']*100:.1f}%)")

        if analysis['failure_points']:
            print(f"\n🎯 Top Failure Points:")
            for i, point in enumerate(analysis['failure_points'][:5], 1):
                print(f"  {i}. {point['location']} ({point['count']}x)")
                print(f"     Exceptions: {', '.join(point['exception_types'])}")

        if analysis['root_causes']:
            print(f"\n🔍 Root Cause Categories:")
            for cause in analysis['root_causes']:
                print(f"  {cause['category']}: {cause['count']} ({cause['percentage']:.1f}%)")

        if analysis['patterns']:
            print(f"\n🔄 Detected Patterns:")
            for pattern in analysis['patterns']:
                print(f"  • {pattern}")

        if analysis['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  {rec}")


def main():
    parser = argparse.ArgumentParser(description='Analyze stack traces for debugging insights')
    parser.add_argument('--input', type=Path, help='Input file containing stack trace')
    parser.add_argument('--clipboard', action='store_true', help='Read from clipboard')
    parser.add_argument('--stdin', action='store_true', help='Read from stdin')

    args = parser.parse_args()

    # Read input
    text = None
    if args.input:
        if not args.input.exists():
            print(f"❌ Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        text = args.input.read_text(encoding='utf-8', errors='ignore')
    elif args.clipboard:
        try:
            import pyperclip
            text = pyperclip.paste()
        except ImportError:
            print("❌ Error: pyperclip not installed. Run: pip install pyperclip", file=sys.stderr)
            sys.exit(1)
    elif args.stdin or not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("❌ Error: No input provided. Use --input, --clipboard, or --stdin", file=sys.stderr)
        sys.exit(1)

    if not text or not text.strip():
        print("❌ Error: Empty input", file=sys.stderr)
        sys.exit(1)

    # Analyze
    analyzer = StackTraceAnalyzer()
    analyzer.parse_input(text)
    analyzer.print_summary()

    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
