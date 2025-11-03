#!/usr/bin/env python3
"""
Optimization Suggester - AI-powered performance optimization recommendations

Analyzes profiling data and generates actionable optimization suggestions
based on detected patterns and best practices.
"""

import json
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    DATABASE = "database"
    ALGORITHM = "algorithm"
    CACHING = "caching"
    PARALLELIZATION = "parallelization"


@dataclass
class Optimization:
    """Represents a single optimization recommendation"""
    category: Category
    severity: Severity
    title: str
    description: str
    impact: str
    effort: str
    code_example: Optional[str] = None
    references: Optional[List[str]] = None
    estimated_improvement: Optional[str] = None

    def to_dict(self):
        data = asdict(self)
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        return data


class OptimizationSuggester:
    """Generates optimization suggestions based on performance data"""

    def __init__(self):
        self.optimizations: List[Optimization] = []

    def analyze_cpu_profile(self, cpu_data: Dict[str, Any]) -> List[Optimization]:
        """Analyze CPU profiling data and suggest optimizations"""
        suggestions = []

        if 'top_functions' in cpu_data:
            for func in cpu_data['top_functions'][:5]:
                time_percent = (func['cumulative_time'] / cpu_data.get('duration', 1)) * 100

                if time_percent > 30:
                    suggestions.append(Optimization(
                        category=Category.CPU,
                        severity=Severity.HIGH,
                        title=f"Hot path detected in {func['function']}",
                        description=f"Function {func['function']} consumes {time_percent:.1f}% of total execution time",
                        impact="High - Optimizing this function could significantly improve performance",
                        effort="Medium",
                        estimated_improvement=f"{time_percent * 0.5:.1f}% potential speedup",
                        code_example=self._get_cpu_optimization_example(func),
                        references=[
                            "https://wiki.python.org/moin/PythonSpeed/PerformanceTips",
                            "https://docs.python.org/3/library/profile.html"
                        ]
                    ))

                # Check for excessive function calls
                if func['calls'] > 10000:
                    suggestions.append(Optimization(
                        category=Category.ALGORITHM,
                        severity=Severity.MEDIUM,
                        title=f"Excessive calls to {func['function']}",
                        description=f"Function called {func['calls']:,} times. Consider caching or batching.",
                        impact="Medium - Reducing call frequency can improve performance",
                        effort="Low to Medium",
                        code_example="""
# Before: Expensive function called repeatedly
for item in items:
    result = expensive_function(item)
    process(result)

# After: Batch processing or memoization
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(item):
    # ... computation
    return result
                        """,
                        references=["https://docs.python.org/3/library/functools.html#functools.lru_cache"]
                    ))

        return suggestions

    def analyze_memory_profile(self, memory_data: Dict[str, Any]) -> List[Optimization]:
        """Analyze memory profiling data and suggest optimizations"""
        suggestions = []

        delta_mb = memory_data.get('delta_mb', 0)

        if delta_mb > 100:
            suggestions.append(Optimization(
                category=Category.MEMORY,
                severity=Severity.HIGH,
                title="Significant memory growth detected",
                description=f"Memory increased by {delta_mb:.1f}MB during execution",
                impact="High - May lead to OOM errors or swapping",
                effort="Medium to High",
                estimated_improvement=f"Reduce memory by {delta_mb * 0.3:.1f}MB",
                code_example="""
# Use generators instead of lists for large datasets
def process_large_file(filename):
    # Before: Loads entire file into memory
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            yield process(line)

    # After: Process line by line
    with open(filename) as f:
        for line in f:
            yield process(line)
                """,
                references=["https://wiki.python.org/moin/Generators"]
            ))

        if 'hotspots' in memory_data:
            for hotspot in memory_data['hotspots'][:3]:
                if hotspot['size_diff_mb'] > 10:
                    suggestions.append(Optimization(
                        category=Category.MEMORY,
                        severity=Severity.MEDIUM,
                        title=f"Memory hotspot: {hotspot.get('file', 'unknown')}",
                        description=f"Allocated {hotspot['size_diff_mb']:.1f}MB",
                        impact="Medium - Optimize memory allocations",
                        effort="Medium",
                        code_example="""
# Use __slots__ for classes with many instances
class Point:
    __slots__ = ['x', 'y']  # Reduces memory per instance

    def __init__(self, x, y):
        self.x = x
        self.y = y

# Use array.array for numeric data instead of lists
from array import array
numbers = array('i', [1, 2, 3, 4, 5])  # More memory efficient
                        """
                    ))

        return suggestions

    def analyze_io_profile(self, io_data: Dict[str, Any]) -> List[Optimization]:
        """Analyze I/O profiling data and suggest optimizations"""
        suggestions = []

        disk = io_data.get('disk', {})
        network = io_data.get('network', {})

        # Analyze disk I/O
        if disk.get('read_mb', 0) > 1000:
            suggestions.append(Optimization(
                category=Category.IO,
                severity=Severity.MEDIUM,
                title="High disk read activity",
                description=f"Read {disk['read_mb']:.1f}MB from disk",
                impact="Medium - I/O operations are blocking",
                effort="Medium",
                estimated_improvement="2-5x speedup with caching",
                code_example="""
import aiofiles
import asyncio

# Use async I/O for better concurrency
async def read_files(filenames):
    tasks = []
    for filename in filenames:
        tasks.append(read_file_async(filename))
    return await asyncio.gather(*tasks)

async def read_file_async(filename):
    async with aiofiles.open(filename, 'r') as f:
        return await f.read()

# Or use memory-mapped files for large files
import mmap

with open('large_file.dat', 'r+b') as f:
    mmapped_file = mmap.mmap(f.fileno(), 0)
    # Access file contents without reading into memory
                """,
                references=["https://docs.python.org/3/library/asyncio.html"]
            ))

        # Analyze network I/O
        if network.get('sent_mb', 0) + network.get('recv_mb', 0) > 500:
            suggestions.append(Optimization(
                category=Category.NETWORK,
                severity=Severity.MEDIUM,
                title="High network traffic",
                description=f"Network I/O: {network.get('sent_mb', 0):.1f}MB sent, {network.get('recv_mb', 0):.1f}MB received",
                impact="Medium - Network latency impacts performance",
                effort="Low to Medium",
                code_example="""
# Use connection pooling
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=0.3)
adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Use compression
import gzip
import json

data = json.dumps(large_object).encode()
compressed = gzip.compress(data)
# Send compressed data
                """
            ))

        return suggestions

    def analyze_bottlenecks(self, bottleneck_data: Dict[str, Any]) -> List[Optimization]:
        """Analyze bottleneck detection data"""
        suggestions = []

        bottlenecks = bottleneck_data.get('bottlenecks', [])

        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'event_loop_lag':
                suggestions.append(Optimization(
                    category=Category.PARALLELIZATION,
                    severity=Severity.HIGH,
                    title="Event loop blocking detected",
                    description=f"Event loop lag: {bottleneck.get('lag', 0)}ms",
                    impact="High - Blocks other async operations",
                    effort="Medium",
                    code_example="""
# Move CPU-intensive work to thread pool
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def cpu_intensive_operation(data):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, heavy_computation, data)
    return result

def heavy_computation(data):
    # CPU-intensive work here
    return result
                    """
                ))

            elif bottleneck['type'] == 'slow_query':
                suggestions.append(Optimization(
                    category=Category.DATABASE,
                    severity=Severity.HIGH,
                    title="Slow database query detected",
                    description=f"Query took {bottleneck.get('duration', 0)}ms",
                    impact="High - Database queries are blocking",
                    effort="Medium",
                    code_example="""
# Add indexes for frequently queried columns
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_order_date ON orders(created_at);

# Use query result caching
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=128)
def get_user_by_email(email, cache_time=None):
    # Cache key includes timestamp for expiration
    return db.query(User).filter(User.email == email).first()

# Batch queries to avoid N+1 problem
# Before: N+1 queries
users = User.query.all()
for user in users:
    posts = user.posts  # Separate query for each user

# After: Single query with join
users = User.query.options(joinedload(User.posts)).all()
                    """
                ))

        return suggestions

    def _get_cpu_optimization_example(self, func: Dict[str, Any]) -> str:
        """Generate CPU optimization code example"""
        return f"""
# Optimize {func['function']}

# 1. Profile to identify bottlenecks
# python -m cProfile -o profile.stats your_script.py

# 2. Use Numba for numerical computations
from numba import jit

@jit(nopython=True)
def {func['function']}(data):
    # Your computation here
    # Numba will compile to native code
    return result

# 3. Vectorize operations with NumPy
import numpy as np

# Before: Python loop
result = []
for x in data:
    result.append(x * 2 + 1)

# After: Vectorized
result = np.array(data) * 2 + 1

# 4. Use multiprocessing for parallel execution
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map({func['function']}, data_chunks)
        """

    def generate_suggestions(self, profile_data: Dict[str, Any]) -> List[Optimization]:
        """Generate all optimization suggestions"""
        all_suggestions = []

        if 'cpu' in profile_data:
            all_suggestions.extend(self.analyze_cpu_profile(profile_data['cpu']))

        if 'memory' in profile_data:
            all_suggestions.extend(self.analyze_memory_profile(profile_data['memory']))

        if 'io' in profile_data:
            all_suggestions.extend(self.analyze_io_profile(profile_data['io']))

        if 'bottlenecks' in profile_data:
            all_suggestions.extend(self.analyze_bottlenecks(profile_data))

        # Sort by severity
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3
        }
        all_suggestions.sort(key=lambda x: severity_order[x.severity])

        self.optimizations = all_suggestions
        return all_suggestions

    def generate_report(self, output_file: str = None):
        """Generate optimization report"""
        report = {
            "generated": Path(__file__).stem,
            "timestamp": "now",
            "total_suggestions": len(self.optimizations),
            "by_severity": {
                "critical": len([o for o in self.optimizations if o.severity == Severity.CRITICAL]),
                "high": len([o for o in self.optimizations if o.severity == Severity.HIGH]),
                "medium": len([o for o in self.optimizations if o.severity == Severity.MEDIUM]),
                "low": len([o for o in self.optimizations if o.severity == Severity.LOW])
            },
            "suggestions": [opt.to_dict() for opt in self.optimizations]
        }

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Optimization report saved to: {output_file}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Generate optimization suggestions")
    parser.add_argument("profile_file", help="Path to profiling data JSON")
    parser.add_argument("--output", "-o", help="Output file for suggestions")
    parser.add_argument("--format", choices=["json", "text"], default="json")

    args = parser.parse_args()

    # Load profile data
    with open(args.profile_file) as f:
        profile_data = json.load(f)

    # Generate suggestions
    suggester = OptimizationSuggester()
    suggestions = suggester.generate_suggestions(profile_data)

    print(f"\nGenerated {len(suggestions)} optimization suggestions:")
    print(f"  Critical: {len([s for s in suggestions if s.severity == Severity.CRITICAL])}")
    print(f"  High: {len([s for s in suggestions if s.severity == Severity.HIGH])}")
    print(f"  Medium: {len([s for s in suggestions if s.severity == Severity.MEDIUM])}")
    print(f"  Low: {len([s for s in suggestions if s.severity == Severity.LOW])}")

    # Generate report
    output_file = args.output or "optimization_suggestions.json"
    suggester.generate_report(output_file)


if __name__ == "__main__":
    main()
