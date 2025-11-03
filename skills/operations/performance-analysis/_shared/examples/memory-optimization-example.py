#!/usr/bin/env python3
"""
Memory Optimization Example - Comprehensive memory profiling and optimization

This example demonstrates:
1. Memory profiling with tracemalloc
2. Detecting memory leaks
3. Optimizing memory allocations
4. Using generators and streaming
5. Memory-efficient data structures

Run: python memory-optimization-example.py
"""

import sys
import gc
import tracemalloc
from pathlib import Path
from typing import List, Iterator, Dict
import time

# Add resources to path
sys.path.insert(0, str(Path(__file__).parent.parent / "resources"))

from profiler import PerformanceProfiler


class MemoryOptimizationDemo:
    """Demonstrates memory profiling and optimization techniques"""

    def __init__(self):
        self.profiler = PerformanceProfiler(output_dir="./memory-optimization-results")

    # INEFFICIENT IMPLEMENTATIONS (High Memory Usage)

    def load_large_file_inefficient(self, filename: str, size_mb: int = 100) -> List[str]:
        """Load entire file into memory at once - inefficient for large files"""
        # Create a temporary large file
        lines = []
        for i in range(size_mb * 10000):  # Approximate lines for size_mb
            lines.append(f"Line {i}: {'x' * 50}\n")

        return lines

    def process_data_inefficient(self, size: int = 1000000) -> List[Dict]:
        """Create large list of dictionaries - high memory usage"""
        results = []

        for i in range(size):
            # Full dictionary for each item (wastes memory)
            results.append({
                'id': i,
                'value': i * 2,
                'squared': i ** 2,
                'description': f"Item number {i} with various properties",
                'metadata': {
                    'created': 'now',
                    'updated': 'now',
                    'tags': ['tag1', 'tag2', 'tag3']
                }
            })

        return results

    def duplicate_strings_inefficient(self, count: int = 100000) -> List[str]:
        """Create many duplicate string objects"""
        strings = []

        for i in range(count):
            # Creates new string object each time
            strings.append("duplicate_string_" + str(i % 100))

        return strings

    def circular_reference_leak(self):
        """Creates circular references that can cause memory leaks"""
        class Node:
            def __init__(self, value):
                self.value = value
                self.next = None
                self.prev = None

        # Create circular linked list
        nodes = []
        for i in range(10000):
            node = Node(i)
            if nodes:
                node.prev = nodes[-1]
                nodes[-1].next = node
            nodes.append(node)

        # Create circular reference
        if nodes:
            nodes[0].prev = nodes[-1]
            nodes[-1].next = nodes[0]

        # Nodes will not be garbage collected until references are broken
        return nodes

    # OPTIMIZED IMPLEMENTATIONS (Low Memory Usage)

    def load_large_file_optimized(self, filename: str, size_mb: int = 100) -> Iterator[str]:
        """Use generator to stream file line by line - memory efficient"""
        for i in range(size_mb * 10000):
            yield f"Line {i}: {'x' * 50}\n"

    def process_data_optimized(self, size: int = 1000000) -> Iterator[Dict]:
        """Use generator to yield data on demand"""
        for i in range(size):
            # Yield one item at a time
            yield {
                'id': i,
                'value': i * 2,
                'squared': i ** 2
            }

    def use_slots_optimized(self, count: int = 100000):
        """Use __slots__ to reduce memory per instance"""
        class DataPoint:
            __slots__ = ['id', 'value', 'timestamp']

            def __init__(self, id, value, timestamp):
                self.id = id
                self.value = value
                self.timestamp = timestamp

        return [DataPoint(i, i * 2, time.time()) for i in range(count)]

    def use_string_interning_optimized(self, count: int = 100000) -> List[str]:
        """Use string interning to share duplicate strings"""
        # Pre-create unique strings
        unique_strings = [sys.intern(f"duplicate_string_{i}") for i in range(100)]

        # Reuse interned strings
        strings = []
        for i in range(count):
            strings.append(unique_strings[i % 100])

        return strings

    # MEMORY LEAK DEMONSTRATION

    def demonstrate_memory_leak(self, iterations: int = 100):
        """Demonstrate and fix memory leak"""
        print(f"\n{'='*60}")
        print("MEMORY LEAK DEMONSTRATION")
        print(f"{'='*60}")

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        print("\nCreating circular references (potential memory leak)...")
        leaked_objects = []

        for i in range(iterations):
            # Create objects with circular references
            leaked_objects.append(self.circular_reference_leak())

        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

        print("\nTop memory allocations:")
        for stat in top_stats[:5]:
            print(f"  {stat}")

        memory_before = sum(stat.size for stat in snapshot_before.statistics('lineno'))
        memory_after = sum(stat.size for stat in snapshot_after.statistics('lineno'))
        leak_mb = (memory_after - memory_before) / 1024 / 1024

        print(f"\nMemory growth: {leak_mb:.2f} MB")

        # Fix: Break circular references
        print("\nBreaking circular references...")
        for nodes in leaked_objects:
            for node in nodes:
                node.next = None
                node.prev = None

        leaked_objects.clear()
        gc.collect()

        snapshot_cleaned = tracemalloc.take_snapshot()
        memory_cleaned = sum(stat.size for stat in snapshot_cleaned.statistics('lineno'))
        recovered_mb = (memory_after - memory_cleaned) / 1024 / 1024

        print(f"Memory recovered: {recovered_mb:.2f} MB")

        tracemalloc.stop()

    # BENCHMARKING METHODS

    def benchmark_file_loading(self, size_mb: int = 50):
        """Benchmark file loading strategies"""
        print(f"\n{'='*60}")
        print(f"Benchmarking File Loading ({size_mb}MB)")
        print(f"{'='*60}")

        # Measure inefficient version
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        print("\n1. Loading all data into memory (Inefficient)...")
        start = time.time()
        all_lines = self.load_large_file_inefficient("dummy.txt", size_mb)
        time_inefficient = time.time() - start

        snapshot_after = tracemalloc.take_snapshot()
        memory_inefficient = (snapshot_after.compare_to(snapshot_before, 'lineno')[0].size_diff
                            if snapshot_after.compare_to(snapshot_before, 'lineno') else 0) / 1024 / 1024

        print(f"   Time: {time_inefficient:.4f}s")
        print(f"   Memory: {memory_inefficient:.2f} MB")

        del all_lines
        gc.collect()

        # Measure optimized version
        snapshot_before = tracemalloc.take_snapshot()

        print("\n2. Streaming with generator (Optimized)...")
        start = time.time()
        line_count = 0
        for line in self.load_large_file_optimized("dummy.txt", size_mb):
            line_count += 1
        time_optimized = time.time() - start

        snapshot_after = tracemalloc.take_snapshot()
        memory_optimized = (snapshot_after.compare_to(snapshot_before, 'lineno')[0].size_diff
                          if snapshot_after.compare_to(snapshot_before, 'lineno') else 0) / 1024 / 1024

        print(f"   Time: {time_optimized:.4f}s")
        print(f"   Memory: {memory_optimized:.2f} MB")
        print(f"   Lines processed: {line_count}")

        tracemalloc.stop()

        # Calculate improvement
        memory_saved = memory_inefficient - memory_optimized
        memory_reduction = (memory_saved / memory_inefficient * 100) if memory_inefficient > 0 else 0

        print(f"\n3. Results:")
        print(f"   Memory Saved: {memory_saved:.2f} MB")
        print(f"   Memory Reduction: {memory_reduction:.1f}%")

        return {
            'inefficient_memory_mb': memory_inefficient,
            'optimized_memory_mb': memory_optimized,
            'memory_saved_mb': memory_saved,
            'reduction_percent': memory_reduction
        }

    def benchmark_data_structures(self, count: int = 100000):
        """Benchmark memory usage of different data structures"""
        print(f"\n{'='*60}")
        print(f"Benchmarking Data Structures (n={count})")
        print(f"{'='*60}")

        tracemalloc.start()

        # Regular class without __slots__
        print("\n1. Regular class (no __slots__)...")
        snapshot_before = tracemalloc.take_snapshot()

        class RegularClass:
            def __init__(self, id, value):
                self.id = id
                self.value = value

        regular_objects = [RegularClass(i, i * 2) for i in range(count)]

        snapshot_after = tracemalloc.take_snapshot()
        memory_regular = (snapshot_after.compare_to(snapshot_before, 'lineno')[0].size_diff
                        if snapshot_after.compare_to(snapshot_before, 'lineno') else 0) / 1024 / 1024

        print(f"   Memory: {memory_regular:.2f} MB")

        del regular_objects
        gc.collect()

        # Class with __slots__
        print("\n2. Class with __slots__ (Optimized)...")
        snapshot_before = tracemalloc.take_snapshot()

        optimized_objects = self.use_slots_optimized(count)

        snapshot_after = tracemalloc.take_snapshot()
        memory_slots = (snapshot_after.compare_to(snapshot_before, 'lineno')[0].size_diff
                      if snapshot_after.compare_to(snapshot_before, 'lineno') else 0) / 1024 / 1024

        print(f"   Memory: {memory_slots:.2f} MB")

        tracemalloc.stop()

        # Calculate improvement
        memory_saved = memory_regular - memory_slots
        memory_reduction = (memory_saved / memory_regular * 100) if memory_regular > 0 else 0

        print(f"\n3. Results:")
        print(f"   Memory Saved: {memory_saved:.2f} MB")
        print(f"   Memory Reduction: {memory_reduction:.1f}%")
        print(f"   Per-object savings: {(memory_saved / count * 1024):.2f} KB")

        return {
            'regular_memory_mb': memory_regular,
            'optimized_memory_mb': memory_slots,
            'memory_saved_mb': memory_saved,
            'reduction_percent': memory_reduction
        }

    def run_comprehensive_analysis(self):
        """Run comprehensive memory optimization analysis"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MEMORY OPTIMIZATION DEMONSTRATION")
        print("="*60)

        results = {}

        # File loading benchmark
        results['file_loading'] = self.benchmark_file_loading(size_mb=50)

        # Data structure benchmark
        results['data_structures'] = self.benchmark_data_structures(count=100000)

        # Memory leak demonstration
        self.demonstrate_memory_leak(iterations=10)

        # Generate summary
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)

        total_saved = 0
        for category, metrics in results.items():
            saved = metrics['memory_saved_mb']
            total_saved += saved

            print(f"\n{category.upper().replace('_', ' ')}:")
            print(f"  Memory Saved: {saved:.2f} MB")
            print(f"  Reduction: {metrics['reduction_percent']:.1f}%")

        print(f"\nTOTAL MEMORY SAVED: {total_saved:.2f} MB")

        return results


def main():
    """Main execution function"""
    print("Starting Memory Optimization Example...")

    demo = MemoryOptimizationDemo()
    results = demo.run_comprehensive_analysis()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. MEMORY PROFILING:
   - Use tracemalloc to track memory allocations
   - Take snapshots before/after operations
   - Analyze memory growth patterns
   - Identify memory hotspots

2. OPTIMIZATION STRATEGIES:
   - Use generators instead of lists for large datasets
   - Implement streaming for file processing
   - Use __slots__ to reduce per-instance memory
   - Intern duplicate strings
   - Break circular references

3. MEMORY LEAKS:
   - Circular references prevent garbage collection
   - Always break references when done
   - Use weak references where appropriate
   - Monitor memory growth over time

4. BEST PRACTICES:
   - Profile memory usage regularly
   - Choose appropriate data structures
   - Stream data when possible
   - Use context managers for cleanup
   - Implement proper __del__ methods

5. TOOLS:
   - tracemalloc for memory profiling
   - gc module for garbage collection
   - sys.getsizeof() for object sizes
   - Memory profilers (memory_profiler, pympler)
    """)

    print("\nMemory Optimization Example Complete!")


if __name__ == "__main__":
    main()
