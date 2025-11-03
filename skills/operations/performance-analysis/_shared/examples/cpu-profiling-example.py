#!/usr/bin/env python3
"""
CPU Profiling Example - Comprehensive demonstration of CPU profiling and optimization

This example demonstrates:
1. Identifying CPU bottlenecks with profiling
2. Analyzing hot paths and expensive functions
3. Applying algorithmic optimizations
4. Measuring performance improvements
5. Generating optimization reports

Run: python cpu-profiling-example.py
"""

import sys
import time
import cProfile
import pstats
import io
from pathlib import Path
from typing import List, Dict

# Add resources to path
sys.path.insert(0, str(Path(__file__).parent.parent / "resources"))

from profiler import PerformanceProfiler
from optimization_suggester import OptimizationSuggester


class CPUProfilingDemo:
    """Demonstrates CPU profiling and optimization"""

    def __init__(self):
        self.profiler = PerformanceProfiler(output_dir="./cpu-profiling-results")
        self.suggester = OptimizationSuggester()

    # INEFFICIENT IMPLEMENTATIONS (Before Optimization)

    def bubble_sort_inefficient(self, arr: List[int]) -> List[int]:
        """O(n²) bubble sort - inefficient for large arrays"""
        n = len(arr)
        arr = arr.copy()

        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]

        return arr

    def fibonacci_inefficient(self, n: int) -> int:
        """Recursive Fibonacci - exponential time complexity O(2^n)"""
        if n <= 1:
            return n
        return self.fibonacci_inefficient(n - 1) + self.fibonacci_inefficient(n - 2)

    def search_inefficient(self, arr: List[int], target: int) -> int:
        """Linear search - O(n) when binary search could be O(log n)"""
        for i, val in enumerate(arr):
            if val == target:
                return i
        return -1

    def duplicate_work_inefficient(self, data: List[str]) -> Dict[str, int]:
        """Redundant computation - same work done multiple times"""
        result = {}

        for item in data:
            # Inefficient: Computing length multiple times
            if len(item) > 5:
                result[item] = len(item) * 2

            # Inefficient: Unnecessary string operations
            upper = item.upper()
            lower = item.lower()
            if upper == lower:
                result[item] = len(item)

        return result

    # OPTIMIZED IMPLEMENTATIONS (After Optimization)

    def bubble_sort_optimized(self, arr: List[int]) -> List[int]:
        """Quick sort - O(n log n) average case"""
        if len(arr) <= 1:
            return arr

        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]

        return self.bubble_sort_optimized(left) + middle + self.bubble_sort_optimized(right)

    def fibonacci_optimized(self, n: int, memo: Dict[int, int] = None) -> int:
        """Memoized Fibonacci - O(n) with dynamic programming"""
        if memo is None:
            memo = {}

        if n in memo:
            return memo[n]

        if n <= 1:
            return n

        memo[n] = self.fibonacci_optimized(n - 1, memo) + self.fibonacci_optimized(n - 2, memo)
        return memo[n]

    def search_optimized(self, arr: List[int], target: int) -> int:
        """Binary search - O(log n) for sorted arrays"""
        left, right = 0, len(arr) - 1

        while left <= right:
            mid = (left + right) // 2

            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1

    def duplicate_work_optimized(self, data: List[str]) -> Dict[str, int]:
        """Optimized - compute once, reuse results"""
        result = {}

        for item in data:
            # Compute length once
            length = len(item)

            if length > 5:
                result[item] = length * 2
            elif item.isalpha() and item.upper() == item.lower():
                result[item] = length

        return result

    # BENCHMARKING METHODS

    def benchmark_sorting(self, size: int = 1000):
        """Benchmark sorting algorithms"""
        import random

        data = [random.randint(1, 10000) for _ in range(size)]

        print(f"\n{'='*60}")
        print(f"Benchmarking Sorting (n={size})")
        print(f"{'='*60}")

        # Profile inefficient version
        print("\n1. Profiling Bubble Sort (Inefficient)...")
        start = time.time()
        result_inefficient = self.bubble_sort_inefficient(data)
        time_inefficient = time.time() - start
        print(f"   Time: {time_inefficient:.4f}s")

        # Profile optimized version
        print("\n2. Profiling Quick Sort (Optimized)...")
        start = time.time()
        result_optimized = self.bubble_sort_optimized(data)
        time_optimized = time.time() - start
        print(f"   Time: {time_optimized:.4f}s")

        # Calculate improvement
        improvement = ((time_inefficient - time_optimized) / time_inefficient) * 100
        speedup = time_inefficient / time_optimized if time_optimized > 0 else float('inf')

        print(f"\n3. Results:")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Verification: {'PASS' if result_inefficient == result_optimized else 'FAIL'}")

        return {
            'inefficient_time': time_inefficient,
            'optimized_time': time_optimized,
            'improvement_percent': improvement,
            'speedup': speedup
        }

    def benchmark_fibonacci(self, n: int = 30):
        """Benchmark Fibonacci implementations"""
        print(f"\n{'='*60}")
        print(f"Benchmarking Fibonacci (n={n})")
        print(f"{'='*60}")

        # Profile inefficient version (smaller n to avoid timeout)
        print("\n1. Profiling Recursive Fibonacci (Inefficient)...")
        start = time.time()
        result_inefficient = self.fibonacci_inefficient(min(n, 35))
        time_inefficient = time.time() - start
        print(f"   Time: {time_inefficient:.4f}s")
        print(f"   Result: {result_inefficient}")

        # Profile optimized version
        print("\n2. Profiling Memoized Fibonacci (Optimized)...")
        start = time.time()
        result_optimized = self.fibonacci_optimized(n)
        time_optimized = time.time() - start
        print(f"   Time: {time_optimized:.4f}s")
        print(f"   Result: {result_optimized}")

        # Calculate improvement
        improvement = ((time_inefficient - time_optimized) / time_inefficient) * 100
        speedup = time_inefficient / time_optimized if time_optimized > 0 else float('inf')

        print(f"\n3. Results:")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   Speedup: {speedup:.2f}x")

        return {
            'inefficient_time': time_inefficient,
            'optimized_time': time_optimized,
            'improvement_percent': improvement,
            'speedup': speedup
        }

    def benchmark_search(self, size: int = 10000, searches: int = 1000):
        """Benchmark search algorithms"""
        import random

        # Create sorted array
        data = sorted([random.randint(1, 100000) for _ in range(size)])
        targets = [random.choice(data) for _ in range(searches)]

        print(f"\n{'='*60}")
        print(f"Benchmarking Search (n={size}, searches={searches})")
        print(f"{'='*60}")

        # Profile inefficient version
        print("\n1. Profiling Linear Search (Inefficient)...")
        start = time.time()
        for target in targets:
            self.search_inefficient(data, target)
        time_inefficient = time.time() - start
        print(f"   Time: {time_inefficient:.4f}s")

        # Profile optimized version
        print("\n2. Profiling Binary Search (Optimized)...")
        start = time.time()
        for target in targets:
            self.search_optimized(data, target)
        time_optimized = time.time() - start
        print(f"   Time: {time_optimized:.4f}s")

        # Calculate improvement
        improvement = ((time_inefficient - time_optimized) / time_inefficient) * 100
        speedup = time_inefficient / time_optimized if time_optimized > 0 else float('inf')

        print(f"\n3. Results:")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   Speedup: {speedup:.2f}x")

        return {
            'inefficient_time': time_inefficient,
            'optimized_time': time_optimized,
            'improvement_percent': improvement,
            'speedup': speedup
        }

    def run_comprehensive_profile(self):
        """Run comprehensive CPU profiling"""
        print("\n" + "="*60)
        print("COMPREHENSIVE CPU PROFILING DEMONSTRATION")
        print("="*60)

        # Run all benchmarks
        results = {
            'sorting': self.benchmark_sorting(size=1000),
            'fibonacci': self.benchmark_fibonacci(n=30),
            'search': self.benchmark_search(size=10000, searches=1000)
        }

        # Generate summary report
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)

        total_time_saved = 0
        avg_improvement = 0

        for category, metrics in results.items():
            time_saved = metrics['inefficient_time'] - metrics['optimized_time']
            total_time_saved += time_saved
            avg_improvement += metrics['improvement_percent']

            print(f"\n{category.upper()}:")
            print(f"  Time Saved: {time_saved:.4f}s")
            print(f"  Improvement: {metrics['improvement_percent']:.1f}%")
            print(f"  Speedup: {metrics['speedup']:.2f}x")

        avg_improvement /= len(results)

        print(f"\nOVERALL:")
        print(f"  Total Time Saved: {total_time_saved:.4f}s")
        print(f"  Average Improvement: {avg_improvement:.1f}%")

        return results


def main():
    """Main execution function"""
    print("Starting CPU Profiling Example...")

    demo = CPUProfilingDemo()

    # Run comprehensive profiling
    results = demo.run_comprehensive_profile()

    print("\n" + "="*60)
    print("KEY TAKEAWAYS")
    print("="*60)
    print("""
1. ALGORITHMIC OPTIMIZATION:
   - Bubble Sort (O(n²)) → Quick Sort (O(n log n))
   - Recursive Fibonacci (O(2^n)) → Memoized (O(n))
   - Linear Search (O(n)) → Binary Search (O(log n))

2. PROFILING INSIGHTS:
   - Use cProfile to identify hot paths
   - Focus on functions with highest cumulative time
   - Look for excessive function calls

3. OPTIMIZATION STRATEGIES:
   - Reduce time complexity with better algorithms
   - Eliminate redundant computations
   - Use memoization/caching for repeated calculations
   - Choose appropriate data structures

4. VALIDATION:
   - Always benchmark before and after
   - Verify correctness of optimized code
   - Measure real-world impact

5. BEST PRACTICES:
   - Profile before optimizing (don't guess!)
   - Optimize bottlenecks first (80/20 rule)
   - Maintain code readability
   - Add comprehensive tests
    """)

    print("\nCPU Profiling Example Complete!")
    print(f"Results saved to: {demo.profiler.output_dir}")


if __name__ == "__main__":
    main()
