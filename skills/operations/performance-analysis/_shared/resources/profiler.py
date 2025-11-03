#!/usr/bin/env python3
"""
Performance Profiler - CPU, Memory, and I/O profiling with detailed metrics

Supports multiple profiling modes:
- CPU profiling with cProfile and line_profiler
- Memory profiling with memory_profiler and tracemalloc
- I/O profiling with system call tracing
- Combined profiling for comprehensive analysis
"""

import os
import sys
import time
import psutil
import argparse
import json
import cProfile
import pstats
import io
import tracemalloc
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class PerformanceProfiler:
    """Comprehensive performance profiler for CPU, memory, and I/O"""

    def __init__(self, output_dir: str = "profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {},
            "memory": {},
            "io": {},
            "processes": []
        }

    def profile_cpu(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Profile CPU usage with cProfile"""
        profiler = cProfile.Profile()
        start_time = time.time()

        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        duration = time.time() - start_time

        # Generate statistics
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.strip_dirs()
        stats.sort_stats('cumulative')

        # Extract top functions
        top_functions = []
        for func_info, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:20]:
            filename, line, func_name = func_info
            top_functions.append({
                "function": func_name,
                "file": filename,
                "line": line,
                "calls": nc,
                "total_time": tt,
                "cumulative_time": ct,
                "time_per_call": tt/nc if nc > 0 else 0
            })

        cpu_metrics = {
            "duration": duration,
            "top_functions": top_functions,
            "total_calls": stats.total_calls,
            "profile_file": str(self.output_dir / "cpu_profile.stats")
        }

        # Save profile
        stats.dump_stats(cpu_metrics["profile_file"])

        self.metrics["cpu"] = cpu_metrics
        return cpu_metrics

    def profile_memory(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage with tracemalloc"""
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()
        start_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        end_snapshot = tracemalloc.take_snapshot()
        end_mem = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        tracemalloc.stop()

        # Analyze memory differences
        top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')

        memory_hotspots = []
        for stat in top_stats[:20]:
            memory_hotspots.append({
                "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size_diff_mb": stat.size_diff / 1024 / 1024,
                "count_diff": stat.count_diff,
                "current_size_mb": stat.size / 1024 / 1024
            })

        memory_metrics = {
            "start_mb": start_mem,
            "end_mb": end_mem,
            "delta_mb": end_mem - start_mem,
            "peak_mb": tracemalloc.get_traced_memory()[1] / 1024 / 1024,
            "hotspots": memory_hotspots
        }

        self.metrics["memory"] = memory_metrics
        return memory_metrics

    def profile_io(self, duration: int = 60) -> Dict[str, Any]:
        """Profile I/O operations over a time period"""
        process = psutil.Process()

        # Initial I/O counters
        io_start = process.io_counters()
        disk_start = psutil.disk_io_counters()
        net_start = psutil.net_io_counters()

        time.sleep(duration)

        # Final I/O counters
        io_end = process.io_counters()
        disk_end = psutil.disk_io_counters()
        net_end = psutil.net_io_counters()

        io_metrics = {
            "process": {
                "read_bytes": io_end.read_bytes - io_start.read_bytes,
                "write_bytes": io_end.write_bytes - io_start.write_bytes,
                "read_count": io_end.read_count - io_start.read_count,
                "write_count": io_end.write_count - io_start.write_count
            },
            "disk": {
                "read_mb": (disk_end.read_bytes - disk_start.read_bytes) / 1024 / 1024,
                "write_mb": (disk_end.write_bytes - disk_start.write_bytes) / 1024 / 1024,
                "read_time_ms": disk_end.read_time - disk_start.read_time,
                "write_time_ms": disk_end.write_time - disk_start.write_time
            },
            "network": {
                "sent_mb": (net_end.bytes_sent - net_start.bytes_sent) / 1024 / 1024,
                "recv_mb": (net_end.bytes_recv - net_start.bytes_recv) / 1024 / 1024,
                "packets_sent": net_end.packets_sent - net_start.packets_sent,
                "packets_recv": net_end.packets_recv - net_start.packets_recv
            }
        }

        self.metrics["io"] = io_metrics
        return io_metrics

    def monitor_system(self, interval: float = 1.0, samples: int = 60) -> Dict[str, Any]:
        """Monitor system resources over time"""
        cpu_samples = []
        memory_samples = []
        disk_samples = []

        for _ in range(samples):
            cpu_samples.append(psutil.cpu_percent(interval=interval))
            mem = psutil.virtual_memory()
            memory_samples.append({
                "percent": mem.percent,
                "available_mb": mem.available / 1024 / 1024,
                "used_mb": mem.used / 1024 / 1024
            })
            disk = psutil.disk_usage('/')
            disk_samples.append({
                "percent": disk.percent,
                "free_gb": disk.free / 1024 / 1024 / 1024
            })

        system_metrics = {
            "cpu": {
                "avg": sum(cpu_samples) / len(cpu_samples),
                "max": max(cpu_samples),
                "min": min(cpu_samples),
                "samples": cpu_samples
            },
            "memory": {
                "avg_percent": sum(s["percent"] for s in memory_samples) / len(memory_samples),
                "max_used_mb": max(s["used_mb"] for s in memory_samples),
                "samples": memory_samples
            },
            "disk": {
                "avg_percent": sum(s["percent"] for s in disk_samples) / len(disk_samples),
                "min_free_gb": min(s["free_gb"] for s in disk_samples),
                "samples": disk_samples
            }
        }

        return system_metrics

    def profile_processes(self) -> List[Dict[str, Any]]:
        """Profile all running processes"""
        processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                info = proc.info
                if info['cpu_percent'] > 1.0 or info['memory_percent'] > 1.0:
                    processes.append({
                        "pid": info['pid'],
                        "name": info['name'],
                        "cpu_percent": info['cpu_percent'],
                        "memory_percent": info['memory_percent'],
                        "status": info['status']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Sort by CPU usage
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)

        self.metrics["processes"] = processes[:20]  # Top 20
        return processes

    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        report_file = self.output_dir / f"performance_report_{int(time.time())}.json"

        with open(report_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        return str(report_file)

    @contextmanager
    def profile_context(self, mode: str = "all"):
        """Context manager for profiling code blocks"""
        start_time = time.time()
        start_mem = psutil.Process().memory_info().rss / 1024 / 1024

        if mode in ("all", "memory"):
            tracemalloc.start()

        yield self

        if mode in ("all", "memory"):
            tracemalloc.stop()

        duration = time.time() - start_time
        end_mem = psutil.Process().memory_info().rss / 1024 / 1024

        print(f"\nProfile Summary ({mode}):")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Memory Delta: {end_mem - start_mem:.2f} MB")
        print(f"  Final Memory: {end_mem:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Performance Profiler")
    parser.add_argument("--mode", choices=["cpu", "memory", "io", "system", "all"],
                       default="all", help="Profiling mode")
    parser.add_argument("--duration", type=int, default=60,
                       help="Duration for monitoring (seconds)")
    parser.add_argument("--output", type=str, default="profiles",
                       help="Output directory for profiles")
    parser.add_argument("--target", type=str, help="Python script to profile")

    args = parser.parse_args()

    profiler = PerformanceProfiler(output_dir=args.output)

    if args.mode == "system":
        print(f"Monitoring system for {args.duration} seconds...")
        metrics = profiler.monitor_system(samples=args.duration)
        print(f"\nCPU Average: {metrics['cpu']['avg']:.1f}%")
        print(f"Memory Average: {metrics['memory']['avg_percent']:.1f}%")
        print(f"Disk Usage: {metrics['disk']['avg_percent']:.1f}%")

    elif args.mode == "io":
        print(f"Profiling I/O for {args.duration} seconds...")
        metrics = profiler.profile_io(duration=args.duration)
        print(f"\nDisk Read: {metrics['disk']['read_mb']:.2f} MB")
        print(f"Disk Write: {metrics['disk']['write_mb']:.2f} MB")
        print(f"Network Sent: {metrics['network']['sent_mb']:.2f} MB")
        print(f"Network Recv: {metrics['network']['recv_mb']:.2f} MB")

    elif args.target:
        print(f"Profiling {args.target}...")
        # Execute target script with profiling
        with open(args.target) as f:
            code = compile(f.read(), args.target, 'exec')

            if args.mode in ("cpu", "all"):
                profiler.profile_cpu(exec, code)
                print("\nCPU profiling complete")

            if args.mode in ("memory", "all"):
                profiler.profile_memory(exec, code)
                print("\nMemory profiling complete")

    # Always profile processes
    processes = profiler.profile_processes()
    print(f"\nTop 5 CPU-intensive processes:")
    for p in processes[:5]:
        print(f"  {p['name']:20s} CPU: {p['cpu_percent']:5.1f}% MEM: {p['memory_percent']:5.1f}%")

    report_file = profiler.generate_report()
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
