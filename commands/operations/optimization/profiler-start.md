---
name: profiler-start
category: optimization
version: 1.0.0
---

# /profiler-start

Start performance profiler to analyze CPU, memory, and execution bottlenecks.

## Usage
```bash
/profiler-start [profile_type] [options]
```

## Parameters
- `profile_type` - Profile type: cpu|memory|heap|alloc|block|all (default: cpu)
- `--duration` - Profiling duration in seconds (default: 30)
- `--service` - Service to profile (default: current)
- `--interval` - Sampling interval in ms (default: 10)
- `--output` - Output file path (default: auto-generated)
- `--format` - Output format: pprof|flamegraph|json (default: flamegraph)
- `--auto-analyze` - Auto-analyze on stop (default: true)

## What It Does

**Performance Profiling**:
1. 🔥 **CPU Profiling**: Sample CPU usage and call stacks
2. 💾 **Memory Profiling**: Track allocations and heap usage
3. 🧱 **Heap Profiling**: Analyze memory layout
4. 🔒 **Block Profiling**: Detect lock contention
5. 🎯 **Hotspot Detection**: Find performance bottlenecks
6. 📊 **Flame Graphs**: Visualize execution patterns
7. 🔍 **Call Tree Analysis**: Function call hierarchy
8. 📈 **Performance Metrics**: Execution time per function

## Examples

```bash
# Start CPU profiler for 30 seconds
/profiler-start cpu

# Memory profiler with custom duration
/profiler-start memory --duration 60

# All profile types
/profiler-start all --duration 120

# Profile specific service
/profiler-start cpu --service api-server --duration 45

# Output as pprof format
/profiler-start cpu --format pprof --output profile.pprof

# Short interval for detailed sampling
/profiler-start cpu --interval 5 --duration 60
```

## Output

```
🔥 Performance Profiler Started

Profile Type: cpu
Service: api-server
Duration: 30 seconds
Sampling Interval: 10ms
Output Format: flamegraph

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Profiling Session
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Started: 2025-11-01 12:35:00.000 UTC
  Service: api-server-abc123 (pid: 45678)
  Profile Type: CPU
  Sampling: Every 10ms

  [00:05] Samples collected: 500 (10% complete)
          CPU: 45% avg
          Top function: processOrder (12.3%)

  [00:10] Samples collected: 1,000 (33% complete)
          CPU: 52% avg
          Top function: processOrder (14.1%)

  [00:15] Samples collected: 1,500 (50% complete)
          CPU: 48% avg
          Top function: validatePayment (11.7%)

  [00:20] Samples collected: 2,000 (67% complete)
          CPU: 51% avg
          Top function: processOrder (13.5%)

  [00:25] Samples collected: 2,500 (83% complete)
          CPU: 46% avg
          Top function: processOrder (12.9%)

  [00:30] Samples collected: 3,000 (100% complete)
          CPU: 49% avg
          Profiling complete

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Profile Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Samples: 3,000
  Duration: 30.0s
  Average CPU: 49%

  Top Functions (by CPU time):
    1. processOrder: 13.2% (396ms)
    2. validatePayment: 11.8% (354ms)
    3. calculateTax: 8.9% (267ms)
    4. serializeResponse: 7.4% (222ms)
    5. parseRequest: 6.1% (183ms)
    6. queryDatabase: 5.7% (171ms)
    7. validateAuth: 4.3% (129ms)
    8. logRequest: 3.2% (96ms)
    9. formatJSON: 2.8% (84ms)
    10. middleware: 2.1% (63ms)

  Hot Paths (call stacks consuming >5%):
    1. HTTP Server → processOrder → validatePayment (11.8%)
    2. HTTP Server → processOrder → calculateTax (8.9%)
    3. HTTP Server → processOrder → serializeResponse (7.4%)
    4. HTTP Server → parseRequest → JSON.parse (6.1%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Flame Graph Generated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                          [HTTP Server] (100%)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
  [processOrder]            [parseRequest]            [middleware]
      (35.2%)                   (6.1%)                   (2.1%)
        │
  ┌─────┼──────┬───────┐
  │     │      │       │
[validate] [calc] [serialize] [log]
Payment   Tax   Response    Request
(11.8%)  (8.9%)  (7.4%)     (3.2%)

Saved: reports/flamegraph-cpu-2025-11-01-123500.svg
View: open reports/flamegraph-cpu-2025-11-01-123500.svg

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Performance Hotspots
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔥 Critical Hotspots:
    1. processOrder (13.2% CPU)
       Location: services/order.service.ts:156
       Calls: 2,345
       Avg time: 5.6ms
       Issue: Complex nested logic
       Recommendation: Refactor into smaller functions

    2. validatePayment (11.8% CPU)
       Location: services/payment.service.ts:89
       Calls: 2,345
       Avg time: 5.0ms
       Issue: Synchronous validation loops
       Recommendation: Use parallel validation

    3. calculateTax (8.9% CPU)
       Location: services/tax.service.ts:234
       Calls: 2,345
       Avg time: 3.8ms
       Issue: Repeated regex operations
       Recommendation: Cache tax rules

  ⚠️  Medium Hotspots:
    4. serializeResponse (7.4% CPU)
       Location: utils/serializer.ts:45
       Issue: Large object serialization
       Recommendation: Use streaming serialization

    5. parseRequest (6.1% CPU)
       Location: middleware/parser.ts:67
       Issue: JSON parsing overhead
       Recommendation: Validate before parse, use schema

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Optimization Opportunities
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Identified Issues:
    1. Inefficient Algorithm
       Function: calculateTax
       Complexity: O(n²) nested loops
       Fix: Use hash map (O(n))
       Expected gain: 60% reduction

    2. Missing Memoization
       Function: validatePayment
       Issue: Repeated calculations
       Fix: Cache validation results
       Expected gain: 40% reduction

    3. Synchronous I/O
       Function: queryDatabase (called in loop)
       Issue: Sequential queries
       Fix: Batch queries or use Promise.all
       Expected gain: 70% reduction

    4. String Concatenation
       Function: serializeResponse
       Issue: Using += in loop
       Fix: Use array join or template literals
       Expected gain: 30% reduction

  Estimated Total Gain: 35-45% CPU reduction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Profiler Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ COMPLETE
Profile Type: CPU
Duration: 30.0s
Samples: 3,000
Average CPU: 49%

Top Bottlenecks:
  1. processOrder: 13.2%
  2. validatePayment: 11.8%
  3. calculateTax: 8.9%

Optimization Potential: 35-45% CPU reduction

Outputs:
  📊 Flame Graph: reports/flamegraph-cpu-2025-11-01-123500.svg
  📈 pprof: reports/profile-cpu-2025-11-01-123500.pprof
  📝 JSON: reports/profile-cpu-2025-11-01-123500.json
  📄 Report: reports/profile-analysis-2025-11-01-123500.md

Next Steps:
  1. Review flame graph for visual analysis
  2. Implement optimizations for top 3 hotspots
  3. Run /profiler-stop to end profiling session
  4. Re-profile after optimizations to validate

Commands:
  View flame graph:
    open reports/flamegraph-cpu-2025-11-01-123500.svg

  Analyze with pprof:
    go tool pprof reports/profile-cpu-2025-11-01-123500.pprof

  Stop profiler:
    /profiler-stop

✅ Profiling Session Active!
Use /profiler-stop to end profiling and generate final report.
```

## Chains With

```bash
# Profile → analyze → optimize
/profiler-start cpu --duration 60 && /profiler-stop

# Load test → profile
/load-test baseline && /profiler-start cpu --duration 120

# Profile → detect bottlenecks
/profiler-start all && /bottleneck-detect

# Profile → optimize → re-profile
/profiler-start cpu && /optimize-code && /profiler-start cpu
```

## See Also
- `/profiler-stop` - Stop profiler and analyze results
- `/bottleneck-detect` - Bottleneck detection
- `/performance-report` - Performance analysis
- `/load-test` - Load testing
- `/monitoring-configure` - Setup continuous profiling
