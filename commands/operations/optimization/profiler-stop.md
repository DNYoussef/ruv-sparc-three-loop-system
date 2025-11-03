---
name: profiler-stop
category: optimization
version: 1.0.0
---

# /profiler-stop

Stop active profiler and generate comprehensive analysis report.

## Usage
```bash
/profiler-stop [options]
```

## Parameters
- `--analyze` - Generate detailed analysis (default: true)
- `--compare` - Compare with baseline profile (optional)
- `--export` - Export formats: svg|pprof|json|html|all (default: all)
- `--recommendations` - Include optimization recommendations (default: true)
- `--diff` - Show diff if comparing profiles (default: true)

## What It Does

**Profile Analysis & Reporting**:
1. 📊 **Data Collection**: Finalize profile data
2. 🔥 **Flame Graph**: Generate flame graph visualization
3. 📈 **Call Tree**: Build call tree analysis
4. 🎯 **Hotspot Analysis**: Identify performance bottlenecks
5. 📉 **Comparison**: Compare with baseline (if provided)
6. 💡 **Recommendations**: AI-powered optimization suggestions
7. 📝 **Report Generation**: Comprehensive analysis report
8. 📤 **Export**: Multiple output formats

## Examples

```bash
# Stop profiler with default analysis
/profiler-stop

# Stop and compare with baseline
/profiler-stop --compare baseline-profile.pprof

# Export specific formats
/profiler-stop --export svg,json

# Stop without recommendations
/profiler-stop --recommendations false

# Quick stop (no analysis)
/profiler-stop --analyze false
```

## Output

```
🛑 Stopping Performance Profiler

Active Session:
  Profile Type: CPU
  Duration: 60.0s
  Samples Collected: 6,000
  Started: 2025-11-01 12:35:00 UTC

Finalizing profiling session...
✅ Profiling stopped
✅ Data collected: 6,000 samples

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Profile Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Samples: 6,000
  Duration: 60.0s
  Average CPU: 52%
  Peak CPU: 78%

  Function Call Distribution:
    Total functions profiled: 1,234
    Unique call stacks: 3,456
    Maximum call depth: 45

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Top Functions by CPU Time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Rank  Function                    CPU%    Self%   Calls   Avg Time
  ────────────────────────────────────────────────────────────────────
  1.    processOrder                15.2%   3.4%    4,567   5.2ms
  2.    validatePayment             12.8%   8.1%    4,567   4.8ms
  3.    calculateTax                10.4%   6.7%    4,567   3.9ms
  4.    serializeResponse           8.7%    5.2%    4,567   3.1ms
  5.    parseRequest                7.1%    4.3%    4,567   2.7ms
  6.    queryDatabase               6.8%    3.9%    2,345   4.1ms
  7.    validateAuth                5.2%    2.8%    4,567   1.9ms
  8.    formatJSON                  4.6%    3.1%    4,567   1.7ms
  9.    logRequest                  3.9%    2.4%    4,567   1.5ms
  10.   middleware/cors             2.8%    1.2%    4,567   1.0ms

  Total profiled: 77.5% (remaining 22.5% in <1% functions)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call Tree Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  HTTP Server (100% / 0.5%)
  ├─ Express Router (98.5% / 1.2%)
  │  ├─ processOrder (15.2% / 3.4%)
  │  │  ├─ validatePayment (12.8% / 8.1%)
  │  │  ├─ calculateTax (10.4% / 6.7%)
  │  │  ├─ queryDatabase (6.8% / 3.9%)
  │  │  └─ serializeResponse (8.7% / 5.2%)
  │  ├─ parseRequest (7.1% / 4.3%)
  │  ├─ validateAuth (5.2% / 2.8%)
  │  └─ middleware/cors (2.8% / 1.2%)
  └─ logRequest (3.9% / 2.4%)

  Self% = time spent in function itself (excluding children)
  CPU% = total time including children

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Baseline Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Comparing with: baseline-profile.pprof
  Baseline date: 2025-10-28

  Performance Changes:
    Function              Current  Baseline  Change
    ──────────────────────────────────────────────────
    processOrder          15.2%    12.1%     +3.1%  ⬆️
    validatePayment       12.8%    10.4%     +2.4%  ⬆️
    calculateTax          10.4%     9.8%     +0.6%  ⬆️
    serializeResponse      8.7%     8.2%     +0.5%  ➡️
    parseRequest           7.1%     7.3%     -0.2%  ⬇️

  Summary:
    ⬆️  Degraded (slower): 3 functions (+6.1% total)
    ➡️  Stable: 1 function
    ⬇️  Improved (faster): 1 function (-0.2% total)

  ⚠️  Performance Regression Detected!
  Overall CPU usage increased by 5.9% compared to baseline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hotspot Deep Dive
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔥 Hotspot #1: validatePayment (12.8% CPU)
     Location: services/payment.service.ts:89-145
     Self Time: 8.1%
     Calls: 4,567
     Avg Duration: 4.8ms

     Code Analysis:
       function validatePayment(order) {
         // Line 89-92: Fast operations (0.3ms)
         const amount = order.total;
         const currency = order.currency;

         // Line 93-110: HOTSPOT (3.2ms, 66.7% of function time)
         for (const item of order.items) {           // ← Loop 1
           for (const rule of validationRules) {     // ← Loop 2 (nested)
             if (rule.test(item)) {                  // ← Regex test (slow)
               applyRule(item, rule);                // ← Function call
             }
           }
         }

         // Line 111-145: Moderate operations (1.3ms)
         return validateCardDetails(order.payment);
       }

     Issue: O(n²) nested loop with regex operations
     Avg items: 5
     Avg rules: 50
     Total iterations: 250 per call × 4,567 calls = 1,141,750 iterations

     Impact: 3.2ms × 4,567 calls = 14.6 seconds total (24.3% of profile)

  🔥 Hotspot #2: calculateTax (10.4% CPU)
     Location: services/tax.service.ts:234-289
     Self Time: 6.7%
     Calls: 4,567

     Issue: Repeated string concatenation in loop
     Impact: 3.9ms per call

  🔥 Hotspot #3: serializeResponse (8.7% CPU)
     Location: utils/serializer.ts:45-98
     Self Time: 5.2%
     Calls: 4,567

     Issue: Large object serialization without streaming
     Impact: 3.1ms per call

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI-Powered Optimization Recommendations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎯 Critical (Implement Immediately):

    1. Optimize validatePayment nested loops
       File: services/payment.service.ts:93-110
       Issue: O(n²) complexity with 1.1M iterations
       Fix:
         ```typescript
         // BEFORE (slow):
         for (const item of order.items) {
           for (const rule of validationRules) {
             if (rule.test(item)) {
               applyRule(item, rule);
             }
           }
         }

         // AFTER (fast):
         const ruleMap = new Map(validationRules.map(r => [r.id, r]));
         for (const item of order.items) {
           const applicableRule = ruleMap.get(item.ruleId);
           if (applicableRule) {
             applyRule(item, applicableRule);
           }
         }
         ```
       Expected Gain: 70-80% reduction (12.8% → 2.6%)
       Estimated Time: 15 minutes

    2. Memoize calculateTax results
       File: services/tax.service.ts:234-289
       Issue: Repeated calculations for same inputs
       Fix:
         ```typescript
         const taxCache = new Map();

         function calculateTax(order) {
           const cacheKey = `${order.region}-${order.total}`;
           if (taxCache.has(cacheKey)) {
             return taxCache.get(cacheKey);
           }
           const result = performTaxCalculation(order);
           taxCache.set(cacheKey, result);
           return result;
         }
         ```
       Expected Gain: 60% reduction (10.4% → 4.2%)
       Cache hit ratio: ~75% (estimated)
       Estimated Time: 10 minutes

  ⚠️  High Priority (Implement Soon):

    3. Use streaming serialization
       File: utils/serializer.ts:45-98
       Issue: Large object serialization blocks event loop
       Fix: Use streaming JSON serializer
       Expected Gain: 40% reduction (8.7% → 5.2%)
       Estimated Time: 30 minutes

    4. Batch database queries
       File: services/order.service.ts:156
       Issue: Sequential queries in processOrder
       Fix: Use Promise.all or batch query
       Expected Gain: 50% reduction (6.8% → 3.4%)
       Estimated Time: 20 minutes

  💡 Medium Priority (Nice to Have):

    5. Compile regex patterns once
       Various locations
       Expected Gain: 5-10% overall
       Estimated Time: 1 hour

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Projected Performance Improvement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Current CPU Usage: 52%

  After Critical Optimizations:
    validatePayment: 12.8% → 2.6% (-10.2%)
    calculateTax: 10.4% → 4.2% (-6.2%)

  After High Priority Optimizations:
    serializeResponse: 8.7% → 5.2% (-3.5%)
    queryDatabase: 6.8% → 3.4% (-3.4%)

  Total Reduction: -23.3%
  Projected CPU Usage: 28.7% (45% improvement!)

  Expected Throughput: +82% (current 4.1K RPS → 7.5K RPS)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exports Generated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ Flame Graph (SVG):
     reports/flamegraph-cpu-2025-11-01-123500.svg
     Interactive visualization of CPU time

  ✅ pprof Format:
     reports/profile-cpu-2025-11-01-123500.pprof
     For use with `go tool pprof`

  ✅ JSON Data:
     reports/profile-cpu-2025-11-01-123500.json
     Raw profile data for custom analysis

  ✅ HTML Report:
     reports/profile-analysis-2025-11-01-123500.html
     Full interactive report with charts

  ✅ Markdown Report:
     reports/profile-summary-2025-11-01-123500.md
     Summary report for documentation

  ✅ Diff Report (vs baseline):
     reports/profile-diff-2025-11-01-123500.html
     Side-by-side comparison

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Profiler Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ COMPLETE
Duration: 60.0s
Samples: 6,000
Average CPU: 52%
Peak CPU: 78%

Top Bottlenecks:
  1. validatePayment: 12.8% (critical)
  2. calculateTax: 10.4% (critical)
  3. serializeResponse: 8.7% (high)

Performance vs Baseline:
  ⚠️  5.9% regression detected
  Critical functions slower by 6.1%

Optimization Potential:
  - Critical fixes: 45% CPU reduction
  - Estimated throughput gain: +82%
  - Implementation time: ~1.5 hours

Next Steps:
  1. Review flame graph: open reports/flamegraph-cpu-2025-11-01-123500.svg
  2. Review HTML report: open reports/profile-analysis-2025-11-01-123500.html
  3. Implement critical optimizations (validatePayment, calculateTax)
  4. Re-run profiler after optimizations
  5. Compare results with /profiler-start --compare

✅ Profiling Analysis Complete!
```

## Chains With

```bash
# Start → stop → analyze
/profiler-start cpu && sleep 60 && /profiler-stop

# Profile → optimize → re-profile
/profiler-start cpu && /profiler-stop && /optimize && /profiler-start cpu

# Compare before/after optimization
/profiler-stop --compare baseline.pprof

# Profile → bottleneck detection
/profiler-stop && /bottleneck-detect
```

## See Also
- `/profiler-start` - Start performance profiler
- `/bottleneck-detect` - Bottleneck detection
- `/performance-report` - Performance analysis
- `/load-test` - Load testing
- `/monitoring-configure` - Continuous profiling setup
