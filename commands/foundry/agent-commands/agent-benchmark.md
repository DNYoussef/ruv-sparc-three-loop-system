---
name: agent-benchmark
description: Benchmark agent performance with comprehensive metrics and analysis
version: 2.0.0
category: agent-management
complexity: medium
tags: [agents, benchmark, performance, metrics, analysis, optimization]
author: ruv-SPARC Agent Team
created: 2025-11-01
last_updated: 2025-11-01
dependencies: [agent-list, agent-metrics, task-orchestrate]
chains_with: [agent-clone, agent-retire, performance-analysis]
evidence_based_techniques: [self-consistency, program-of-thought]
---

# /agent-benchmark - Agent Performance Benchmarking

## Overview

The `/agent-benchmark` command runs comprehensive performance benchmarks on agents to measure speed, accuracy, resource efficiency, and task completion quality.

## Usage

```bash
# Benchmark specific agent
npx claude-flow@alpha agent benchmark --agent-id "coder-123"

# Benchmark all agents of a type
npx claude-flow@alpha agent benchmark --type "coder"

# Run specific benchmark suite
npx claude-flow@alpha agent benchmark \
  --agent-id "coder-123" \
  --suite "comprehensive"

# Quick benchmark
npx claude-flow@alpha agent benchmark \
  --agent-id "tester-456" \
  --suite "quick"

# Benchmark and compare
npx claude-flow@alpha agent benchmark \
  --agents "coder-123,coder-456,coder-789" \
  --compare

# Export results
npx claude-flow@alpha agent benchmark \
  --agent-id "coder-123" \
  --export "benchmark-results.json"
```

## Parameters

- `--agent-id <id>` - Agent to benchmark
- `--agents <ids>` - Multiple agents (comma-separated)
- `--type <type>` - Benchmark all agents of type
- `--suite <suite>` - Benchmark suite: `quick`, `standard`, `comprehensive`
- `--compare` - Compare multiple agents
- `--export <path>` - Export results to file
- `--iterations <n>` - Number of iterations (default: 10)

## Benchmark Metrics

### Performance Metrics
- **Task Completion Rate**: % of successfully completed tasks
- **Average Task Time**: Mean time to complete tasks
- **Error Rate**: % of tasks with errors
- **Success Rate**: % of correct/high-quality outputs

### Resource Metrics
- **Memory Usage**: Average and peak memory consumption
- **CPU Usage**: Average CPU utilization
- **API Calls**: Number of LLM API calls
- **Token Usage**: Total tokens consumed

### Quality Metrics
- **Output Quality**: Quality score (0-100)
- **Code Quality** (for coder agents): Linting, complexity, style
- **Test Coverage** (for tester agents): Coverage percentage
- **Accuracy** (for reviewer agents): Review accuracy

### Efficiency Metrics
- **Tasks per Hour**: Throughput metric
- **Cost Efficiency**: Cost per task completion
- **Resource Efficiency**: Quality per resource unit

## Benchmark Suites

### Quick Suite (~2 minutes)
- 5 simple tasks
- Basic metrics only
- Fast validation

### Standard Suite (~10 minutes)
- 20 varied tasks
- Full metrics
- Performance analysis

### Comprehensive Suite (~30 minutes)
- 50+ complex tasks
- Extended metrics
- Stress testing
- Edge case handling

## Example Output

```bash
npx claude-flow@alpha agent benchmark --agent-id "coder-123"

# Output:
# ════════════════════════════════════════════════════════════
#           AGENT BENCHMARK RESULTS: coder-123
# ════════════════════════════════════════════════════════════
#
# PERFORMANCE METRICS
# ────────────────────────────────────────────────────────────
# Task Completion Rate:    95.0% (19/20 tasks)
# Average Task Time:       42.3 seconds
# Error Rate:              5.0% (1/20 tasks)
# Success Rate:            90.0% (18/20 high quality)
#
# RESOURCE METRICS
# ────────────────────────────────────────────────────────────
# Memory Usage (avg):      124 MB
# Memory Usage (peak):     256 MB
# CPU Usage (avg):         23%
# API Calls:               145
# Token Usage:             234,567 tokens
#
# QUALITY METRICS
# ────────────────────────────────────────────────────────────
# Output Quality:          87/100
# Code Quality:            92/100
#   Linting:               ✓ No errors
#   Complexity:            8.2 avg (threshold: 10)
#   Style:                 98% compliance
#
# EFFICIENCY METRICS
# ────────────────────────────────────────────────────────────
# Tasks per Hour:          85 tasks/hr
# Cost per Task:           $0.23
# Resource Efficiency:     7.2 quality/MB
#
# RANKING
# ────────────────────────────────────────────────────────────
# Overall Score:           88/100
# Rank:                    2nd of 8 coder agents
# Top in:                  Code Quality, Efficiency
# Needs Improvement:       Error Rate
#
# RECOMMENDATIONS
# ────────────────────────────────────────────────────────────
# ✓ High performer - consider cloning
# ✓ Excellent code quality
# ⚠ Reduce error rate (currently 5%, target: <2%)
# ⚠ Memory usage higher than average (124 MB vs 98 MB avg)
```

## See Also

- `/agent-clone` - Clone high performers
- `/agent-retire` - Retire underperformers
- `/agent-metrics` - Real-time metrics
- `/performance-analysis` - Detailed analysis

---

**Version**: 2.0.0
