#!/usr/bin/env python3
"""
AgentDB Cache Optimization Script
Analyzes access patterns and recommends optimal cache configuration

Usage:
    python cache_optimize.py --db .agentdb/vectors.db --analyze
    python cache_optimize.py --db .agentdb/vectors.db --recommend
    python cache_optimize.py --db .agentdb/vectors.db --generate-config
"""

import argparse
import sqlite3
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
from pathlib import Path
import math


class CacheOptimizer:
    """Analyzes AgentDB access patterns and optimizes cache configuration"""

    def __init__(self, db_path: str):
        """Initialize optimizer with database path"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        self.analysis = {
            'total_patterns': 0,
            'access_distribution': {},
            'domain_distribution': {},
            'type_distribution': {},
            'hot_patterns': [],
            'cold_patterns': [],
            'cache_recommendation': {}
        }

    def analyze_access_patterns(self):
        """Analyze pattern access frequency and distribution"""
        print("Analyzing access patterns...")

        # Get all patterns with usage statistics
        self.cursor.execute('''
            SELECT id, type, domain, usage_count, last_used, confidence
            FROM patterns
            ORDER BY usage_count DESC
        ''')

        patterns = self.cursor.fetchall()
        self.analysis['total_patterns'] = len(patterns)

        if self.analysis['total_patterns'] == 0:
            print("⚠️  Warning: No patterns found in database")
            return

        # Analyze usage distribution
        usage_counts = [p[3] for p in patterns]
        total_accesses = sum(usage_counts)

        # Calculate percentiles
        sorted_usage = sorted(usage_counts, reverse=True)

        percentiles = {
            'p50': sorted_usage[len(sorted_usage) // 2] if sorted_usage else 0,
            'p75': sorted_usage[len(sorted_usage) * 3 // 4] if sorted_usage else 0,
            'p90': sorted_usage[len(sorted_usage) * 9 // 10] if sorted_usage else 0,
            'p95': sorted_usage[len(sorted_usage) * 19 // 20] if sorted_usage else 0,
            'p99': sorted_usage[len(sorted_usage) * 99 // 100] if sorted_usage else 0,
        }

        self.analysis['access_distribution'] = {
            'total_accesses': total_accesses,
            'avg_accesses_per_pattern': total_accesses / self.analysis['total_patterns'] if self.analysis['total_patterns'] > 0 else 0,
            'percentiles': percentiles,
            'max_usage': max(usage_counts) if usage_counts else 0,
            'min_usage': min(usage_counts) if usage_counts else 0
        }

        # Identify hot patterns (top 20% by usage)
        hot_threshold = percentiles['p90']
        hot_patterns = [p for p in patterns if p[3] >= hot_threshold]
        self.analysis['hot_patterns'] = hot_patterns[:100]  # Top 100 hot patterns

        # Identify cold patterns (bottom 20% by usage)
        cold_threshold = percentiles['p50']
        cold_patterns = [p for p in patterns if p[3] <= cold_threshold]
        self.analysis['cold_patterns'] = cold_patterns[:100]  # Bottom 100 cold patterns

        # Domain distribution
        domain_counts = Counter(p[2] for p in patterns)
        self.analysis['domain_distribution'] = dict(domain_counts.most_common(10))

        # Type distribution
        type_counts = Counter(p[1] for p in patterns)
        self.analysis['type_distribution'] = dict(type_counts.most_common(10))

        print(f"✅ Analyzed {self.analysis['total_patterns']:,} patterns")

    def calculate_cache_recommendation(self):
        """Calculate optimal cache size and configuration"""
        print("\nCalculating cache recommendations...")

        total_patterns = self.analysis['total_patterns']
        hot_pattern_count = len(self.analysis['hot_patterns'])

        # Calculate optimal cache size using Pareto principle (80/20 rule)
        # Cache should hold patterns that account for 80% of accesses

        access_dist = self.analysis['access_distribution']
        total_accesses = access_dist['total_accesses']

        # Sort patterns by usage
        self.cursor.execute('''
            SELECT id, usage_count
            FROM patterns
            ORDER BY usage_count DESC
        ''')

        patterns_by_usage = self.cursor.fetchall()

        cumulative_accesses = 0
        target_accesses = total_accesses * 0.8  # 80% of accesses
        cache_size = 0

        for pattern_id, usage_count in patterns_by_usage:
            cumulative_accesses += usage_count
            cache_size += 1

            if cumulative_accesses >= target_accesses:
                break

        # Round up to nearest 100
        cache_size = math.ceil(cache_size / 100) * 100

        # Calculate expected hit rate
        expected_hit_rate = (cumulative_accesses / total_accesses) if total_accesses > 0 else 0

        # Memory estimation (assume 10KB per cached pattern on average)
        cache_memory_mb = (cache_size * 10) / 1024

        # Recommendations for different scenarios
        recommendations = {
            'optimal': {
                'cache_size': cache_size,
                'expected_hit_rate': expected_hit_rate,
                'memory_mb': cache_memory_mb,
                'description': '80% hit rate (Pareto optimal)'
            },
            'conservative': {
                'cache_size': cache_size // 2,
                'expected_hit_rate': expected_hit_rate * 0.7,
                'memory_mb': cache_memory_mb / 2,
                'description': 'Lower memory usage, ~60-70% hit rate'
            },
            'aggressive': {
                'cache_size': min(cache_size * 2, total_patterns),
                'expected_hit_rate': min(expected_hit_rate * 1.2, 0.95),
                'memory_mb': min(cache_memory_mb * 2, (total_patterns * 10) / 1024),
                'description': 'Maximum performance, >90% hit rate'
            },
            'minimal': {
                'cache_size': 100,
                'expected_hit_rate': 0.4,
                'memory_mb': 1,
                'description': 'Minimal memory footprint'
            }
        }

        self.analysis['cache_recommendation'] = recommendations

        print(f"✅ Cache size recommendation: {cache_size} patterns")
        print(f"   Expected hit rate: {expected_hit_rate*100:.1f}%")
        print(f"   Memory usage: {cache_memory_mb:.2f} MB")

    def print_analysis(self):
        """Print comprehensive analysis report"""
        print("\n" + "="*70)
        print("AGENTDB CACHE OPTIMIZATION ANALYSIS")
        print("="*70)

        # Database overview
        print(f"\n📊 Database Overview:")
        print(f"  Total Patterns: {self.analysis['total_patterns']:,}")
        print(f"  Hot Patterns (top 10%): {len(self.analysis['hot_patterns']):,}")
        print(f"  Cold Patterns (bottom 50%): {len(self.analysis['cold_patterns']):,}")

        # Access distribution
        dist = self.analysis['access_distribution']
        print(f"\n📈 Access Distribution:")
        print(f"  Total Accesses: {dist['total_accesses']:,}")
        print(f"  Avg Accesses/Pattern: {dist['avg_accesses_per_pattern']:.1f}")
        print(f"  Max Usage: {dist['max_usage']:,}")
        print(f"  Min Usage: {dist['min_usage']:,}")

        print(f"\n  Percentiles:")
        for p, val in dist['percentiles'].items():
            print(f"    {p.upper()}: {val:,} accesses")

        # Domain distribution
        print(f"\n🏷️  Top Domains:")
        for domain, count in list(self.analysis['domain_distribution'].items())[:5]:
            percentage = (count / self.analysis['total_patterns']) * 100
            print(f"  {domain}: {count:,} ({percentage:.1f}%)")

        # Cache recommendations
        print(f"\n💾 Cache Configuration Recommendations:")
        print()

        for scenario, config in self.analysis['cache_recommendation'].items():
            print(f"  {scenario.upper()}:")
            print(f"    Cache Size: {config['cache_size']:,} patterns")
            print(f"    Expected Hit Rate: {config['expected_hit_rate']*100:.1f}%")
            print(f"    Memory Usage: {config['memory_mb']:.2f} MB")
            print(f"    Description: {config['description']}")
            print()

        # Hot patterns sample
        print(f"\n🔥 Sample Hot Patterns (Top 10):")
        for i, pattern in enumerate(self.analysis['hot_patterns'][:10], 1):
            pattern_id, pattern_type, domain, usage_count, last_used, confidence = pattern
            print(f"  {i}. {pattern_type}/{domain}: {usage_count:,} accesses (confidence: {confidence:.2f})")

        print("\n" + "="*70)

    def generate_config_file(self, output_path: str = None):
        """Generate optimized TypeScript configuration file"""
        if output_path is None:
            output_path = ".agentdb/cache-config-optimized.ts"

        optimal_config = self.analysis['cache_recommendation']['optimal']

        config_content = f'''/**
 * AgentDB Optimized Cache Configuration
 * Auto-generated by cache_optimize.py
 *
 * Database: {self.db_path}
 * Total Patterns: {self.analysis['total_patterns']:,}
 * Analysis Date: {Path(self.db_path).stat().st_mtime}
 */

import {{ createAgentDBAdapter }} from 'agentic-flow/reasoningbank';

// Optimal cache configuration (80% hit rate)
export const cacheConfig = {{
  dbPath: '{self.db_path}',
  cacheSize: {optimal_config['cache_size']},

  // Expected Performance:
  // - Hit Rate: {optimal_config['expected_hit_rate']*100:.1f}%
  // - Memory: {optimal_config['memory_mb']:.2f} MB
  // - Cache Latency: <1ms (hit), ~2ms (miss)

  enableLearning: true,
  enableReasoning: true,
}};

// Alternative configurations for different scenarios
export const cacheConfigs = {{
  optimal: {{
    cacheSize: {optimal_config['cache_size']},
    description: '{optimal_config['description']}',
    expectedHitRate: {optimal_config['expected_hit_rate']:.3f},
  }},
  conservative: {{
    cacheSize: {self.analysis['cache_recommendation']['conservative']['cache_size']},
    description: '{self.analysis['cache_recommendation']['conservative']['description']}',
    expectedHitRate: {self.analysis['cache_recommendation']['conservative']['expected_hit_rate']:.3f},
  }},
  aggressive: {{
    cacheSize: {self.analysis['cache_recommendation']['aggressive']['cache_size']},
    description: '{self.analysis['cache_recommendation']['aggressive']['description']}',
    expectedHitRate: {self.analysis['cache_recommendation']['aggressive']['expected_hit_rate']:.3f},
  }},
  minimal: {{
    cacheSize: {self.analysis['cache_recommendation']['minimal']['cache_size']},
    description: '{self.analysis['cache_recommendation']['minimal']['description']}',
    expectedHitRate: {self.analysis['cache_recommendation']['minimal']['expected_hit_rate']:.3f},
  }},
}};

// Hot domains (cache priority)
export const hotDomains = {json.dumps(list(self.analysis['domain_distribution'].keys())[:5], indent=2)};

// Initialize with optimal cache
export async function createCachedAdapter(scenario: keyof typeof cacheConfigs = 'optimal') {{
  const config = cacheConfigs[scenario];

  const adapter = await createAgentDBAdapter({{
    ...cacheConfig,
    cacheSize: config.cacheSize,
  }});

  console.log(`AgentDB cache initialized: ${{scenario}}`);
  console.log(`  Cache Size: ${{config.cacheSize}} patterns`);
  console.log(`  Expected Hit Rate: ${{(config.expectedHitRate * 100).toFixed(1)}}%`);

  return adapter;
}}

// Monitor cache performance
export async function monitorCache(adapter: any) {{
  const stats = await adapter.getStats();

  console.log('Cache Performance:');
  console.log(`  Hit Rate: ${{(stats.cacheHitRate * 100).toFixed(1)}}%`);
  console.log(`  Total Patterns: ${{stats.totalPatterns?.toLocaleString()}}`);
  console.log(`  Database Size: ${{(stats.dbSize / (1024 * 1024)).toFixed(2)}} MB`);

  return stats;
}}
'''

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(config_content)

        print(f"\n✅ Configuration file generated: {output_path}")

        return output_path

    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and optimize AgentDB cache configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Full analysis with recommendations
  python cache_optimize.py --db .agentdb/vectors.db --analyze --recommend

  # Generate optimized configuration file
  python cache_optimize.py --db .agentdb/vectors.db --generate-config

  # Specify custom output path
  python cache_optimize.py --db .agentdb/vectors.db --generate-config --output my-cache-config.ts
        '''
    )

    parser.add_argument(
        '--db',
        required=True,
        help='Path to AgentDB database'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze access patterns'
    )
    parser.add_argument(
        '--recommend',
        action='store_true',
        help='Calculate cache recommendations'
    )
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate TypeScript configuration file'
    )
    parser.add_argument(
        '--output',
        help='Output path for configuration file'
    )

    args = parser.parse_args()

    # Validate database
    if not Path(args.db).exists():
        print(f"❌ Error: Database not found: {args.db}")
        return 1

    # Initialize optimizer
    optimizer = CacheOptimizer(args.db)

    try:
        # Run analysis
        optimizer.analyze_access_patterns()

        if args.recommend or args.generate_config:
            optimizer.calculate_cache_recommendation()

        if args.analyze:
            optimizer.print_analysis()

        if args.generate_config:
            output_path = args.output if args.output else None
            optimizer.generate_config_file(output_path)

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        optimizer.close()


if __name__ == '__main__':
    exit(main())
