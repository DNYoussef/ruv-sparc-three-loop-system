#!/usr/bin/env node
/**
 * Latency Reduction Example - Comprehensive demonstration of latency optimization
 *
 * This example demonstrates:
 * 1. Identifying latency bottlenecks
 * 2. Async/await optimization
 * 3. Parallelization strategies
 * 4. Caching techniques
 * 5. Request batching and pooling
 *
 * Run: node latency-reduction-example.js
 */

const http = require('http');
const { performance } = require('perf_hooks');

class LatencyOptimizationDemo {
  constructor() {
    this.cache = new Map();
    this.results = {};
  }

  // INEFFICIENT IMPLEMENTATIONS (High Latency)

  async fetchDataSequentialInefficient(urls) {
    /**
     * Fetch data sequentially - high latency due to waiting
     */
    const results = [];
    const start = performance.now();

    for (const url of urls) {
      const data = await this.simulateFetch(url, 100); // 100ms each
      results.push(data);
    }

    const duration = performance.now() - start;
    return { results, duration };
  }

  async processDataSyncInefficient(items) {
    /**
     * Process data synchronously - blocks event loop
     */
    const start = performance.now();
    const results = [];

    for (const item of items) {
      // Simulate CPU-intensive work (synchronous)
      const result = this.cpuIntensiveSync(item);
      results.push(result);
    }

    const duration = performance.now() - start;
    return { results, duration };
  }

  async databaseQueriesInefficient(ids) {
    /**
     * N+1 query problem - multiple round trips to database
     */
    const start = performance.now();
    const results = [];

    for (const id of ids) {
      const data = await this.simulateDbQuery(id);
      results.push(data);
    }

    const duration = performance.now() - start;
    return { results, duration };
  }

  async cacheNoneInefficient(key) {
    /**
     * No caching - always fetch from source
     */
    const start = performance.now();
    const data = await this.simulateFetch(key, 50);
    const duration = performance.now() - start;

    return { data, duration };
  }

  // OPTIMIZED IMPLEMENTATIONS (Low Latency)

  async fetchDataParallelOptimized(urls) {
    /**
     * Fetch data in parallel - reduced latency
     */
    const start = performance.now();

    // Execute all requests in parallel
    const promises = urls.map(url => this.simulateFetch(url, 100));
    const results = await Promise.all(promises);

    const duration = performance.now() - start;
    return { results, duration };
  }

  async processDataAsyncOptimized(items) {
    /**
     * Process data asynchronously - non-blocking
     */
    const start = performance.now();

    // Use setImmediate to break up work
    const results = await this.processInChunks(items, 100);

    const duration = performance.now() - start;
    return { results, duration };
  }

  async databaseQueriesBatchedOptimized(ids) {
    /**
     * Batch queries - single round trip to database
     */
    const start = performance.now();

    // Batch all IDs into single query
    const results = await this.simulateDbBatchQuery(ids);

    const duration = performance.now() - start;
    return { results, duration };
  }

  async cacheWithLRUOptimized(key) {
    /**
     * Use LRU cache to reduce fetches
     */
    const start = performance.now();

    // Check cache first
    if (this.cache.has(key)) {
      const data = this.cache.get(key);
      const duration = performance.now() - start;
      return { data, duration, cached: true };
    }

    // Fetch and cache
    const data = await this.simulateFetch(key, 50);
    this.cache.set(key, data);

    // Implement simple LRU (limit cache size)
    if (this.cache.size > 100) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    const duration = performance.now() - start;
    return { data, duration, cached: false };
  }

  // HELPER METHODS

  async simulateFetch(url, latency = 100) {
    /**
     * Simulate network request with latency
     */
    return new Promise(resolve => {
      setTimeout(() => {
        resolve({ url, data: `Data from ${url}`, timestamp: Date.now() });
      }, latency);
    });
  }

  async simulateDbQuery(id) {
    /**
     * Simulate database query with latency
     */
    return new Promise(resolve => {
      setTimeout(() => {
        resolve({ id, value: id * 2, timestamp: Date.now() });
      }, 10);
    });
  }

  async simulateDbBatchQuery(ids) {
    /**
     * Simulate batched database query
     */
    return new Promise(resolve => {
      setTimeout(() => {
        resolve(ids.map(id => ({ id, value: id * 2, timestamp: Date.now() })));
      }, 15); // Slightly longer but much better than N queries
    });
  }

  cpuIntensiveSync(item) {
    /**
     * Simulate CPU-intensive synchronous work
     */
    let result = item;
    for (let i = 0; i < 1000000; i++) {
      result = Math.sqrt(result + i);
    }
    return result;
  }

  async processInChunks(items, chunkSize) {
    /**
     * Process large array in chunks to avoid blocking
     */
    const results = [];

    for (let i = 0; i < items.length; i += chunkSize) {
      const chunk = items.slice(i, i + chunkSize);

      // Process chunk
      for (const item of chunk) {
        results.push(this.cpuIntensiveSync(item));
      }

      // Yield to event loop
      if (i + chunkSize < items.length) {
        await new Promise(resolve => setImmediate(resolve));
      }
    }

    return results;
  }

  // BENCHMARKING METHODS

  async benchmarkParallelization() {
    console.log('\n' + '='.repeat(60));
    console.log('Benchmarking Parallelization (10 requests)');
    console.log('='.repeat(60));

    const urls = Array.from({ length: 10 }, (_, i) => `http://api.example.com/data/${i}`);

    // Sequential (inefficient)
    console.log('\n1. Sequential Fetching (Inefficient)...');
    const sequential = await this.fetchDataSequentialInefficient(urls);
    console.log(`   Time: ${sequential.duration.toFixed(2)}ms`);
    console.log(`   Requests: ${sequential.results.length}`);

    // Parallel (optimized)
    console.log('\n2. Parallel Fetching (Optimized)...');
    const parallel = await this.fetchDataParallelOptimized(urls);
    console.log(`   Time: ${parallel.duration.toFixed(2)}ms`);
    console.log(`   Requests: ${parallel.results.length}`);

    // Calculate improvement
    const improvement = ((sequential.duration - parallel.duration) / sequential.duration) * 100;
    const speedup = sequential.duration / parallel.duration;

    console.log('\n3. Results:');
    console.log(`   Latency Reduction: ${improvement.toFixed(1)}%`);
    console.log(`   Speedup: ${speedup.toFixed(2)}x`);
    console.log(`   Time Saved: ${(sequential.duration - parallel.duration).toFixed(2)}ms`);

    return {
      sequential_ms: sequential.duration,
      parallel_ms: parallel.duration,
      improvement_percent: improvement,
      speedup: speedup
    };
  }

  async benchmarkAsyncProcessing() {
    console.log('\n' + '='.repeat(60));
    console.log('Benchmarking Async Processing (1000 items)');
    console.log('='.repeat(60));

    const items = Array.from({ length: 1000 }, (_, i) => i);

    // Synchronous (inefficient)
    console.log('\n1. Synchronous Processing (Inefficient)...');
    const sync = await this.processDataSyncInefficient(items);
    console.log(`   Time: ${sync.duration.toFixed(2)}ms`);
    console.log(`   Items processed: ${sync.results.length}`);

    // Asynchronous (optimized)
    console.log('\n2. Asynchronous Processing (Optimized)...');
    const async = await this.processDataAsyncOptimized(items);
    console.log(`   Time: ${async.duration.toFixed(2)}ms`);
    console.log(`   Items processed: ${async.results.length}`);

    // Calculate improvement
    const improvement = ((sync.duration - async.duration) / sync.duration) * 100;
    const speedup = sync.duration / async.duration;

    console.log('\n3. Results:');
    console.log(`   Latency Reduction: ${improvement.toFixed(1)}%`);
    console.log(`   Speedup: ${speedup.toFixed(2)}x`);
    console.log(`   Event Loop: Non-blocking with async approach`);

    return {
      sync_ms: sync.duration,
      async_ms: async.duration,
      improvement_percent: improvement,
      speedup: speedup
    };
  }

  async benchmarkDatabaseBatching() {
    console.log('\n' + '='.repeat(60));
    console.log('Benchmarking Database Batching (100 queries)');
    console.log('='.repeat(60));

    const ids = Array.from({ length: 100 }, (_, i) => i);

    // N+1 queries (inefficient)
    console.log('\n1. Individual Queries - N+1 Problem (Inefficient)...');
    const individual = await this.databaseQueriesInefficient(ids);
    console.log(`   Time: ${individual.duration.toFixed(2)}ms`);
    console.log(`   Queries: ${ids.length}`);

    // Batched query (optimized)
    console.log('\n2. Batched Query (Optimized)...');
    const batched = await this.databaseQueriesBatchedOptimized(ids);
    console.log(`   Time: ${batched.duration.toFixed(2)}ms`);
    console.log(`   Queries: 1 (batched)`);

    // Calculate improvement
    const improvement = ((individual.duration - batched.duration) / individual.duration) * 100;
    const speedup = individual.duration / batched.duration;

    console.log('\n3. Results:');
    console.log(`   Latency Reduction: ${improvement.toFixed(1)}%`);
    console.log(`   Speedup: ${speedup.toFixed(2)}x`);
    console.log(`   Round Trips: ${ids.length} → 1`);

    return {
      individual_ms: individual.duration,
      batched_ms: batched.duration,
      improvement_percent: improvement,
      speedup: speedup
    };
  }

  async benchmarkCaching() {
    console.log('\n' + '='.repeat(60));
    console.log('Benchmarking Caching (100 requests, 50% cache hit rate)');
    console.log('='.repeat(60));

    const keys = Array.from({ length: 100 }, (_, i) => `key_${i % 50}`); // 50% duplicates

    // No caching (inefficient)
    console.log('\n1. No Caching (Inefficient)...');
    this.cache.clear();
    const start1 = performance.now();
    for (const key of keys) {
      await this.cacheNoneInefficient(key);
    }
    const noCacheDuration = performance.now() - start1;
    console.log(`   Time: ${noCacheDuration.toFixed(2)}ms`);
    console.log(`   Cache hits: 0`);

    // With caching (optimized)
    console.log('\n2. With LRU Cache (Optimized)...');
    this.cache.clear();
    let cacheHits = 0;
    const start2 = performance.now();
    for (const key of keys) {
      const result = await this.cacheWithLRUOptimized(key);
      if (result.cached) cacheHits++;
    }
    const cacheDuration = performance.now() - start2;
    console.log(`   Time: ${cacheDuration.toFixed(2)}ms`);
    console.log(`   Cache hits: ${cacheHits}`);

    // Calculate improvement
    const improvement = ((noCacheDuration - cacheDuration) / noCacheDuration) * 100;
    const speedup = noCacheDuration / cacheDuration;

    console.log('\n3. Results:');
    console.log(`   Latency Reduction: ${improvement.toFixed(1)}%`);
    console.log(`   Speedup: ${speedup.toFixed(2)}x`);
    console.log(`   Cache Hit Rate: ${(cacheHits / keys.length * 100).toFixed(1)}%`);

    return {
      no_cache_ms: noCacheDuration,
      with_cache_ms: cacheDuration,
      improvement_percent: improvement,
      speedup: speedup,
      cache_hit_rate: cacheHits / keys.length
    };
  }

  async runComprehensiveAnalysis() {
    console.log('\n' + '='.repeat(60));
    console.log('COMPREHENSIVE LATENCY OPTIMIZATION DEMONSTRATION');
    console.log('='.repeat(60));

    const results = {
      parallelization: await this.benchmarkParallelization(),
      async_processing: await this.benchmarkAsyncProcessing(),
      database_batching: await this.benchmarkDatabaseBatching(),
      caching: await this.benchmarkCaching()
    };

    // Generate summary
    console.log('\n' + '='.repeat(60));
    console.log('SUMMARY REPORT');
    console.log('='.repeat(60));

    let totalLatencySaved = 0;
    let avgImprovement = 0;

    for (const [category, metrics] of Object.entries(results)) {
      const latencySaved = (metrics.sequential_ms || metrics.sync_ms || metrics.individual_ms || metrics.no_cache_ms) -
                          (metrics.parallel_ms || metrics.async_ms || metrics.batched_ms || metrics.with_cache_ms);
      totalLatencySaved += latencySaved;
      avgImprovement += metrics.improvement_percent;

      console.log(`\n${category.toUpperCase().replace(/_/g, ' ')}:`);
      console.log(`  Latency Saved: ${latencySaved.toFixed(2)}ms`);
      console.log(`  Improvement: ${metrics.improvement_percent.toFixed(1)}%`);
      console.log(`  Speedup: ${metrics.speedup.toFixed(2)}x`);
    }

    avgImprovement /= Object.keys(results).length;

    console.log(`\nOVERALL:`);
    console.log(`  Total Latency Saved: ${totalLatencySaved.toFixed(2)}ms`);
    console.log(`  Average Improvement: ${avgImprovement.toFixed(1)}%`);

    return results;
  }
}

async function main() {
  console.log('Starting Latency Reduction Example...\n');

  const demo = new LatencyOptimizationDemo();
  const results = await demo.runComprehensiveAnalysis();

  console.log('\n' + '='.repeat(60));
  console.log('KEY TAKEAWAYS');
  console.log('='.repeat(60));
  console.log(`
1. PARALLELIZATION:
   - Use Promise.all() for independent async operations
   - Fetch data in parallel instead of sequentially
   - Can achieve near-linear speedup (10x for 10 parallel requests)

2. ASYNC/AWAIT BEST PRACTICES:
   - Break up CPU-intensive work with setImmediate()
   - Use async/await for I/O operations
   - Avoid blocking the event loop
   - Process large datasets in chunks

3. DATABASE OPTIMIZATION:
   - Batch queries to reduce round trips
   - Solve N+1 query problems
   - Use connection pooling
   - Implement query result caching

4. CACHING STRATEGIES:
   - Implement LRU cache for frequently accessed data
   - Set appropriate cache TTLs
   - Monitor cache hit rates
   - Use multi-level caching (memory + Redis)

5. LATENCY REDUCTION TECHNIQUES:
   - Request batching and pooling
   - HTTP/2 multiplexing
   - CDN for static assets
   - Compression (gzip, brotli)
   - Connection keep-alive

6. MONITORING:
   - Track P50, P95, P99 latencies
   - Set up latency budgets
   - Use distributed tracing
   - Monitor cache effectiveness
  `);

  console.log('\nLatency Reduction Example Complete!');
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = { LatencyOptimizationDemo };
