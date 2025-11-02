# Example 3: Debugging Memory Leak (Advanced)

## Overview

This example demonstrates systematic investigation of a memory leak causing gradual performance degradation and eventual crashes in a long-running Node.js application.

**Difficulty**: Advanced
**Estimated Time**: 90-120 minutes
**Agents Used**: code-analyzer, coder, tester
**Tools Required**: Node.js heap profiler, Chrome DevTools, clinic.js

## The Problem

### Initial Symptom

**User Report**: "Application becomes slow over time and crashes after 24-48 hours of uptime with 'JavaScript heap out of memory' error."

**Error Logs**:
```
<--- Last few GCs --->
[94832:0x104808000]   127569 ms: Mark-sweep 2047.8 (2083.2) -> 2047.7 (2083.2) MB, 1523.5 / 0.0 ms  (average mu = 0.126, current mu = 0.001) allocation failure scavenge might not succeed
[94832:0x104808000]   129156 ms: Mark-sweep 2048.7 (2084.2) -> 2048.6 (2084.2) MB, 1586.2 / 0.0 ms  (average mu = 0.063, current mu = 0.001) allocation failure scavenge might not succeed

<--- JS stacktrace --->
FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory
```

**Monitoring Data**:
- Memory usage grows linearly: +50MB per hour
- Garbage collection becoming more frequent and longer
- Response times degrading proportionally with memory usage
- CPU usage spikes during GC cycles

## Phase 1: Symptom Identification

### Agent: code-analyzer

**Investigation Steps**:

1. **Collect baseline metrics**:
```bash
# Memory profile at startup
Initial heap: 45 MB
Expected steady-state: 100-200 MB
Actual after 24h: 1.2 GB (6-12x expected!)

# GC statistics
Minor GC: Every 30s initially → Every 5s after 24h
Major GC: Every 10min initially → Every 2min after 24h
GC pause time: 20ms average → 800ms average
```

2. **Capture heap snapshots**:
```bash
# Take snapshots at intervals
node --expose-gc --inspect app.js

# In Chrome DevTools:
# Snapshot 1 (startup): 45 MB
# Snapshot 2 (1 hour): 95 MB
# Snapshot 3 (4 hours): 245 MB
# Snapshot 4 (12 hours): 650 MB
```

3. **Analyze memory growth patterns**:
```javascript
// Using clinic.js bubbleprof
clinic bubbleprof -- node app.js

// Output shows:
// - Event listener count growing unbounded
// - Array of objects continuously expanding
// - Closure retaining references
```

**Symptom Report**:
```
Issue ID: BUG-2024-087
Type: Memory Leak
Severity: Critical (causes crashes)
Growth Rate: ~50 MB/hour
Root Location: data-stream.js (suspected)
Expected: Memory stabilizes after initialization
Actual: Linear memory growth until crash
Pattern: Leaked event listeners and unclosed subscriptions
Time to Failure: 24-48 hours
```

## Phase 2: Root Cause Analysis

### Agent: code-analyzer + coder

**Heap Snapshot Analysis**:

Using Chrome DevTools Memory Profiler:

1. **Snapshot Comparison** (Hour 1 vs Hour 4):
```
Retained Size Deltas:
+1,250 Array objects (23 MB)
+4,800 Closure objects (45 MB)
+120,000 (function) objects (89 MB)

Suspicious Patterns:
- `window` object retaining 4,800 event listeners
- DataStream.listeners array growing unbounded
- WebSocket callbacks not being removed
```

2. **Retainer Path Analysis**:
```
Object → Retained by → listeners (Array) →
  DataStream instance → global.activeStreams →
  window → <global>

Conclusion: Event listeners accumulating on global window object
```

**Source Code Analysis**:

```javascript
// data-stream.js - PROBLEMATIC CODE
class DataStream {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.listeners = [];
    this.connection = null;
  }

  // PROBLEM 1: Event listeners never removed
  subscribe(callback) {
    window.addEventListener('data-received', callback);
    this.listeners.push(callback);
    // Missing: No unsubscribe function returned
    // Missing: No cleanup on stream close
  }

  // PROBLEM 2: WebSocket not properly closed
  async connect() {
    this.connection = new WebSocket(this.endpoint);

    this.connection.onmessage = (event) => {
      const data = JSON.parse(event.data);
      window.dispatchEvent(new CustomEvent('data-received', { detail: data }));
    };

    // Missing: Connection cleanup logic
    // Missing: Error handlers that close connection
  }

  // PROBLEM 3: Incomplete cleanup
  close() {
    if (this.connection) {
      this.connection.close();
    }
    // Missing: Remove event listeners from window
    // Missing: Clear listeners array
    // Missing: Remove instance from global tracking
  }
}

// app.js - PROBLEM 4: Instances never cleaned up
const activeStreams = new Map();

function createStream(userId) {
  const stream = new DataStream(`wss://api.example.com/user/${userId}`);
  activeStreams.set(userId, stream);

  stream.subscribe((data) => {
    processUserData(userId, data);
  });

  stream.connect();
  // Missing: No stream cleanup when user disconnects
  // Missing: No maximum stream limit
}

// PROBLEM 5: Called repeatedly without cleanup
setInterval(() => {
  // Refreshes user data every minute
  users.forEach(user => {
    createStream(user.id); // Creates new stream every time!
  });
}, 60000);
```

**Root Cause Statement**:

> The application suffers from multiple memory leaks:
> 1. **Event listeners never removed**: `subscribe()` adds listeners to `window` but never removes them
> 2. **Unclosed connections**: WebSocket connections accumulate without cleanup
> 3. **Retained closures**: Callback functions capture context and remain in memory
> 4. **Global tracking without pruning**: `activeStreams` Map grows unbounded
> 5. **Repeated instantiation**: New streams created without closing old ones
>
> These combine to cause linear memory growth at ~50MB/hour, eventually exhausting heap space.

**Why This Occurs**:
- No explicit lifecycle management for streams
- Missing cleanup/destructor pattern
- No WeakMap/WeakRef for automatic garbage collection
- Infinite interval creating new resources without bounds
- Event listeners on global `window` prevent GC of DataStream instances

## Phase 3: Fix Generation

### Agent: coder

**Solution Approaches Evaluated**:

### Option 1: Add Manual Cleanup (Partial Fix)
```javascript
close() {
  this.listeners.forEach(callback => {
    window.removeEventListener('data-received', callback);
  });
  this.listeners = [];
  this.connection?.close();
}
```
**Pros**: Simple, minimal changes
**Cons**: Relies on manual cleanup calls, easy to forget
**Trade-offs**: Still prone to leaks if close() not called

### Option 2: Return Unsubscribe Functions (Better)
```javascript
subscribe(callback) {
  window.addEventListener('data-received', callback);
  this.listeners.add(callback);

  return () => {
    window.removeEventListener('data-received', callback);
    this.listeners.delete(callback);
  };
}
```
**Pros**: Explicit contract, prevents accumulation
**Cons**: Caller must remember to call unsubscribe
**Trade-offs**: Better but not foolproof

### Option 3: Automatic Lifecycle Management + WeakMap (CHOSEN)
```javascript
class DataStream {
  constructor() {
    this.listeners = new Set(); // Use Set instead of Array
    this.cleanupHandlers = [];
    this.isActive = true;
  }

  subscribe(callback) {
    const handler = callback.bind(this);
    window.addEventListener('data-received', handler);
    this.listeners.add(handler);

    const unsubscribe = () => {
      window.removeEventListener('data-received', handler);
      this.listeners.delete(handler);
    };

    this.cleanupHandlers.push(unsubscribe);
    return unsubscribe;
  }

  destroy() {
    this.isActive = false;
    this.cleanupHandlers.forEach(cleanup => cleanup());
    this.cleanupHandlers = [];
    this.listeners.clear();
    this.connection?.close();
  }
}

// Use WeakMap for automatic cleanup
const streamRegistry = new WeakMap();
```
**Pros**: Automatic cleanup, hard to misuse, explicit lifecycle
**Cons**: Requires discipline to call destroy()
**Trade-offs**: Best balance of safety and usability

**Selected Solution**: **Option 3** with additional safeguards

### Implementation

```javascript
// data-stream.js - FIXED VERSION

/**
 * DataStream - Real-time data streaming with proper lifecycle management
 *
 * MEMORY SAFETY:
 * - All event listeners are tracked and cleaned up
 * - WebSocket connections properly closed
 * - Automatic cleanup on errors
 * - Explicit destroy() method for lifecycle management
 *
 * Historical Note: Previous version leaked event listeners and connections.
 * See BUG-2024-087 for details on the memory leak incident.
 */
class DataStream {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.listeners = new Set(); // Set prevents duplicates
    this.cleanupHandlers = [];
    this.connection = null;
    this.isActive = true;
    this.reconnectAttempts = 0;
    this.maxReconnects = 5;
  }

  /**
   * Subscribe to data events
   * @param {Function} callback - Event handler
   * @returns {Function} Unsubscribe function
   */
  subscribe(callback) {
    if (!this.isActive) {
      throw new Error('Cannot subscribe to inactive stream');
    }

    // Wrap callback to prevent memory leaks from closures
    const handler = (event) => {
      if (this.isActive) {
        callback(event.detail);
      }
    };

    window.addEventListener('data-received', handler);
    this.listeners.add(handler);

    // Return unsubscribe function
    const unsubscribe = () => {
      window.removeEventListener('data-received', handler);
      this.listeners.delete(handler);
    };

    this.cleanupHandlers.push(unsubscribe);
    return unsubscribe;
  }

  /**
   * Connect to WebSocket endpoint with automatic cleanup on errors
   */
  async connect() {
    if (!this.isActive) {
      throw new Error('Cannot connect inactive stream');
    }

    return new Promise((resolve, reject) => {
      try {
        this.connection = new WebSocket(this.endpoint);

        this.connection.onopen = () => {
          this.reconnectAttempts = 0;
          resolve();
        };

        this.connection.onmessage = (event) => {
          if (!this.isActive) return;

          try {
            const data = JSON.parse(event.data);
            window.dispatchEvent(new CustomEvent('data-received', { detail: data }));
          } catch (error) {
            console.error('Failed to parse message:', error);
          }
        };

        this.connection.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.handleConnectionError();
        };

        this.connection.onclose = () => {
          if (this.isActive && this.reconnectAttempts < this.maxReconnects) {
            this.reconnect();
          } else {
            this.destroy(); // Auto-cleanup on connection loss
          }
        };

      } catch (error) {
        this.destroy(); // Cleanup on connection failure
        reject(error);
      }
    });
  }

  /**
   * Reconnect with exponential backoff
   */
  async reconnect() {
    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnects})`);

    await new Promise(resolve => setTimeout(resolve, delay));

    if (this.isActive) {
      await this.connect();
    }
  }

  /**
   * Handle connection errors with cleanup
   */
  handleConnectionError() {
    if (this.reconnectAttempts >= this.maxReconnects) {
      console.error('Max reconnection attempts reached, destroying stream');
      this.destroy();
    }
  }

  /**
   * Close stream and clean up ALL resources
   * CRITICAL: Must be called to prevent memory leaks
   */
  destroy() {
    if (!this.isActive) {
      return; // Already destroyed
    }

    this.isActive = false;

    // Remove all event listeners
    this.cleanupHandlers.forEach(cleanup => {
      try {
        cleanup();
      } catch (error) {
        console.error('Error during cleanup:', error);
      }
    });
    this.cleanupHandlers = [];
    this.listeners.clear();

    // Close WebSocket connection
    if (this.connection) {
      try {
        this.connection.onclose = null; // Prevent reconnect attempt
        this.connection.close();
        this.connection = null;
      } catch (error) {
        console.error('Error closing connection:', error);
      }
    }

    console.log(`DataStream destroyed for ${this.endpoint}`);
  }
}

// app.js - FIXED stream management

/**
 * Stream manager with automatic cleanup and limits
 */
class StreamManager {
  constructor(maxStreams = 100) {
    this.streams = new Map();
    this.maxStreams = maxStreams;

    // Periodic cleanup of inactive streams
    this.cleanupInterval = setInterval(() => {
      this.cleanupInactiveStreams();
    }, 300000); // Every 5 minutes
  }

  /**
   * Get or create stream for user
   */
  getStream(userId, endpoint) {
    // Return existing stream if active
    if (this.streams.has(userId)) {
      const stream = this.streams.get(userId);
      if (stream.isActive) {
        return stream;
      }
      // Clean up inactive stream
      this.removeStream(userId);
    }

    // Enforce maximum streams limit
    if (this.streams.size >= this.maxStreams) {
      this.evictOldestStream();
    }

    // Create new stream
    const stream = new DataStream(endpoint);
    this.streams.set(userId, stream);

    return stream;
  }

  /**
   * Remove stream and clean up resources
   */
  removeStream(userId) {
    const stream = this.streams.get(userId);
    if (stream) {
      stream.destroy();
      this.streams.delete(userId);
    }
  }

  /**
   * Clean up inactive streams
   */
  cleanupInactiveStreams() {
    const toRemove = [];

    for (const [userId, stream] of this.streams) {
      if (!stream.isActive) {
        toRemove.push(userId);
      }
    }

    toRemove.forEach(userId => this.removeStream(userId));

    if (toRemove.length > 0) {
      console.log(`Cleaned up ${toRemove.length} inactive streams`);
    }
  }

  /**
   * Evict oldest stream when limit reached
   */
  evictOldestStream() {
    const oldestUserId = this.streams.keys().next().value;
    if (oldestUserId) {
      console.warn(`Stream limit reached, evicting stream for user ${oldestUserId}`);
      this.removeStream(oldestUserId);
    }
  }

  /**
   * Destroy all streams (for shutdown)
   */
  destroyAll() {
    clearInterval(this.cleanupInterval);

    for (const [userId, stream] of this.streams) {
      stream.destroy();
    }

    this.streams.clear();
  }
}

// Global stream manager instance
const streamManager = new StreamManager(100);

// Graceful shutdown handling
process.on('SIGTERM', () => {
  console.log('Shutting down gracefully...');
  streamManager.destroyAll();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('Shutting down gracefully...');
  streamManager.destroyAll();
  process.exit(0);
});

// Usage example
function subscribeUserToData(userId) {
  const endpoint = `wss://api.example.com/user/${userId}`;
  const stream = streamManager.getStream(userId, endpoint);

  // Subscribe with automatic cleanup
  const unsubscribe = stream.subscribe((data) => {
    processUserData(userId, data);
  });

  // Connect stream
  stream.connect().catch(error => {
    console.error(`Failed to connect stream for user ${userId}:`, error);
    streamManager.removeStream(userId);
  });

  // Return cleanup function
  return () => {
    unsubscribe();
    streamManager.removeStream(userId);
  };
}
```

**Why This Approach**:
1. **Explicit lifecycle**: `destroy()` method ensures complete cleanup
2. **Automatic cleanup**: Errors trigger automatic resource cleanup
3. **Bounded resources**: StreamManager enforces max streams limit
4. **No silent failures**: All cleanup errors logged
5. **Graceful shutdown**: SIGTERM/SIGINT handlers ensure cleanup
6. **Periodic maintenance**: Background cleanup of inactive streams
7. **Return cleanup functions**: Callers can manually trigger cleanup

## Phase 4: Validation Testing

### Agent: tester

**Test Suite**:

```javascript
// tests/memory-leak.test.js
const { DataStream, StreamManager } = require('../data-stream');
const { measureMemory, forceGC } = require('./test-utils');

describe('Memory Leak Prevention', () => {
  let manager;

  beforeEach(() => {
    manager = new StreamManager(10);
    if (global.gc) {
      global.gc(); // Force GC before each test
    }
  });

  afterEach(() => {
    manager.destroyAll();
  });

  describe('Event listener cleanup', () => {
    it('should remove all event listeners on destroy', async () => {
      const stream = new DataStream('ws://localhost:8080');
      const callback = jest.fn();

      // Count initial listeners
      const initialListeners = window.listenerCount?.('data-received') || 0;

      stream.subscribe(callback);
      stream.subscribe(callback);
      stream.subscribe(callback);

      expect(stream.listeners.size).toBe(3);
      const afterSubscribe = window.listenerCount?.('data-received') || 0;
      expect(afterSubscribe).toBe(initialListeners + 3);

      // Destroy stream
      stream.destroy();

      expect(stream.listeners.size).toBe(0);
      const afterDestroy = window.listenerCount?.('data-received') || 0;
      expect(afterDestroy).toBe(initialListeners);
    });

    it('should clean up listeners when unsubscribe called', () => {
      const stream = new DataStream('ws://localhost:8080');
      const callback = jest.fn();

      const unsubscribe = stream.subscribe(callback);
      expect(stream.listeners.size).toBe(1);

      unsubscribe();
      expect(stream.listeners.size).toBe(0);
    });
  });

  describe('Memory usage under load', () => {
    it('should not leak memory with repeated subscribe/unsubscribe', async () => {
      const stream = new DataStream('ws://localhost:8080');

      const initialMemory = await measureMemory();

      // Create and destroy 1000 subscriptions
      for (let i = 0; i < 1000; i++) {
        const unsubscribe = stream.subscribe(() => {});
        unsubscribe();
      }

      // Force garbage collection
      if (global.gc) global.gc();
      await new Promise(resolve => setTimeout(resolve, 100));

      const finalMemory = await measureMemory();
      const memoryGrowth = finalMemory - initialMemory;

      // Should not grow significantly (< 1MB tolerance)
      expect(memoryGrowth).toBeLessThan(1 * 1024 * 1024);
      stream.destroy();
    });

    it('should not leak memory with repeated stream creation', async () => {
      const initialMemory = await measureMemory();

      // Create and destroy 100 streams
      for (let i = 0; i < 100; i++) {
        const stream = new DataStream(`ws://localhost:808${i}`);
        stream.subscribe(() => {});
        stream.destroy();
      }

      if (global.gc) global.gc();
      await new Promise(resolve => setTimeout(resolve, 100));

      const finalMemory = await measureMemory();
      const memoryGrowth = finalMemory - initialMemory;

      // Should not grow significantly (< 5MB tolerance)
      expect(memoryGrowth).toBeLessThan(5 * 1024 * 1024);
    });
  });

  describe('StreamManager limits', () => {
    it('should enforce maximum streams limit', () => {
      const manager = new StreamManager(5);

      for (let i = 0; i < 10; i++) {
        manager.getStream(`user${i}`, 'ws://localhost:8080');
      }

      // Should have evicted oldest to maintain limit
      expect(manager.streams.size).toBe(5);
      manager.destroyAll();
    });

    it('should clean up inactive streams periodically', async () => {
      jest.useFakeTimers();
      const manager = new StreamManager(10);

      // Create streams
      for (let i = 0; i < 5; i++) {
        manager.getStream(`user${i}`, 'ws://localhost:8080');
      }

      // Manually deactivate some streams
      manager.streams.get('user1').isActive = false;
      manager.streams.get('user3').isActive = false;

      // Fast-forward 5 minutes (cleanup interval)
      jest.advanceTimersByTime(300000);

      // Inactive streams should be removed
      expect(manager.streams.size).toBe(3);
      expect(manager.streams.has('user1')).toBe(false);
      expect(manager.streams.has('user3')).toBe(false);

      manager.destroyAll();
      jest.useRealTimers();
    });
  });

  describe('Long-running stability', () => {
    it('should maintain stable memory over extended period', async () => {
      // Skip in CI (too slow)
      if (process.env.CI) return;

      const measurements = [];
      const duration = 60000; // 1 minute test
      const interval = 5000; // Measure every 5 seconds

      const startTime = Date.now();

      // Simulate continuous usage
      const simulateUsage = setInterval(() => {
        const userId = `user${Math.floor(Math.random() * 20)}`;
        const stream = manager.getStream(userId, 'ws://localhost:8080');
        const unsubscribe = stream.subscribe(() => {});

        // Randomly cleanup some streams
        if (Math.random() > 0.7) {
          manager.removeStream(userId);
        }
      }, 100);

      // Measure memory periodically
      const measureInterval = setInterval(async () => {
        if (global.gc) global.gc();
        const memory = process.memoryUsage().heapUsed;
        measurements.push(memory);

        if (Date.now() - startTime >= duration) {
          clearInterval(simulateUsage);
          clearInterval(measureInterval);
        }
      }, interval);

      // Wait for test to complete
      await new Promise(resolve => {
        setTimeout(resolve, duration + 1000);
      });

      // Analyze memory trend
      const initialMemory = measurements[0];
      const finalMemory = measurements[measurements.length - 1];
      const growth = finalMemory - initialMemory;
      const growthRate = growth / (duration / 3600000); // MB per hour

      console.log(`Memory growth over ${duration}ms: ${growth / 1024 / 1024} MB`);
      console.log(`Projected growth rate: ${growthRate / 1024 / 1024} MB/hour`);

      // Should not grow more than 10MB over 1 minute
      expect(growth).toBeLessThan(10 * 1024 * 1024);
    }, 120000); // 2 minute timeout
  });
});
```

**Heap Profiling Results**:

```bash
# Before fix (after 4 hours):
Heap Size: 650 MB
Objects: 2.4M
Event Listeners: 48,000
Growth Rate: ~50 MB/hour

# After fix (after 24 hours):
Heap Size: 125 MB (stable)
Objects: 450K
Event Listeners: 120 (bounded)
Growth Rate: ~0 MB/hour ✅
```

**Production Validation**:
- Deployed to staging for 72-hour soak test
- Memory remained stable at 120-150 MB
- No crashes observed
- GC pauses reduced to < 50ms (was 800ms)

## Phase 5: Regression Prevention

### Agent: tester + coder

**Continuous Memory Monitoring**:

```javascript
// monitoring/memory-monitor.js
const { EventEmitter } = require('events');

class MemoryMonitor extends EventEmitter {
  constructor(options = {}) {
    super();
    this.thresholdMB = options.thresholdMB || 500;
    this.checkIntervalMs = options.checkIntervalMs || 60000;
    this.measurements = [];
    this.maxMeasurements = 100;
  }

  start() {
    this.interval = setInterval(() => {
      this.checkMemory();
    }, this.checkIntervalMs);
  }

  stop() {
    clearInterval(this.interval);
  }

  checkMemory() {
    const usage = process.memoryUsage();
    const heapMB = usage.heapUsed / 1024 / 1024;

    this.measurements.push({
      timestamp: Date.now(),
      heapUsed: heapMB,
      heapTotal: usage.heapTotal / 1024 / 1024,
      external: usage.external / 1024 / 1024
    });

    // Keep only recent measurements
    if (this.measurements.length > this.maxMeasurements) {
      this.measurements.shift();
    }

    // Check for memory leak indicators
    if (heapMB > this.thresholdMB) {
      this.emit('threshold-exceeded', { heapMB, threshold: this.thresholdMB });
    }

    // Check for linear growth (leak indicator)
    if (this.measurements.length >= 10) {
      const trend = this.calculateTrend();
      if (trend.slope > 1) { // Growing at > 1 MB per measurement
        this.emit('leak-detected', {
          slope: trend.slope,
          measurements: this.measurements.slice(-10)
        });
      }
    }
  }

  calculateTrend() {
    const n = this.measurements.length;
    const x = Array.from({ length: n }, (_, i) => i);
    const y = this.measurements.map(m => m.heapUsed);

    // Linear regression
    const sumX = x.reduce((a, b) => a + b);
    const sumY = y.reduce((a, b) => a + b);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return { slope, intercept };
  }

  getReport() {
    if (this.measurements.length === 0) {
      return { status: 'no-data' };
    }

    const current = this.measurements[this.measurements.length - 1];
    const trend = this.calculateTrend();

    return {
      current: current.heapUsed,
      trend: trend.slope,
      status: trend.slope > 1 ? 'leak-suspected' : 'healthy'
    };
  }
}

// Setup monitoring
const monitor = new MemoryMonitor({ thresholdMB: 500 });

monitor.on('threshold-exceeded', ({ heapMB, threshold }) => {
  console.error(`Memory threshold exceeded: ${heapMB} MB > ${threshold} MB`);
  alerts.send({
    severity: 'high',
    message: `Application memory usage: ${heapMB} MB`,
    action: 'Investigate potential memory leak'
  });
});

monitor.on('leak-detected', ({ slope }) => {
  console.error(`Memory leak detected: growing at ${slope} MB per minute`);
  alerts.send({
    severity: 'critical',
    message: `Linear memory growth detected: ${slope} MB/min`,
    action: 'Immediate investigation required'
  });
});

monitor.start();
```

**Documentation Updates**:

```markdown
## Memory Leak Prevention Guidelines

### Mandatory Practices

1. **Event Listener Hygiene**:
   - Always remove event listeners in cleanup/destroy methods
   - Return unsubscribe functions from subscribe methods
   - Use `Set` instead of `Array` for listener tracking

2. **Resource Lifecycle**:
   - Implement explicit `destroy()` method for all resource classes
   - Call cleanup in error handlers
   - Use `try/finally` to ensure cleanup
   - Handle SIGTERM/SIGINT for graceful shutdown

3. **Bounded Collections**:
   - Set maximum sizes for Maps, Sets, Arrays
   - Implement eviction policies (LRU, FIFO)
   - Use periodic cleanup for long-lived collections
   - Consider `WeakMap`/`WeakSet` for automatic GC

4. **Connection Management**:
   - Close sockets, database connections, file handles
   - Use connection pools with max size limits
   - Implement connection timeouts
   - Clean up on errors

5. **Memory Monitoring**:
   - Run heap profiler in staging regularly
   - Set up alerts for memory thresholds
   - Track heap growth trends
   - Log large object allocations

### Code Review Checklist

- [ ] All event listeners have corresponding removal
- [ ] Classes with resources have destroy() method
- [ ] Collections have maximum size limits
- [ ] Error handlers clean up resources
- [ ] No unbounded intervals/timers
- [ ] WeakMap/WeakSet used where appropriate
- [ ] Tests verify no memory leaks

### Tools

- **Heap Profiler**: `node --inspect --expose-gc app.js`
- **Memory Analysis**: Chrome DevTools → Memory
- **Leak Detection**: `clinic.js bubbleprof`
- **Monitoring**: Custom MemoryMonitor class (see monitoring/)
```

**Automated Leak Detection**:

```javascript
// Added to CI/CD pipeline
// tests/ci-memory-check.js

/**
 * Run memory leak detection in CI
 * Fails build if memory growth detected
 */
async function ciMemoryCheck() {
  const { spawn } = require('child_process');
  const app = spawn('node', [
    '--expose-gc',
    '--max-old-space-size=512',
    'app.js'
  ]);

  const measurements = [];

  // Measure memory every 10 seconds for 2 minutes
  for (let i = 0; i < 12; i++) {
    await new Promise(resolve => setTimeout(resolve, 10000));

    // Trigger GC and measure
    process.kill(app.pid, 'SIGUSR2'); // Custom signal to trigger GC
    const memory = await getProcessMemory(app.pid);
    measurements.push(memory);
  }

  // Kill app
  app.kill();

  // Analyze trend
  const initialMemory = measurements[0];
  const finalMemory = measurements[measurements.length - 1];
  const growth = finalMemory - initialMemory;
  const growthPercent = (growth / initialMemory) * 100;

  console.log(`Memory: ${initialMemory} MB → ${finalMemory} MB`);
  console.log(`Growth: ${growth} MB (${growthPercent.toFixed(1)}%)`);

  // Fail if memory grew > 20%
  if (growthPercent > 20) {
    throw new Error(`Memory leak detected: ${growthPercent}% growth over 2 minutes`);
  }

  console.log('✅ No memory leak detected');
}

// Run in CI
if (require.main === module) {
  ciMemoryCheck().catch(error => {
    console.error(error);
    process.exit(1);
  });
}
```

## Summary

**Issue**: Memory leak causing crashes after 24-48 hours
**Root Cause**: Event listeners and connections not cleaned up
**Fix**: Explicit lifecycle management with destroy() pattern
**Tests**: Comprehensive memory leak tests + CI automation
**Impact**: Stable memory usage, no crashes, 85% reduction in GC pauses
**Time to Resolution**: 105 minutes

## Key Takeaways

1. **Use heap profiler early**: Don't wait for production crashes
2. **Retainer paths are key**: Follow references to find leak sources
3. **Explicit lifecycle**: Always implement destroy/cleanup methods
4. **Return cleanup functions**: Make cleanup easy for callers
5. **Bound all collections**: Never allow unbounded growth
6. **Automate detection**: Add memory tests to CI/CD
7. **Monitor trends**: Linear growth = leak

## Best Practices for Memory Safety

### Do
- Use Set/Map with cleanup instead of Array
- Return unsubscribe functions
- Implement destroy() for resources
- Handle errors with cleanup
- Set collection size limits
- Use WeakMap for automatic GC
- Monitor memory in production

### Don't
- Add listeners without removal
- Create unbounded intervals
- Accumulate closures
- Ignore memory warnings
- Skip cleanup in error paths
- Use global state without bounds

## Related Examples

- [Example 1: Null Pointer](example-1-null-pointer.md) - Basic null safety
- [Example 2: Race Condition](example-2-race-condition.md) - Async debugging
