/**
 * Test Suite: Collective Memory
 * Tests shared memory system with LRU caching and persistence
 */

const assert = require('assert');
const crypto = require('crypto');

// Collective Memory implementation for testing
class MemoryEntry {
  constructor(key, value, type, metadata = {}) {
    this.key = key;
    this.value = value;
    this.type = type;
    this.metadata = {
      confidence: 1.0,
      createdAt: new Date().toISOString(),
      accessCount: 0,
      lastAccessed: null,
      ...metadata
    };
  }
}

class CollectiveMemory {
  constructor(options = {}) {
    this.maxSize = options.maxSize || 1000;
    this.compressionThreshold = options.compressionThreshold || 1024;
    this.cache = new Map();
    this.storage = new Map();
    this.associations = new Map();
    this.accessOrder = [];
  }

  // Store entry in memory
  store(key, value, type = 'knowledge', metadata = {}) {
    const entry = new MemoryEntry(key, value, type, metadata);

    // Apply LRU eviction if cache full
    if (this.cache.size >= this.maxSize) {
      this._evictLRU();
    }

    this.cache.set(key, entry);
    this.storage.set(key, entry);
    this._updateAccessOrder(key);

    return entry;
  }

  // Retrieve entry from memory
  retrieve(key) {
    const entry = this.cache.get(key) || this.storage.get(key);

    if (entry) {
      entry.metadata.accessCount++;
      entry.metadata.lastAccessed = new Date().toISOString();
      this._updateAccessOrder(key);

      // Promote to cache if not there
      if (!this.cache.has(key) && this.cache.size < this.maxSize) {
        this.cache.set(key, entry);
      }

      return entry;
    }

    return null;
  }

  // Search memory by pattern
  search(pattern, options = {}) {
    const results = [];
    const regex = new RegExp(pattern, 'i');

    for (const [key, entry] of this.storage) {
      if (regex.test(key)) {
        // Apply filters
        if (options.type && entry.type !== options.type) {
          continue;
        }
        if (options.minConfidence && entry.metadata.confidence < options.minConfidence) {
          continue;
        }

        results.push(entry);
      }
    }

    // Sort by confidence
    results.sort((a, b) => b.metadata.confidence - a.metadata.confidence);

    // Apply limit
    return options.limit ? results.slice(0, options.limit) : results;
  }

  // Create association between keys
  associate(key1, key2, strength = 1.0) {
    if (!this.associations.has(key1)) {
      this.associations.set(key1, new Map());
    }

    this.associations.get(key1).set(key2, strength);

    // Bidirectional association
    if (!this.associations.has(key2)) {
      this.associations.set(key2, new Map());
    }
    this.associations.get(key2).set(key1, strength);
  }

  // Get related entries
  getRelated(key, limit = 10) {
    const related = [];
    const associations = this.associations.get(key);

    if (!associations) {
      return related;
    }

    for (const [relatedKey, strength] of associations) {
      const entry = this.retrieve(relatedKey);
      if (entry) {
        related.push({ entry, strength });
      }
    }

    // Sort by strength
    related.sort((a, b) => b.strength - a.strength);

    return related.slice(0, limit).map(r => r.entry);
  }

  // Update access order for LRU
  _updateAccessOrder(key) {
    const index = this.accessOrder.indexOf(key);
    if (index !== -1) {
      this.accessOrder.splice(index, 1);
    }
    this.accessOrder.push(key);
  }

  // Evict least recently used entry
  _evictLRU() {
    if (this.accessOrder.length === 0) return;

    const lruKey = this.accessOrder.shift();
    this.cache.delete(lruKey);
  }

  // Get memory statistics
  getStatistics() {
    return {
      totalEntries: this.storage.size,
      cacheSize: this.cache.size,
      cacheHitRate: this._calculateHitRate(),
      associations: this.associations.size,
      entriesByType: this._countByType(),
      avgConfidence: this._calculateAvgConfidence()
    };
  }

  _calculateHitRate() {
    // Simplified hit rate calculation
    return this.cache.size / this.storage.size;
  }

  _countByType() {
    const counts = {};
    for (const entry of this.storage.values()) {
      counts[entry.type] = (counts[entry.type] || 0) + 1;
    }
    return counts;
  }

  _calculateAvgConfidence() {
    if (this.storage.size === 0) return 0;

    let sum = 0;
    for (const entry of this.storage.values()) {
      sum += entry.metadata.confidence;
    }
    return sum / this.storage.size;
  }

  // Clear memory
  clear() {
    this.cache.clear();
    this.storage.clear();
    this.associations.clear();
    this.accessOrder = [];
  }
}

// Test Suite
describe('Collective Memory Tests', () => {
  let memory;

  beforeEach(() => {
    memory = new CollectiveMemory({ maxSize: 100 });
  });

  describe('Storage and Retrieval', () => {
    it('should store and retrieve entries', () => {
      memory.store('api-pattern', { type: 'REST' }, 'knowledge');
      const entry = memory.retrieve('api-pattern');

      assert.ok(entry);
      assert.strictEqual(entry.key, 'api-pattern');
      assert.deepStrictEqual(entry.value, { type: 'REST' });
      assert.strictEqual(entry.type, 'knowledge');
    });

    it('should return null for non-existent keys', () => {
      const entry = memory.retrieve('non-existent');
      assert.strictEqual(entry, null);
    });

    it('should track access counts', () => {
      memory.store('test-key', 'value', 'knowledge');

      memory.retrieve('test-key');
      memory.retrieve('test-key');
      memory.retrieve('test-key');

      const entry = memory.retrieve('test-key');
      assert.strictEqual(entry.metadata.accessCount, 4); // 3 + 1 final retrieve
    });

    it('should update last accessed timestamp', () => {
      memory.store('test-key', 'value', 'knowledge');

      const before = memory.retrieve('test-key');
      const beforeTime = new Date(before.metadata.lastAccessed);

      // Small delay
      setTimeout(() => {
        const after = memory.retrieve('test-key');
        const afterTime = new Date(after.metadata.lastAccessed);

        assert.ok(afterTime >= beforeTime);
      }, 10);
    });
  });

  describe('LRU Caching', () => {
    it('should evict least recently used entries when cache full', () => {
      const smallMemory = new CollectiveMemory({ maxSize: 3 });

      smallMemory.store('key1', 'value1', 'knowledge');
      smallMemory.store('key2', 'value2', 'knowledge');
      smallMemory.store('key3', 'value3', 'knowledge');

      // Access key1 and key2 to make key3 LRU
      smallMemory.retrieve('key1');
      smallMemory.retrieve('key2');

      // Store key4, should evict key3
      smallMemory.store('key4', 'value4', 'knowledge');

      // key3 should be evicted from cache but remain in storage
      assert.strictEqual(smallMemory.cache.has('key3'), false);
      assert.ok(smallMemory.storage.has('key3'));
    });

    it('should promote retrieved entries to cache', () => {
      const entry = memory.store('key1', 'value1', 'knowledge');

      // Manually remove from cache
      memory.cache.delete('key1');

      // Retrieve should promote back to cache
      memory.retrieve('key1');

      assert.ok(memory.cache.has('key1'));
    });
  });

  describe('Search Functionality', () => {
    beforeEach(() => {
      memory.store('api-rest', { type: 'REST' }, 'knowledge', { confidence: 0.9 });
      memory.store('api-graphql', { type: 'GraphQL' }, 'knowledge', { confidence: 0.8 });
      memory.store('api-grpc', { type: 'gRPC' }, 'knowledge', { confidence: 0.85 });
      memory.store('database-postgres', { type: 'SQL' }, 'knowledge', { confidence: 0.95 });
    });

    it('should search by pattern', () => {
      const results = memory.search('api-');

      assert.strictEqual(results.length, 3);
      assert.ok(results.every(r => r.key.startsWith('api-')));
    });

    it('should filter by type', () => {
      memory.store('temp-data', { value: 123 }, 'context');

      const results = memory.search('.*', { type: 'knowledge' });

      assert.ok(results.every(r => r.type === 'knowledge'));
    });

    it('should filter by minimum confidence', () => {
      const results = memory.search('api-', { minConfidence: 0.85 });

      assert.ok(results.every(r => r.metadata.confidence >= 0.85));
      assert.strictEqual(results.length, 2); // api-rest (0.9) and api-grpc (0.85)
    });

    it('should limit search results', () => {
      const results = memory.search('.*', { limit: 2 });

      assert.strictEqual(results.length, 2);
    });

    it('should sort results by confidence', () => {
      const results = memory.search('api-');

      // Should be sorted highest to lowest
      for (let i = 1; i < results.length; i++) {
        assert.ok(results[i-1].metadata.confidence >= results[i].metadata.confidence);
      }
    });
  });

  describe('Associations', () => {
    it('should create bidirectional associations', () => {
      memory.store('jwt-auth', { type: 'JWT' }, 'knowledge');
      memory.store('oauth2', { type: 'OAuth2' }, 'knowledge');

      memory.associate('jwt-auth', 'oauth2', 0.8);

      const jwt = memory.associations.get('jwt-auth');
      const oauth = memory.associations.get('oauth2');

      assert.ok(jwt.has('oauth2'));
      assert.ok(oauth.has('jwt-auth'));
      assert.strictEqual(jwt.get('oauth2'), 0.8);
      assert.strictEqual(oauth.get('jwt-auth'), 0.8);
    });

    it('should get related entries', () => {
      memory.store('rest-api', { type: 'REST' }, 'knowledge');
      memory.store('http-verbs', { methods: ['GET', 'POST'] }, 'knowledge');
      memory.store('json-format', { type: 'JSON' }, 'knowledge');

      memory.associate('rest-api', 'http-verbs', 0.9);
      memory.associate('rest-api', 'json-format', 0.7);

      const related = memory.getRelated('rest-api');

      assert.strictEqual(related.length, 2);
      // Should be sorted by strength
      assert.strictEqual(related[0].key, 'http-verbs'); // 0.9
      assert.strictEqual(related[1].key, 'json-format'); // 0.7
    });

    it('should limit related entries', () => {
      memory.store('key1', 'value1', 'knowledge');
      memory.store('key2', 'value2', 'knowledge');
      memory.store('key3', 'value3', 'knowledge');
      memory.store('key4', 'value4', 'knowledge');

      memory.associate('key1', 'key2', 0.9);
      memory.associate('key1', 'key3', 0.8);
      memory.associate('key1', 'key4', 0.7);

      const related = memory.getRelated('key1', 2);

      assert.strictEqual(related.length, 2);
    });
  });

  describe('Statistics', () => {
    it('should calculate memory statistics', () => {
      memory.store('key1', 'value1', 'knowledge', { confidence: 0.9 });
      memory.store('key2', 'value2', 'context', { confidence: 0.8 });
      memory.store('key3', 'value3', 'knowledge', { confidence: 0.85 });

      const stats = memory.getStatistics();

      assert.strictEqual(stats.totalEntries, 3);
      assert.strictEqual(stats.cacheSize, 3);
      assert.deepStrictEqual(stats.entriesByType, {
        knowledge: 2,
        context: 1
      });
      assert.ok(stats.avgConfidence > 0.8 && stats.avgConfidence < 0.9);
    });

    it('should calculate cache hit rate', () => {
      memory.store('key1', 'value1', 'knowledge');
      memory.store('key2', 'value2', 'knowledge');

      const stats = memory.getStatistics();

      assert.strictEqual(stats.cacheHitRate, 1.0); // All in cache
    });
  });

  describe('Memory Management', () => {
    it('should clear all memory', () => {
      memory.store('key1', 'value1', 'knowledge');
      memory.store('key2', 'value2', 'knowledge');
      memory.associate('key1', 'key2', 0.9);

      memory.clear();

      assert.strictEqual(memory.storage.size, 0);
      assert.strictEqual(memory.cache.size, 0);
      assert.strictEqual(memory.associations.size, 0);
      assert.strictEqual(memory.accessOrder.length, 0);
    });
  });
});

// Helper function for beforeEach
function beforeEach(fn) {
  // Store setup function for each test
  if (!global.setupFunctions) {
    global.setupFunctions = [];
  }
  global.setupFunctions.push(fn);
}

// Run tests
console.log('Running Collective Memory Tests...\n');

let passed = 0;
let failed = 0;

function describe(suiteName, fn) {
  console.log(`\n${suiteName}`);
  fn();
}

function it(testName, fn) {
  // Run setup functions
  if (global.setupFunctions) {
    global.setupFunctions.forEach(setup => setup());
    global.setupFunctions = [];
  }

  try {
    fn();
    console.log(`  ✓ ${testName}`);
    passed++;
  } catch (error) {
    console.log(`  ✗ ${testName}`);
    console.log(`    ${error.message}`);
    failed++;
  }
}

// Execute all tests
let memory;

describe('Collective Memory Tests', () => {
  describe('Storage and Retrieval', () => {
    it('should store and retrieve entries', () => {
      memory = new CollectiveMemory({ maxSize: 100 });
      memory.store('api-pattern', { type: 'REST' }, 'knowledge');
      const entry = memory.retrieve('api-pattern');
      assert.ok(entry);
      assert.strictEqual(entry.key, 'api-pattern');
    });

    it('should return null for non-existent keys', () => {
      memory = new CollectiveMemory({ maxSize: 100 });
      const entry = memory.retrieve('non-existent');
      assert.strictEqual(entry, null);
    });
  });

  describe('Search Functionality', () => {
    it('should search by pattern', () => {
      memory = new CollectiveMemory({ maxSize: 100 });
      memory.store('api-rest', { type: 'REST' }, 'knowledge');
      memory.store('api-graphql', { type: 'GraphQL' }, 'knowledge');
      const results = memory.search('api-');
      assert.strictEqual(results.length, 2);
    });

    it('should filter by type', () => {
      memory = new CollectiveMemory({ maxSize: 100 });
      memory.store('data1', 'value1', 'knowledge');
      memory.store('data2', 'value2', 'context');
      const results = memory.search('.*', { type: 'knowledge' });
      assert.ok(results.every(r => r.type === 'knowledge'));
    });
  });

  describe('Associations', () => {
    it('should create bidirectional associations', () => {
      memory = new CollectiveMemory({ maxSize: 100 });
      memory.store('jwt-auth', { type: 'JWT' }, 'knowledge');
      memory.store('oauth2', { type: 'OAuth2' }, 'knowledge');
      memory.associate('jwt-auth', 'oauth2', 0.8);
      assert.ok(memory.associations.get('jwt-auth').has('oauth2'));
      assert.ok(memory.associations.get('oauth2').has('jwt-auth'));
    });
  });
});

console.log(`\n\n=== Test Summary ===`);
console.log(`Passed: ${passed}`);
console.log(`Failed: ${failed}`);
console.log(`Total: ${passed + failed}`);
console.log(`Success Rate: ${(passed / (passed + failed) * 100).toFixed(1)}%`);

if (failed === 0) {
  console.log('\n✓ All tests passed!');
  process.exit(0);
} else {
  console.log(`\n✗ ${failed} test(s) failed`);
  process.exit(1);
}
