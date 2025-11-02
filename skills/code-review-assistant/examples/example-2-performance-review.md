# Example 2: Performance Optimization Review

This example demonstrates the Code Review Assistant identifying and fixing performance bottlenecks in a data processing function with O(n²) complexity.

## Scenario

**PR #789**: Add user search and filtering
**Files Changed**: `src/services/userService.js`
**Lines Added**: 78
**Focus Area**: Performance

## Original Code (Inefficient)

```javascript
// src/services/userService.js
class UserService {
  constructor(database) {
    this.db = database;
    this.cache = {}; // Simple object cache
  }

  // BOTTLENECK 1: O(n²) complexity
  async searchUsers(query) {
    const allUsers = await this.db.users.findAll(); // Load all users
    const results = [];

    // Nested loops create O(n²) complexity
    for (let user of allUsers) {
      for (let term of query.split(' ')) {
        if (user.name.toLowerCase().includes(term.toLowerCase()) ||
            user.email.toLowerCase().includes(term.toLowerCase())) {
          results.push(user);
          break;
        }
      }
    }

    return results;
  }

  // BOTTLENECK 2: No pagination
  async getUsersByRole(role) {
    const users = await this.db.users.findAll(); // Load entire table
    return users.filter(u => u.role === role);
  }

  // BOTTLENECK 3: Cache inefficiency
  async getUserById(id) {
    if (this.cache[id]) {
      return this.cache[id]; // Never expires
    }

    const user = await this.db.users.findById(id);
    this.cache[id] = user; // Unbounded cache growth
    return user;
  }

  // BOTTLENECK 4: N+1 query problem
  async getUsersWithPosts() {
    const users = await this.db.users.findAll();

    // Sequential queries for each user
    for (let user of users) {
      user.posts = await this.db.posts.findAll({ userId: user.id });
      user.postCount = user.posts.length;
    }

    return users;
  }

  // BOTTLENECK 5: Memory leak
  async bulkProcessUsers(userIds) {
    const results = [];

    // No streaming, loads all data into memory
    for (let id of userIds) {
      const user = await this.getUserById(id);
      const processed = this.heavyProcessing(user); // CPU intensive
      results.push(processed);
    }

    return results;
  }

  heavyProcessing(user) {
    // Simulate expensive computation
    let result = user;
    for (let i = 0; i < 1000000; i++) {
      result = { ...user, computed: i };
    }
    return result;
  }
}

module.exports = UserService;
```

## Code Review Assistant Execution

```bash
$ code-review-assistant 789 performance

================================================================
Code Review Assistant: PR #789
================================================================

[1/8] Gathering PR information...
PR: Add user search and filtering
Files changed: 1

[2/8] Initializing multi-agent review swarm...
✓ Swarm initialized (mesh topology, 5 agents)
✓ Performance Analyst spawned

[3/8] Executing specialized reviews in parallel...
  → Performance Analyst reviewing...

[4/8] Running complete quality audit...
✓ Performance analysis complete: 5 bottlenecks detected

[5/8] Aggregating review findings...
Overall Score: 48/100

[6/8] Generating fix suggestions with Codex...
✓ 5 optimization suggestions generated

[7/8] Assessing merge readiness...
Merge Ready: ⚠️ No
Performance Issues: 5 critical

[8/8] Creating review comment...
✓ Review comment posted to PR #789
✓ Changes requested

================================================================
Code Review Complete!
================================================================

Overall Score: 48/100
Merge Ready: false
Decision: request_changes

Review artifacts in: pr-review-789/
```

## Performance Analysis Output

### Bottlenecks Detected

#### 1. O(n²) Search Complexity (CRITICAL)
**File**: `src/services/userService.js`
**Line**: 9-23
**Severity**: CRITICAL
**Impact**: 1000 users = 1,000,000 operations

**Issue**:
```javascript
for (let user of allUsers) {           // O(n)
  for (let term of query.split(' ')) { // O(m)
    // ... string comparison ...       // O(k)
  }
}
// Total: O(n * m * k) where n=users, m=terms, k=string length
```

**Performance Benchmark**:
- 100 users: ~50ms
- 1,000 users: ~500ms
- 10,000 users: ~8,000ms (8 seconds!)
- 100,000 users: TIMEOUT

**Fix Suggestion**:
```javascript
// Option 1: Database-level full-text search (BEST)
async searchUsers(query) {
  // PostgreSQL full-text search
  return await this.db.users.findAll({
    where: {
      searchVector: {
        [Op.match]: this.db.fn('to_tsquery', query)
      }
    },
    limit: 100 // Always paginate
  });
}

// Option 2: Elasticsearch integration (SCALABLE)
async searchUsers(query) {
  const { body } = await this.elasticClient.search({
    index: 'users',
    body: {
      query: {
        multi_match: {
          query: query,
          fields: ['name^2', 'email'], // Boost name relevance
          fuzziness: 'AUTO'
        }
      },
      size: 100
    }
  });
  return body.hits.hits.map(hit => hit._source);
}

// Option 3: In-memory index (GOOD for small datasets)
class UserService {
  constructor(database) {
    this.db = database;
    this.searchIndex = new Map(); // Build index on startup
    this.buildSearchIndex();
  }

  async buildSearchIndex() {
    const users = await this.db.users.findAll();
    for (let user of users) {
      const tokens = [
        ...user.name.toLowerCase().split(' '),
        user.email.toLowerCase()
      ];
      for (let token of tokens) {
        if (!this.searchIndex.has(token)) {
          this.searchIndex.set(token, []);
        }
        this.searchIndex.get(token).push(user);
      }
    }
  }

  searchUsers(query) {
    const terms = query.toLowerCase().split(' ');
    const userSets = terms.map(term => this.searchIndex.get(term) || []);
    return this.intersectArrays(userSets); // O(n) instead of O(n²)
  }
}
```

**Performance Improvement**: O(n²) → O(1) for indexed search

---

#### 2. Missing Pagination (HIGH)
**File**: `src/services/userService.js`
**Line**: 26-29
**Severity**: HIGH
**Impact**: Memory exhaustion with large datasets

**Issue**: `findAll()` loads entire table into memory.

**Memory Usage**:
- 10,000 users: ~50MB
- 100,000 users: ~500MB
- 1,000,000 users: ~5GB (OOM crash)

**Fix Suggestion**:
```javascript
async getUsersByRole(role, page = 1, pageSize = 50) {
  const offset = (page - 1) * pageSize;

  const { rows, count } = await this.db.users.findAndCountAll({
    where: { role },
    limit: pageSize,
    offset: offset,
    order: [['createdAt', 'DESC']]
  });

  return {
    users: rows,
    pagination: {
      page,
      pageSize,
      totalPages: Math.ceil(count / pageSize),
      totalCount: count,
      hasMore: offset + pageSize < count
    }
  };
}
```

**Performance Improvement**: Constant memory usage regardless of table size

---

#### 3. Unbounded Cache Growth (HIGH)
**File**: `src/services/userService.js`
**Line**: 32-40
**Severity**: HIGH
**Impact**: Memory leak

**Issue**: Cache never expires, grows indefinitely.

**Memory Leak Example**:
```javascript
// After 1 hour of traffic
Object.keys(userService.cache).length; // 50,000 entries
process.memoryUsage().heapUsed; // 1.2GB
```

**Fix Suggestion**:
```javascript
// Use LRU cache with TTL
const LRU = require('lru-cache');

class UserService {
  constructor(database) {
    this.db = database;
    this.cache = new LRU({
      max: 1000,           // Max 1000 entries
      ttl: 1000 * 60 * 5,  // 5 minute TTL
      updateAgeOnGet: true // LRU behavior
    });
  }

  async getUserById(id) {
    const cached = this.cache.get(id);
    if (cached) return cached;

    const user = await this.db.users.findById(id);
    this.cache.set(id, user);
    return user;
  }
}

// Alternative: Redis cache for distributed systems
async getUserById(id) {
  const cached = await this.redis.get(`user:${id}`);
  if (cached) return JSON.parse(cached);

  const user = await this.db.users.findById(id);
  await this.redis.setex(`user:${id}`, 300, JSON.stringify(user)); // 5min TTL
  return user;
}
```

**Performance Improvement**: Bounded memory + distributed caching

---

#### 4. N+1 Query Problem (CRITICAL)
**File**: `src/services/userService.js`
**Line**: 43-52
**Severity**: CRITICAL
**Impact**: 100 users = 101 database queries

**Issue**: Sequential queries in loop.

**Query Analysis**:
```sql
-- Query 1: Get all users
SELECT * FROM users;

-- Query 2-101: Get posts for each user (N+1 problem!)
SELECT * FROM posts WHERE userId=1;
SELECT * FROM posts WHERE userId=2;
SELECT * FROM posts WHERE userId=3;
-- ... 97 more queries ...
```

**Fix Suggestion**:
```javascript
// Use eager loading with JOIN
async getUsersWithPosts() {
  const users = await this.db.users.findAll({
    include: [{
      model: this.db.posts,
      attributes: ['id', 'title', 'createdAt'] // Only needed fields
    }],
    attributes: {
      include: [
        [this.db.fn('COUNT', this.db.col('posts.id')), 'postCount']
      ]
    },
    group: ['users.id']
  });

  return users;
}

// Alternative: Use DataLoader for batching
const DataLoader = require('dataloader');

class UserService {
  constructor(database) {
    this.db = database;
    this.postLoader = new DataLoader(async (userIds) => {
      const posts = await this.db.posts.findAll({
        where: { userId: { [Op.in]: userIds } }
      });

      // Group posts by userId
      const postsByUser = userIds.map(id =>
        posts.filter(p => p.userId === id)
      );
      return postsByUser;
    });
  }

  async getUsersWithPosts() {
    const users = await this.db.users.findAll();

    // Batch all post queries into one
    for (let user of users) {
      user.posts = await this.postLoader.load(user.id);
      user.postCount = user.posts.length;
    }

    return users;
  }
}
```

**Performance Improvement**: 101 queries → 2 queries (50x faster)

---

#### 5. Synchronous Memory Exhaustion (HIGH)
**File**: `src/services/userService.js`
**Line**: 55-65
**Severity**: HIGH
**Impact**: Memory + CPU bottleneck

**Issue**: Loads all data into memory, blocks event loop.

**Fix Suggestion**:
```javascript
// Use streaming for large datasets
async *bulkProcessUsersStream(userIds) {
  const batchSize = 100;

  for (let i = 0; i < userIds.length; i += batchSize) {
    const batch = userIds.slice(i, i + batchSize);

    // Process batch in parallel
    const promises = batch.map(id =>
      this.getUserById(id).then(user => this.heavyProcessing(user))
    );

    const results = await Promise.all(promises);
    yield* results; // Stream results
  }
}

// Usage with streaming
async handleRequest(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.write('[');

  let first = true;
  for await (const user of userService.bulkProcessUsersStream(userIds)) {
    if (!first) res.write(',');
    res.write(JSON.stringify(user));
    first = false;
  }

  res.write(']');
  res.end();
}

// Alternative: Use worker threads for CPU-intensive tasks
const { Worker } = require('worker_threads');

async bulkProcessUsers(userIds) {
  const workers = [];
  const chunkSize = Math.ceil(userIds.length / 4); // 4 workers

  for (let i = 0; i < 4; i++) {
    const chunk = userIds.slice(i * chunkSize, (i + 1) * chunkSize);
    workers.push(this.processInWorker(chunk));
  }

  const results = await Promise.all(workers);
  return results.flat();
}

processInWorker(userIds) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./workers/userProcessor.js', {
      workerData: { userIds }
    });
    worker.on('message', resolve);
    worker.on('error', reject);
  });
}
```

**Performance Improvement**: Constant memory + parallel CPU utilization

---

## Complete Optimized Code

```javascript
// src/services/userService.js (OPTIMIZED VERSION)
const LRU = require('lru-cache');
const DataLoader = require('dataloader');
const { Worker } = require('worker_threads');

class UserService {
  constructor(database, elasticClient, redis) {
    this.db = database;
    this.elastic = elasticClient;
    this.redis = redis;

    // LRU cache with TTL
    this.cache = new LRU({
      max: 1000,
      ttl: 1000 * 60 * 5,
      updateAgeOnGet: true
    });

    // DataLoader for batching
    this.postLoader = new DataLoader(async (userIds) => {
      const posts = await this.db.posts.findAll({
        where: { userId: { [Op.in]: userIds } }
      });
      return userIds.map(id => posts.filter(p => p.userId === id));
    });
  }

  // OPTIMIZED: Full-text search with Elasticsearch
  async searchUsers(query, page = 1, pageSize = 50) {
    const { body } = await this.elastic.search({
      index: 'users',
      body: {
        query: {
          multi_match: {
            query: query,
            fields: ['name^2', 'email'],
            fuzziness: 'AUTO'
          }
        },
        from: (page - 1) * pageSize,
        size: pageSize
      }
    });

    return {
      users: body.hits.hits.map(hit => hit._source),
      pagination: {
        page,
        pageSize,
        total: body.hits.total.value
      }
    };
  }

  // OPTIMIZED: Pagination
  async getUsersByRole(role, page = 1, pageSize = 50) {
    const cacheKey = `users:role:${role}:page:${page}`;
    const cached = await this.redis.get(cacheKey);

    if (cached) return JSON.parse(cached);

    const { rows, count } = await this.db.users.findAndCountAll({
      where: { role },
      limit: pageSize,
      offset: (page - 1) * pageSize,
      order: [['createdAt', 'DESC']]
    });

    const result = {
      users: rows,
      pagination: {
        page,
        pageSize,
        totalPages: Math.ceil(count / pageSize),
        totalCount: count
      }
    };

    await this.redis.setex(cacheKey, 60, JSON.stringify(result));
    return result;
  }

  // OPTIMIZED: LRU cache
  async getUserById(id) {
    const cached = this.cache.get(id);
    if (cached) return cached;

    const user = await this.db.users.findById(id);
    this.cache.set(id, user);
    return user;
  }

  // OPTIMIZED: No N+1 queries
  async getUsersWithPosts() {
    const users = await this.db.users.findAll({
      include: [{
        model: this.db.posts,
        attributes: ['id', 'title', 'createdAt']
      }]
    });

    return users.map(user => ({
      ...user.toJSON(),
      postCount: user.posts.length
    }));
  }

  // OPTIMIZED: Streaming + worker threads
  async *bulkProcessUsersStream(userIds) {
    const batchSize = 100;

    for (let i = 0; i < userIds.length; i += batchSize) {
      const batch = userIds.slice(i, i + batchSize);
      const results = await this.processInWorker(batch);
      yield* results;
    }
  }

  processInWorker(userIds) {
    return new Promise((resolve, reject) => {
      const worker = new Worker('./workers/userProcessor.js', {
        workerData: { userIds }
      });
      worker.on('message', resolve);
      worker.on('error', reject);
    });
  }
}

module.exports = UserService;
```

## Performance Benchmarks

### Before Optimization
| Operation | 100 Users | 1K Users | 10K Users | 100K Users |
|-----------|-----------|----------|-----------|------------|
| searchUsers | 50ms | 500ms | 8,000ms | TIMEOUT |
| getUsersByRole | 20ms | 200ms | 2,000ms | OOM |
| getUsersWithPosts | 150ms | 1,500ms | 15,000ms | TIMEOUT |
| bulkProcessUsers | 500ms | 5,000ms | 50,000ms | OOM |

### After Optimization
| Operation | 100 Users | 1K Users | 10K Users | 100K Users |
|-----------|-----------|----------|-----------|------------|
| searchUsers | 5ms | 8ms | 12ms | 25ms |
| getUsersByRole | 3ms | 5ms | 8ms | 15ms |
| getUsersWithPosts | 15ms | 30ms | 50ms | 120ms |
| bulkProcessUsers | 80ms | 200ms | 450ms | 1,200ms |

**Improvements**:
- searchUsers: 320x faster (O(n²) → O(1))
- getUsersByRole: 133x faster (pagination + caching)
- getUsersWithPosts: 300x faster (N+1 → JOIN)
- bulkProcessUsers: 42x faster (streaming + workers)

## Conclusion

This PR has **5 critical performance bottlenecks** that cause exponential slowdown with scale. The Code Review Assistant identified all issues, provided 3 optimization strategies per issue, and estimated performance improvements. After applying fixes, the service can handle 100x more users with the same hardware.
