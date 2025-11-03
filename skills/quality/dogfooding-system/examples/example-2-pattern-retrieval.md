# Example 2: Pattern Retrieval Phase - Memory-MCP Vector Search

## Scenario Overview

**Context**: You've completed Phase 1 and discovered a **God Object** violation in `memory-manager.js` (34 methods, threshold: 15). You need proven refactoring patterns to fix it efficiently.

**Goal**: Use Phase 2 (Pattern Retrieval) to search Memory-MCP for similar God Object fixes that worked in the past, then optionally apply the best pattern.

**Duration**: 10-30 seconds
**Agents**: `code-analyzer`, `coder`

---

## Step-by-Step Walkthrough

### Step 1: Trigger Pattern Retrieval

**User Request**:
```
"Find fixes for God Object pattern in memory-manager.js"
```

**Auto-Trigger Detection**:
- Keywords: "find fixes", "God Object", "pattern"
- Previous context: Phase 1 analysis stored in Memory-MCP
- Skill invoked: `sop-dogfooding-pattern-retrieval`

### Step 2: Agent Spawning (Automatic)

**Behind the Scenes**:
```javascript
// Skill spawns 2 agents concurrently via Task tool
[Single Message - Parallel Execution]:
  Task("Pattern Search Analyst",
    "Search Memory-MCP for past God Object refactorings. Query: 'God Object fix decomposition'. Retrieve top 5 patterns with success metrics.",
    "code-analyzer")

  Task("Pattern Application Developer",
    "Review retrieved patterns. Rank by similarity to current violation (34 methods → target: 5-6 classes). Optionally apply best pattern.",
    "coder")
```

### Step 3: Vector Search Execution

**Pattern Search Analyst Actions**:
1. Constructs semantic query from violation data
2. Calls `mcp__memory-mcp__vector_search` with mode-aware context
3. Retrieves top 5 most similar patterns (384-dim embeddings, HNSW index)

**Search Query Construction**:
```javascript
const query = {
  text: "God Object refactoring: decompose 34-method class into specialized cohesive classes using Single Responsibility Principle",
  metadata_filters: {
    intent: "refactor",
    violation_type: "God Object",
    success: true
  },
  mode: "execution", // Planning/Execution/Brainstorming
  limit: 5
};
```

**Vector Search Results** (Ranked by Cosine Similarity):
```json
{
  "results": [
    {
      "similarity": 0.94,
      "content": "Refactored APIController (28 methods) → 4 specialized classes: AuthController, UserController, DataController, ValidationService. Reduced coupling by 67%, increased test coverage to 94%.",
      "metadata": {
        "agent": "coder",
        "timestamp": "2025-10-15T09:12:34Z",
        "project": "rest-api-refactor",
        "intent": "refactor",
        "violation_type": "God Object",
        "before_methods": 28,
        "after_classes": 4,
        "coupling_reduction": 67,
        "test_coverage_improvement": 18,
        "success": true,
        "fix_pattern": "domain-based-decomposition"
      }
    },
    {
      "similarity": 0.91,
      "content": "Split DatabaseManager (41 methods) → 6 classes: ConnectionPool, QueryBuilder, TransactionManager, CacheLayer, MigrationRunner, HealthMonitor. Applied Interface Segregation Principle.",
      "metadata": {
        "agent": "backend-dev",
        "timestamp": "2025-09-28T14:45:12Z",
        "project": "database-layer-refactor",
        "intent": "refactor",
        "violation_type": "God Object",
        "before_methods": 41,
        "after_classes": 6,
        "fix_pattern": "responsibility-based-decomposition",
        "performance_improvement": 23,
        "success": true
      }
    },
    {
      "similarity": 0.88,
      "content": "Decomposed OrderProcessor (33 methods) → 5 services: PaymentService, InventoryService, ShippingService, NotificationService, AuditLogger. Used Event-Driven Architecture for decoupling.",
      "metadata": {
        "agent": "system-architect",
        "timestamp": "2025-10-01T11:23:45Z",
        "project": "ecommerce-backend",
        "intent": "refactor",
        "violation_type": "God Object",
        "before_methods": 33,
        "after_classes": 5,
        "fix_pattern": "service-oriented-decomposition",
        "event_driven": true,
        "success": true
      }
    },
    {
      "similarity": 0.82,
      "content": "Extracted FileManager (19 methods) → 3 classes: FileReader, FileWriter, FileValidator. Simple separation by I/O operations.",
      "metadata": {
        "agent": "coder",
        "timestamp": "2025-08-12T16:34:21Z",
        "project": "file-processing",
        "intent": "refactor",
        "violation_type": "God Object",
        "before_methods": 19,
        "after_classes": 3,
        "fix_pattern": "io-based-decomposition",
        "success": true
      }
    },
    {
      "similarity": 0.79,
      "content": "Refactored ConfigManager (22 methods) → ConfigLoader, ConfigValidator, ConfigCache, ConfigWatcher. Applied Observer pattern for config changes.",
      "metadata": {
        "agent": "code-analyzer",
        "timestamp": "2025-09-05T10:11:09Z",
        "project": "configuration-system",
        "intent": "refactor",
        "violation_type": "God Object",
        "before_methods": 22,
        "after_classes": 4,
        "fix_pattern": "lifecycle-based-decomposition",
        "pattern_used": "Observer",
        "success": true
      }
    }
  ],
  "search_time": "0.008s",
  "total_matches": 47,
  "mode": "execution"
}
```

### Step 4: Pattern Ranking & Analysis

**Pattern Application Developer Actions**:
1. Receives 5 patterns from analyst
2. Ranks by similarity to current violation (34 methods)
3. Analyzes fix strategies and success metrics
4. Recommends best pattern

**Ranking Report**:
```markdown
## Pattern Ranking for MemoryManager God Object (34 methods)

### 🥇 BEST MATCH: Result #3 (Similarity: 0.88)
**Pattern**: Service-Oriented Decomposition
**Source**: OrderProcessor refactoring (33 methods → 5 services)
**Why Best**:
- Nearly identical method count (33 vs 34)
- Event-driven architecture fits memory operations
- Clear service boundaries (similar to memory operations)

**Recommended Decomposition for MemoryManager**:
1. **StorageService** (10 methods)
   - `storeMemory`, `bulkStore`, `updateMemory`, `deleteMemory`, `clearExpired`
   - `backup`, `restore`, `migrate`, `shard`, `merge`

2. **RetrievalService** (8 methods)
   - `retrieveMemory`, `searchSimilar`, `bulkRetrieve`, `exportMemories`
   - `importMemories`, `validateMemory`, `optimizeIndex`, `rebuildIndex`

3. **SecurityService** (6 methods)
   - `sanitize`, `encrypt`, `decrypt`, `compress`, `decompress`
   - `validateMemory`

4. **MonitoringService** (6 methods)
   - `getStats`, `healthCheck`, `benchmark`, `profile`, `monitor`, `alert`

5. **VectorService** (4 methods)
   - `vectorStore`, `initialize`, `log`, `trace`

**Expected Benefits**:
- 67% coupling reduction (based on similar refactor)
- Easier testing (mock services independently)
- Clear separation of concerns
- Event-driven coordination between services

---

### 🥈 SECOND CHOICE: Result #2 (Similarity: 0.91)
**Pattern**: Responsibility-Based Decomposition
**Source**: DatabaseManager (41 methods → 6 classes)
**Why Good**: Higher similarity score, more granular separation
**Caveat**: 6 classes might be over-engineering for 34 methods

---

### 🥉 THIRD CHOICE: Result #1 (Similarity: 0.94)
**Pattern**: Domain-Based Decomposition
**Source**: APIController (28 methods → 4 classes)
**Why Considered**: Highest similarity, good test coverage improvement
**Caveat**: Fewer methods (28 vs 34), domain-based split less clear for memory ops
```

### Step 5: Optional Pattern Application

**User Decision Point**:
```
Do you want to:
A) Apply the recommended pattern automatically (coder agent)
B) Generate refactoring plan only (manual implementation)
C) See code examples for all 3 patterns
```

**If User Chooses A (Auto-Apply)**:

**Coder Agent Actions**:
1. Reads `memory-manager.js`
2. Applies Service-Oriented Decomposition pattern
3. Generates 5 new service files
4. Updates imports and dependency injection
5. Creates facade for backward compatibility

---

## Code Example: Pattern Application

### Before: God Object (34 methods)
```javascript
// memory-manager.js (982 lines, 34 methods)
class MemoryManager {
  constructor(config) {
    this.db = null;
    this.vectorIndex = null;
    this.cache = new Map();
    this.encryptionKey = config.key;
    this.monitoringEnabled = config.monitoring;
    // ... 30+ instance variables
  }

  // ALL 34 methods in one class
  async storeMemory(content, tags, metadata, ...) { /* ... */ }
  async retrieveMemory(query) { /* ... */ }
  async searchSimilar(query, options) { /* ... */ }
  async updateMemory(id, updates) { /* ... */ }
  async deleteMemory(id) { /* ... */ }
  async bulkStore(memories) { /* ... */ }
  async bulkRetrieve(ids) { /* ... */ }
  async clearExpired() { /* ... */ }
  async getStats() { /* ... */ }
  async exportMemories(format) { /* ... */ }
  async importMemories(data) { /* ... */ }
  async backup(destination) { /* ... */ }
  async restore(source) { /* ... */ }
  async optimizeIndex() { /* ... */ }
  async rebuildIndex() { /* ... */ }
  async validateMemory(memory) { /* ... */ }
  async sanitize(content) { /* ... */ }
  async encrypt(data) { /* ... */ }
  async decrypt(data) { /* ... */ }
  async compress(data) { /* ... */ }
  async decompress(data) { /* ... */ }
  async shard(memories, count) { /* ... */ }
  async merge(shards) { /* ... */ }
  async migrate(fromVersion, toVersion) { /* ... */ }
  async healthCheck() { /* ... */ }
  async benchmark(operations) { /* ... */ }
  async profile(duration) { /* ... */ }
  async monitor(interval) { /* ... */ }
  async alert(condition) { /* ... */ }
  async log(message) { /* ... */ }
  async trace(operation) { /* ... */ }
  async vectorStore(embedding) { /* ... */ }
  async initialize(options) { /* ... */ }
}
```

### After: Service-Oriented Decomposition (5 services)

**1. StorageService.js** (10 methods, 215 lines):
```javascript
// services/StorageService.js
class StorageService {
  constructor(db, eventBus) {
    this.db = db;
    this.eventBus = eventBus;
  }

  async storeMemory(content, tags, metadata, options) {
    const memory = await this.db.insert({ content, tags, metadata });
    this.eventBus.emit('memory.stored', { id: memory.id });
    return memory;
  }

  async bulkStore(memories) {
    const results = await this.db.batchInsert(memories);
    this.eventBus.emit('memory.bulk-stored', { count: results.length });
    return results;
  }

  async updateMemory(id, updates) {
    const memory = await this.db.update(id, updates);
    this.eventBus.emit('memory.updated', { id });
    return memory;
  }

  async deleteMemory(id) {
    await this.db.delete(id);
    this.eventBus.emit('memory.deleted', { id });
  }

  async clearExpired() {
    const count = await this.db.deleteExpired();
    this.eventBus.emit('memory.expired-cleared', { count });
    return count;
  }

  async backup(destination) { /* ... */ }
  async restore(source) { /* ... */ }
  async migrate(fromVersion, toVersion) { /* ... */ }
  async shard(memories, count) { /* ... */ }
  async merge(shards) { /* ... */ }
}
```

**2. RetrievalService.js** (8 methods, 187 lines):
```javascript
// services/RetrievalService.js
class RetrievalService {
  constructor(db, vectorIndex, eventBus) {
    this.db = db;
    this.vectorIndex = vectorIndex;
    this.eventBus = eventBus;
  }

  async retrieveMemory(query) {
    const memory = await this.db.findOne(query);
    this.eventBus.emit('memory.retrieved', { query });
    return memory;
  }

  async searchSimilar(query, options) {
    const results = await this.vectorIndex.search(query, options);
    this.eventBus.emit('memory.searched', { query, count: results.length });
    return results;
  }

  async bulkRetrieve(ids) {
    return await this.db.findMany(ids);
  }

  async exportMemories(format) { /* ... */ }
  async importMemories(data) { /* ... */ }
  async validateMemory(memory) { /* ... */ }
  async optimizeIndex() { /* ... */ }
  async rebuildIndex() { /* ... */ }
}
```

**3. SecurityService.js** (6 methods, 142 lines):
```javascript
// services/SecurityService.js
class SecurityService {
  constructor(encryptionKey) {
    this.cipher = new AES256(encryptionKey);
    this.compressor = new Brotli();
  }

  async sanitize(content) {
    return DOMPurify.sanitize(content);
  }

  async encrypt(data) {
    return this.cipher.encrypt(JSON.stringify(data));
  }

  async decrypt(encryptedData) {
    const decrypted = this.cipher.decrypt(encryptedData);
    return JSON.parse(decrypted);
  }

  async compress(data) {
    return this.compressor.compress(Buffer.from(JSON.stringify(data)));
  }

  async decompress(compressedData) {
    const decompressed = this.compressor.decompress(compressedData);
    return JSON.parse(decompressed.toString());
  }

  async validateMemory(memory) {
    const schema = Joi.object({ content: Joi.string().required(), /* ... */ });
    return schema.validate(memory);
  }
}
```

**4. MonitoringService.js** (6 methods, 156 lines):
```javascript
// services/MonitoringService.js
class MonitoringService {
  constructor(metricsCollector, alertManager) {
    this.metrics = metricsCollector;
    this.alerts = alertManager;
  }

  async getStats() {
    return {
      total: await this.metrics.get('memory.total'),
      active: await this.metrics.get('memory.active'),
      expired: await this.metrics.get('memory.expired'),
      avgResponseTime: await this.metrics.get('memory.avg-response-time')
    };
  }

  async healthCheck() {
    const checks = {
      db: await this.checkDatabase(),
      vector: await this.checkVectorIndex(),
      cache: await this.checkCache()
    };
    return { healthy: Object.values(checks).every(c => c), checks };
  }

  async benchmark(operations) { /* ... */ }
  async profile(duration) { /* ... */ }
  async monitor(interval) { /* ... */ }
  async alert(condition) { /* ... */ }
}
```

**5. VectorService.js** (4 methods, 98 lines):
```javascript
// services/VectorService.js
class VectorService {
  constructor(config) {
    this.index = null;
    this.config = config;
    this.logger = new Logger('VectorService');
  }

  async initialize(options) {
    this.index = await ChromaDB.connect(options);
    this.logger.info('VectorService initialized');
  }

  async vectorStore(embedding) {
    return await this.index.insert(embedding);
  }

  async log(message) {
    this.logger.log(message);
  }

  async trace(operation) {
    this.logger.trace(operation);
  }
}
```

**6. MemoryManager.js (Facade - Backward Compatibility)** (152 lines):
```javascript
// memory-manager.js (now a lightweight facade)
class MemoryManager {
  constructor(config) {
    const eventBus = new EventEmitter();
    const db = new Database(config.db);
    const vectorIndex = new ChromaDB(config.vector);

    this.storage = new StorageService(db, eventBus);
    this.retrieval = new RetrievalService(db, vectorIndex, eventBus);
    this.security = new SecurityService(config.encryptionKey);
    this.monitoring = new MonitoringService(config.metrics, config.alerts);
    this.vector = new VectorService(config.vector);

    this.eventBus = eventBus;
  }

  // Delegate to services (maintains API compatibility)
  async storeMemory(...args) { return this.storage.storeMemory(...args); }
  async retrieveMemory(...args) { return this.retrieval.retrieveMemory(...args); }
  async searchSimilar(...args) { return this.retrieval.searchSimilar(...args); }
  async encrypt(...args) { return this.security.encrypt(...args); }
  async getStats(...args) { return this.monitoring.getStats(...args); }
  // ... 29 more delegations
}
```

---

## Outcomes & Results

### Quantitative Metrics

**Before Refactoring**:
- Lines of Code: 982 (single file)
- Methods: 34 (single class)
- Cyclomatic Complexity: 156 (aggregate)
- Test Coverage: 72%
- Coupling: High (all methods share 30+ instance variables)

**After Refactoring**:
- Lines of Code: 950 total (6 files: 215+187+142+156+98+152)
- Average Methods per Class: 6.8 (34 ÷ 5 services)
- Cyclomatic Complexity: 89 (43% reduction)
- Test Coverage: 91% (+19%)
- Coupling: 67% reduction (services isolated)

**Memory-MCP Performance**:
- Vector search time: 0.008s (HNSW index)
- Patterns retrieved: 5 (from 47 total matches)
- Similarity scores: 0.79–0.94 (high relevance)
- Automatic ranking: Best pattern identified in <1s

### Qualitative Insights

✅ **Service boundaries are clear** - Each service has single responsibility
✅ **Event-driven coordination** - Services communicate via EventBus (decoupled)
✅ **Testability improved** - Mock services independently
✅ **Backward compatibility** - Facade maintains existing API
✅ **Pattern reuse successful** - 88% similarity to proven fix

---

## Tips & Best Practices

### ✅ DO:

1. **Search before implementing** - Memory-MCP has 47+ God Object fixes
2. **Filter by success metrics** - Only retrieve `success: true` patterns
3. **Rank by similarity** - Closest method count = best pattern match
4. **Consider context** - Event-driven works for memory, not all domains
5. **Apply incrementally** - Refactor one service at a time, test each

### ❌ DON'T:

1. **Copy patterns blindly** - Adapt to your domain (memory ≠ API ≠ database)
2. **Ignore similarity scores** - <0.7 similarity = different problem
3. **Over-decompose** - 6 classes for 19 methods is overkill
4. **Skip facade pattern** - Breaking existing API causes downstream failures
5. **Forget to store results** - Save successful refactor for future retrieval

### Advanced Techniques

**1. Multi-Pattern Search**:
```javascript
// Search for multiple violation types simultaneously
const queries = [
  "God Object fix decomposition",
  "Parameter Bomb fix options object",
  "Deep Nesting fix early returns"
];

const allPatterns = await Promise.all(
  queries.map(q => mcp__memory-mcp__vector_search({ text: q, limit: 3 }))
);
```

**2. Context-Aware Filtering**:
```javascript
// Filter by project type, language, framework
const patterns = await vector_search({
  text: "God Object refactoring",
  metadata_filters: {
    language: "JavaScript",
    framework: "Node.js",
    project_type: "backend",
    success: true,
    test_coverage_improvement: { $gte: 10 } // 10%+ improvement
  }
});
```

**3. Ensemble Ranking**:
```javascript
// Combine similarity score + success metrics for ranking
const rankedPatterns = patterns.map(p => ({
  ...p,
  score: (
    p.similarity * 0.4 +
    (p.metadata.test_coverage_improvement / 100) * 0.3 +
    (p.metadata.coupling_reduction / 100) * 0.3
  )
})).sort((a, b) => b.score - a.score);
```

### Common Pitfalls

**Pitfall 1: Ignoring Domain Differences**
- **Problem**: Applying API controller pattern to database manager
- **Solution**: Check domain similarity in metadata (backend vs frontend vs data)

**Pitfall 2: Low Similarity Threshold**
- **Problem**: Using 0.65 similarity pattern for critical refactor
- **Solution**: Require ≥0.80 similarity for auto-apply, manual review for <0.80

**Pitfall 3: Missing Event Bus Setup**
- **Problem**: Event-driven pattern requires EventEmitter infrastructure
- **Solution**: Check pattern dependencies before applying

---

## Next Steps

After completing Phase 2 (Pattern Retrieval):

1. **If auto-applied** → Proceed to Phase 3 (Continuous Improvement) for validation
2. **If manual** → Implement refactoring, then store success in Memory-MCP
3. **If uncertain** → Request more patterns or manual review

**Command to Trigger Phase 3**:
```
"Run improvement cycle on refactored memory-manager.js"
```

---

## Summary

**Phase 2 Pattern Retrieval** provides:
- ✅ Semantic vector search across 47+ proven patterns (0.008s)
- ✅ Automatic ranking by similarity + success metrics
- ✅ Context-aware filtering (language, framework, domain)
- ✅ Optional auto-application via coder agent
- ✅ 88% similarity match for 34-method God Object
- ✅ 67% coupling reduction + 19% test coverage improvement (predicted)

**Integration**: Works with any Memory-MCP stored pattern, requires Phase 1 analysis as input

**ROI**: 10-30 seconds to find proven fix vs 2-4 hours of trial-and-error refactoring
