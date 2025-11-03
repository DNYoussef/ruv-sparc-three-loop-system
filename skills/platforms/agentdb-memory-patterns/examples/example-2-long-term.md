# Example 2: Long-Term Memory (Persistent Knowledge)

## Overview

Long-term memory stores important facts, learned patterns, and persistent knowledge that should survive across sessions. This pattern is essential for building intelligent agents that remember user preferences, domain knowledge, and successful strategies.

**Characteristics**:
- **Capacity**: Unlimited (with consolidation)
- **Retention**: Permanent until explicitly deleted
- **Access Pattern**: Semantic search and importance ranking
- **Use Case**: User preferences, domain expertise, learned behaviors

## Implementation

### Basic Long-Term Memory

```typescript
import { createAgentDBAdapter } from 'agentic-flow/reasoningbank';

class LongTermMemory {
  private adapter: any;

  async initialize() {
    this.adapter = await createAgentDBAdapter({
      dbPath: '.agentdb/long-term.db',
      enableLearning: true,       // Enable pattern learning
      enableReasoning: true,       // Enable context synthesis
      quantizationType: 'scalar',  // 4x memory reduction
      cacheSize: 1000              // Cache frequent patterns
    });
  }

  async storeFact(
    category: string,
    key: string,
    value: any,
    metadata: {
      confidence?: number;
      source?: string;
      tags?: string[];
    } = {}
  ) {
    const content = JSON.stringify({ key, value, metadata });
    const embedding = await this.computeEmbedding(content);

    return await this.adapter.insertPattern({
      id: '',
      type: 'fact',
      domain: category,
      pattern_data: JSON.stringify({
        embedding,
        pattern: {
          key,
          value,
          category,
          ...metadata,
          timestamp: Date.now()
        }
      }),
      confidence: metadata.confidence || 0.9,
      usage_count: 1,
      success_count: 1,
      created_at: Date.now(),
      last_used: Date.now()
    });
  }

  async getFact(category: string, key: string): Promise<any> {
    const query = `${category} ${key}`;
    const queryEmbedding = await this.computeEmbedding(query);

    const results = await this.adapter.searchPatterns(queryEmbedding, {
      domain: category,
      k: 1,
      threshold: 0.9
    });

    if (results.length > 0) {
      const data = JSON.parse(results[0].pattern_data);
      return data.pattern.value;
    }

    return null;
  }

  async getFacts(category: string, limit: number = 100): Promise<Array<any>> {
    const results = await this.adapter.searchPatterns(null, {
      domain: category,
      k: limit,
      orderBy: 'confidence DESC, usage_count DESC'
    });

    return results.map(r => {
      const data = JSON.parse(r.pattern_data);
      return data.pattern;
    });
  }

  async searchKnowledge(query: string, limit: number = 10) {
    const queryEmbedding = await this.computeEmbedding(query);

    return await this.adapter.retrieveWithReasoning(queryEmbedding, {
      k: limit,
      useMMR: true,              // Maximal Marginal Relevance
      synthesizeContext: true,    // Generate rich context
      optimizeMemory: true        // Consolidate similar patterns
    });
  }

  async updateFact(category: string, key: string, newValue: any) {
    // Store new version
    await this.storeFact(category, key, newValue, {
      confidence: 1.0,
      source: 'update'
    });

    // Optional: Mark old version as outdated
    // (AgentDB will consolidate during optimization)
  }

  async incrementUsage(patternId: string) {
    await this.adapter.updatePattern(patternId, {
      usage_count: '+1',
      last_used: Date.now()
    });
  }

  private async computeEmbedding(text: string): Promise<number[]> {
    // Use your embedding model
    return new Array(384).fill(0).map(() => Math.random());
  }

  async consolidate(options: {
    strategy?: 'importance' | 'recency' | 'hybrid';
    minConfidence?: number;
    maxPatterns?: number;
  } = {}) {
    const {
      strategy = 'importance',
      minConfidence = 0.5,
      maxPatterns = 10000
    } = options;

    await this.adapter.consolidateMemory({
      strategy,
      minConfidence,
      maxPatterns,
      removeDuplicates: true,
      mergeThreshold: 0.95 // Merge very similar patterns
    });
  }
}
```

## Usage Example: User Preferences

```typescript
class UserPreferences extends LongTermMemory {
  async setPreference(userId: string, key: string, value: any) {
    return await this.storeFact(
      `user-prefs-${userId}`,
      key,
      value,
      { confidence: 1.0, source: 'explicit' }
    );
  }

  async getPreference(userId: string, key: string): Promise<any> {
    return await this.getFact(`user-prefs-${userId}`, key);
  }

  async getAllPreferences(userId: string): Promise<Record<string, any>> {
    const facts = await this.getFacts(`user-prefs-${userId}`);

    const prefs: Record<string, any> = {};
    facts.forEach(f => {
      prefs[f.key] = f.value;
    });

    return prefs;
  }

  async inferPreference(userId: string, context: string): Promise<any> {
    // Use reasoning to infer preference from similar contexts
    const results = await this.searchKnowledge(
      `user ${userId} preference ${context}`
    );

    return results.synthesizedContext;
  }
}

// Usage
const prefs = new UserPreferences();
await prefs.initialize();

// Store explicit preferences
await prefs.setPreference('user-123', 'language', 'English');
await prefs.setPreference('user-123', 'theme', 'dark');
await prefs.setPreference('user-123', 'timezone', 'UTC-5');
await prefs.setPreference('user-123', 'notifications', { email: true, sms: false });

// Retrieve specific preference
const lang = await prefs.getPreference('user-123', 'language');
console.log('Preferred language:', lang); // 'English'

// Get all preferences
const allPrefs = await prefs.getAllPreferences('user-123');
console.log('All preferences:', allPrefs);

// Infer preference from context
const inferred = await prefs.inferPreference('user-123', 'contact method');
// Might infer email based on notification preferences
```

## Usage Example: Domain Knowledge Base

```typescript
class KnowledgeBase extends LongTermMemory {
  async addKnowledge(
    domain: string,
    topic: string,
    content: string,
    source: string,
    tags: string[] = []
  ) {
    return await this.storeFact(
      `knowledge-${domain}`,
      topic,
      content,
      { confidence: 0.85, source, tags }
    );
  }

  async queryKnowledge(domain: string, query: string, limit: number = 5) {
    const queryEmbedding = await this.computeEmbedding(query);

    return await this.adapter.retrieveWithReasoning(queryEmbedding, {
      domain: `knowledge-${domain}`,
      k: limit,
      useMMR: true,
      synthesizeContext: true
    });
  }

  async relatedTopics(domain: string, topic: string, limit: number = 10) {
    const fact = await this.getFact(`knowledge-${domain}`, topic);
    if (!fact) return [];

    const embedding = await this.computeEmbedding(fact);

    return await this.adapter.searchPatterns(embedding, {
      domain: `knowledge-${domain}`,
      k: limit + 1, // +1 to exclude self
      threshold: 0.7
    });
  }

  async buildConceptMap(domain: string): Promise<Map<string, string[]>> {
    const allFacts = await this.getFacts(`knowledge-${domain}`, 1000);
    const conceptMap = new Map<string, string[]>();

    for (const fact of allFacts) {
      const related = await this.relatedTopics(domain, fact.key, 5);
      conceptMap.set(
        fact.key,
        related.slice(1).map(r => JSON.parse(r.pattern_data).pattern.key)
      );
    }

    return conceptMap;
  }
}

// Usage
const kb = new KnowledgeBase();
await kb.initialize();

// Build software engineering knowledge base
await kb.addKnowledge(
  'software-engineering',
  'SOLID Principles',
  'Five design principles for object-oriented programming...',
  'Design Patterns Book',
  ['design', 'oop', 'architecture']
);

await kb.addKnowledge(
  'software-engineering',
  'DRY Principle',
  'Don\'t Repeat Yourself - reduce duplication...',
  'The Pragmatic Programmer',
  ['design', 'best-practices']
);

await kb.addKnowledge(
  'software-engineering',
  'TDD',
  'Test-Driven Development - write tests first...',
  'Kent Beck',
  ['testing', 'methodology']
);

// Query knowledge
const results = await kb.queryKnowledge(
  'software-engineering',
  'best practices for code organization'
);
console.log('Relevant knowledge:', results.synthesizedContext);

// Find related topics
const related = await kb.relatedTopics('software-engineering', 'SOLID Principles');
console.log('Related concepts:', related);

// Build concept map
const conceptMap = await kb.buildConceptMap('software-engineering');
console.log('Knowledge graph:', conceptMap);
```

## Usage Example: Learned Behavioral Patterns

```typescript
class BehavioralMemory extends LongTermMemory {
  async recordSuccess(
    context: string,
    action: string,
    result: string,
    metrics: { score: number; time: number }
  ) {
    return await this.storeFact(
      'successful-patterns',
      `${context}-${action}`,
      {
        context,
        action,
        result,
        metrics,
        timestamp: Date.now()
      },
      {
        confidence: metrics.score,
        source: 'experience',
        tags: ['success']
      }
    );
  }

  async recordFailure(
    context: string,
    action: string,
    error: string
  ) {
    return await this.storeFact(
      'failed-patterns',
      `${context}-${action}`,
      {
        context,
        action,
        error,
        timestamp: Date.now()
      },
      {
        confidence: 0.1, // Low confidence = avoid
        source: 'experience',
        tags: ['failure']
      }
    );
  }

  async getBestAction(context: string, options: string[]): Promise<string> {
    const results = await Promise.all(
      options.map(async action => {
        const pattern = await this.getFact('successful-patterns', `${context}-${action}`);
        return {
          action,
          score: pattern?.metrics?.score || 0,
          found: !!pattern
        };
      })
    );

    // Sort by success score
    results.sort((a, b) => b.score - a.score);

    // Avoid known failures
    for (const result of results) {
      const failure = await this.getFact('failed-patterns', `${context}-${result.action}`);
      if (!failure) {
        return result.action;
      }
    }

    // Fallback to highest score (even if failed before)
    return results[0].action;
  }

  async getSuccessRate(context: string): Promise<number> {
    const successes = await this.getFacts('successful-patterns');
    const failures = await this.getFacts('failed-patterns');

    const contextSuccesses = successes.filter(s => s.value.context === context);
    const contextFailures = failures.filter(f => f.value.context === context);

    const total = contextSuccesses.length + contextFailures.length;
    return total > 0 ? contextSuccesses.length / total : 0;
  }
}

// Usage
const behavior = new BehavioralMemory();
await behavior.initialize();

// Learn from experiences
await behavior.recordSuccess(
  'api-error-500',
  'retry-with-backoff',
  'Success after 2 retries',
  { score: 0.95, time: 2000 }
);

await behavior.recordFailure(
  'api-error-500',
  'immediate-retry',
  'Failed again immediately'
);

await behavior.recordSuccess(
  'database-timeout',
  'increase-timeout',
  'Query completed',
  { score: 0.90, time: 5000 }
);

// Apply learned behavior
const bestAction = await behavior.getBestAction(
  'api-error-500',
  ['immediate-retry', 'retry-with-backoff', 'give-up']
);
console.log('Best action:', bestAction); // 'retry-with-backoff'

// Analyze success rate
const successRate = await behavior.getSuccessRate('api-error-500');
console.log('Success rate:', successRate); // 0.5 (1 success, 1 failure)
```

## Memory Consolidation

Periodically consolidate long-term memory to optimize storage:

```typescript
// Run consolidation weekly
async function weeklyMaintenance() {
  const memory = new LongTermMemory();
  await memory.initialize();

  await memory.consolidate({
    strategy: 'hybrid',      // Balance importance and recency
    minConfidence: 0.3,      // Remove low-confidence patterns
    maxPatterns: 50000       // Keep top 50k patterns
  });

  console.log('Memory consolidation complete');
}

// Or automatic consolidation after N inserts
let insertCount = 0;
const CONSOLIDATION_INTERVAL = 10000;

async function storeWithAutoConsolidate(memory: LongTermMemory, ...args) {
  await memory.storeFact(...args);
  insertCount++;

  if (insertCount % CONSOLIDATION_INTERVAL === 0) {
    await memory.consolidate();
    console.log('Auto-consolidation triggered');
  }
}
```

## Performance Optimization

```typescript
// Benchmark long-term memory
const benchmark = async () => {
  const memory = new LongTermMemory();
  await memory.initialize();

  // Store 10,000 facts (batch)
  const start = Date.now();
  const promises = [];
  for (let i = 0; i < 10000; i++) {
    promises.push(
      memory.storeFact('benchmark', `key-${i}`, `value-${i}`)
    );

    if (promises.length === 100) {
      await Promise.all(promises);
      promises.length = 0;
    }
  }
  console.log(`Store 10k facts: ${Date.now() - start}ms`); // ~200ms

  // Semantic search
  const searchStart = Date.now();
  const results = await memory.searchKnowledge('key-5000', 10);
  console.log(`Search: ${Date.now() - searchStart}ms`); // <1ms (cached)

  // Consolidation
  const consStart = Date.now();
  await memory.consolidate({ maxPatterns: 5000 });
  console.log(`Consolidate: ${Date.now() - consStart}ms`); // ~50ms
};
```

## Best Practices

1. **Confidence Scoring**: Use 0.9-1.0 for explicit facts, 0.7-0.9 for learned patterns
2. **Regular Consolidation**: Run weekly or after 10k+ inserts
3. **Semantic Tagging**: Use tags for efficient filtering
4. **Usage Tracking**: Increment usage_count to identify valuable patterns
5. **Quantization**: Use scalar (4x reduction) or binary (32x) for large databases
6. **Caching**: Set cache size to 1000+ for frequent patterns

## Related Examples

- [Example 1: Short-Term Memory](./example-1-short-term.md) - Recent context
- [Example 3: Episodic Memory](./example-3-episodic.md) - Experience tracking
