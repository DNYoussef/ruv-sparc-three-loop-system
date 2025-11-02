# Example 1: Short-Term Memory (Recent Context)

## Overview

Short-term memory maintains recent conversation context and active task state. This pattern is ideal for chat systems, command history, and session-based interactions.

**Characteristics**:
- **Capacity**: 1-100 items (configurable)
- **Retention**: Session-based or 24 hours
- **Access Pattern**: FIFO (First In, First Out) with importance scoring
- **Use Case**: Maintain conversational coherence

## Implementation

### Basic Short-Term Memory

```typescript
import { createAgentDBAdapter } from 'agentic-flow/reasoningbank';

class ShortTermMemory {
  private adapter: any;
  private maxItems: number = 50;
  private sessionId: string;

  constructor(sessionId: string, maxItems: number = 50) {
    this.sessionId = sessionId;
    this.maxItems = maxItems;
  }

  async initialize() {
    this.adapter = await createAgentDBAdapter({
      dbPath: '.agentdb/short-term.db',
      enableLearning: false,    // Not needed for short-term
      cacheSize: this.maxItems, // Cache entire short-term memory
      quantizationType: 'none'  // No quantization for recent context
    });
  }

  async storeMessage(role: 'user' | 'assistant', content: string) {
    const embedding = await this.computeEmbedding(content);

    const patternId = await this.adapter.insertPattern({
      id: '',
      type: 'short-term',
      domain: `session-${this.sessionId}`,
      pattern_data: JSON.stringify({
        embedding,
        pattern: {
          role,
          content,
          timestamp: Date.now(),
          sessionId: this.sessionId
        }
      }),
      confidence: 1.0, // High confidence for recent messages
      usage_count: 1,
      success_count: 1,
      created_at: Date.now(),
      last_used: Date.now()
    });

    // Enforce capacity limit
    await this.pruneOldMessages();

    return patternId;
  }

  async getRecentHistory(limit: number = 20): Promise<Array<{role: string, content: string}>> {
    const results = await this.adapter.searchPatterns(
      null, // No specific query, get all
      {
        domain: `session-${this.sessionId}`,
        k: limit,
        orderBy: 'created_at DESC'
      }
    );

    return results.map(r => {
      const data = JSON.parse(r.pattern_data);
      return {
        role: data.pattern.role,
        content: data.pattern.content
      };
    });
  }

  async searchContext(query: string, limit: number = 10) {
    const queryEmbedding = await this.computeEmbedding(query);

    return await this.adapter.searchPatterns(queryEmbedding, {
      domain: `session-${this.sessionId}`,
      k: limit,
      threshold: 0.7 // Relevance threshold
    });
  }

  private async pruneOldMessages() {
    const count = await this.adapter.countPatterns({
      domain: `session-${this.sessionId}`
    });

    if (count > this.maxItems) {
      // Remove oldest messages
      const toRemove = count - this.maxItems;
      await this.adapter.deleteOldest({
        domain: `session-${this.sessionId}`,
        limit: toRemove
      });
    }
  }

  private async computeEmbedding(text: string): Promise<number[]> {
    // Use your embedding model (OpenAI, local, etc.)
    // This is a placeholder
    return new Array(384).fill(0).map(() => Math.random());
  }

  async clearSession() {
    await this.adapter.deleteByDomain(`session-${this.sessionId}`);
  }
}
```

## Usage Example: Chat Application

```typescript
// Initialize short-term memory for a chat session
const memory = new ShortTermMemory('user-123-session-abc', 50);
await memory.initialize();

// Store conversation
await memory.storeMessage('user', 'What is the capital of France?');
await memory.storeMessage('assistant', 'The capital of France is Paris.');
await memory.storeMessage('user', 'And what about Italy?');
await memory.storeMessage('assistant', 'The capital of Italy is Rome.');

// Retrieve recent history (last 10 messages)
const history = await memory.getRecentHistory(10);
console.log('Recent conversation:', history);

// Search for relevant context
const context = await memory.searchContext('European capitals');
console.log('Relevant messages about capitals:', context);

// Clear session on logout
await memory.clearSession();
```

## Usage Example: Command History

```typescript
class CommandHistory extends ShortTermMemory {
  async storeCommand(command: string, result: string, exitCode: number) {
    const content = `Command: ${command}\nResult: ${result}\nExit Code: ${exitCode}`;
    return await this.storeMessage('user', content);
  }

  async getLastCommand(): Promise<string | null> {
    const history = await this.getRecentHistory(1);
    return history.length > 0 ? history[0].content : null;
  }

  async searchSimilarCommands(command: string, limit: number = 5) {
    return await this.searchContext(command, limit);
  }
}

// Usage
const cmdHistory = new CommandHistory('terminal-session-1', 100);
await cmdHistory.initialize();

await cmdHistory.storeCommand('npm install', 'Dependencies installed', 0);
await cmdHistory.storeCommand('npm run build', 'Build successful', 0);
await cmdHistory.storeCommand('npm test', '10 tests passed', 0);

// Find similar commands
const similar = await cmdHistory.searchSimilarCommands('npm build');
```

## Usage Example: Active Task Context

```typescript
class TaskContext extends ShortTermMemory {
  async setActiveTask(taskId: string, description: string, status: string) {
    return await this.storeMessage('system', JSON.stringify({
      type: 'task',
      taskId,
      description,
      status,
      timestamp: Date.now()
    }));
  }

  async getActiveTasks(): Promise<Array<any>> {
    const tasks = await this.getRecentHistory(50);
    return tasks
      .map(t => JSON.parse(t.content))
      .filter(t => t.type === 'task' && t.status === 'active');
  }

  async updateTaskStatus(taskId: string, newStatus: string) {
    const tasks = await this.getActiveTasks();
    const task = tasks.find(t => t.taskId === taskId);

    if (task) {
      task.status = newStatus;
      await this.storeMessage('system', JSON.stringify(task));
    }
  }
}

// Usage
const taskContext = new TaskContext('work-session-1', 30);
await taskContext.initialize();

await taskContext.setActiveTask('TASK-001', 'Implement authentication', 'active');
await taskContext.setActiveTask('TASK-002', 'Write unit tests', 'active');
await taskContext.setActiveTask('TASK-003', 'Update documentation', 'pending');

const active = await taskContext.getActiveTasks();
console.log('Active tasks:', active);

await taskContext.updateTaskStatus('TASK-001', 'completed');
```

## Performance Considerations

### Optimization Tips

1. **Cache Size**: Set cache to match maxItems for instant retrieval
   ```typescript
   cacheSize: this.maxItems // All items in cache
   ```

2. **No Quantization**: Short-term memory should preserve full precision
   ```typescript
   quantizationType: 'none'
   ```

3. **Session Isolation**: Use domain filtering for multi-session support
   ```typescript
   domain: `session-${this.sessionId}`
   ```

4. **Pruning Strategy**: Automatic FIFO pruning when limit reached
   ```typescript
   if (count > maxItems) {
     await deleteOldest(count - maxItems);
   }
   ```

### Performance Metrics

```typescript
// Benchmark short-term memory operations
const benchmark = async () => {
  const start = Date.now();

  // Store 50 messages
  for (let i = 0; i < 50; i++) {
    await memory.storeMessage('user', `Message ${i}`);
  }
  console.log(`Store 50 messages: ${Date.now() - start}ms`);

  // Retrieve history
  const histStart = Date.now();
  const history = await memory.getRecentHistory(20);
  console.log(`Retrieve 20 messages: ${Date.now() - histStart}ms`);

  // Search context
  const searchStart = Date.now();
  const results = await memory.searchContext('test query', 10);
  console.log(`Search context: ${Date.now() - searchStart}ms`);
};

// Expected results:
// Store 50 messages: 10-20ms (2ms per batch of 10)
// Retrieve 20 messages: <1ms (cached)
// Search context: <100µs (HNSW indexing)
```

## Integration with Long-Term Memory

Short-term memory can promote important items to long-term storage:

```typescript
async promoteToLongTerm(patternId: string, longTermAdapter: any) {
  const pattern = await this.adapter.getPattern(patternId);

  if (pattern.confidence > 0.8) { // Importance threshold
    await longTermAdapter.insertPattern({
      ...pattern,
      type: 'long-term',
      domain: 'persistent-knowledge'
    });
  }
}
```

## Best Practices

1. **Capacity Planning**: 50-100 items for chat, 100-200 for command history
2. **Session Management**: Clear memory on session end to prevent leaks
3. **Importance Scoring**: Track usage_count to identify valuable patterns
4. **Context Window**: Match cache size to typical context window needs
5. **Pruning Frequency**: Check count after each insert, prune immediately
6. **Search Threshold**: Use 0.7-0.8 for relevant context retrieval

## Related Examples

- [Example 2: Long-Term Memory](./example-2-long-term.md) - Persistent knowledge storage
- [Example 3: Episodic Memory](./example-3-episodic.md) - Experience tracking
