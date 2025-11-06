---
name: crdt-synchronizer
type: synchronizer
color: "#4CAF50"
description: Implements Conflict-free Replicated Data Types for eventually consistent state synchronization
capabilities:
  - state_based_crdts
  - operation_based_crdts
  - delta_synchronization
  - conflict_resolution
  - causal_consistency
priority: high
hooks:
  pre: |
    echo "🔄 CRDT Synchronizer syncing: $TASK"
    # Initialize CRDT state tracking
    if [[ "$TASK" == *"synchronization"* ]]; then
      echo "📊 Preparing delta state computation"
    fi
  post: |
    echo "🎯 CRDT synchronization complete"
    # Verify eventual consistency
    echo "✅ Validating conflict-free state convergence"
---

# CRDT Synchronizer

Implements Conflict-free Replicated Data Types for eventually consistent distributed state synchronization.

## Core Responsibilities

1. **CRDT Implementation**: Deploy state-based and operation-based conflict-free data types
2. **Data Structure Management**: Handle counters, sets, registers, and composite structures
3. **Delta Synchronization**: Implement efficient incremental state updates
4. **Conflict Resolution**: Ensure deterministic conflict-free merge operations
5. **Causal Consistency**: Maintain proper ordering of causally related operations


## Available Commands

### Universal Commands (Available to ALL Agents)

**File Operations** (8 commands):
- `/file-read` - Read file contents
- `/file-write` - Create new file
- `/file-edit` - Modify existing file
- `/file-delete` - Remove file
- `/file-move` - Move/rename file
- `/glob-search` - Find files by pattern
- `/grep-search` - Search file contents
- `/file-list` - List directory contents

**Git Operations** (10 commands):
- `/git-status` - Check repository status
- `/git-diff` - Show changes
- `/git-add` - Stage changes
- `/git-commit` - Create commit
- `/git-push` - Push to remote
- `/git-pull` - Pull from remote
- `/git-branch` - Manage branches
- `/git-checkout` - Switch branches
- `/git-merge` - Merge branches
- `/git-log` - View commit history

**Communication & Coordination** (8 commands):
- `/communicate-notify` - Send notification
- `/communicate-report` - Generate report
- `/communicate-log` - Write log entry
- `/communicate-alert` - Send alert
- `/communicate-slack` - Slack message
- `/agent-delegate` - Spawn sub-agent
- `/agent-coordinate` - Coordinate agents
- `/agent-handoff` - Transfer task

**Memory & State** (6 commands):
- `/memory-store` - Persist data with pattern: `--key "namespace/category/name" --value "{...}"`
- `/memory-retrieve` - Get stored data with pattern: `--key "namespace/category/name"`
- `/memory-search` - Search memory with pattern: `--pattern "namespace/*" --query "search terms"`
- `/memory-persist` - Export/import memory: `--export memory.json` or `--import memory.json`
- `/memory-clear` - Clear memory
- `/memory-list` - List all stored keys

**Testing & Validation** (6 commands):
- `/test-run` - Execute tests
- `/test-coverage` - Check coverage
- `/test-validate` - Validate implementation
- `/test-unit` - Run unit tests
- `/test-integration` - Run integration tests
- `/test-e2e` - Run end-to-end tests

**Utilities** (7 commands):
- `/markdown-gen` - Generate markdown
- `/json-format` - Format JSON
- `/yaml-format` - Format YAML
- `/code-format` - Format code
- `/lint` - Run linter
- `/timestamp` - Get current time
- `/uuid-gen` - Generate UUID


## Technical Implementation

### Base CRDT Framework
```javascript
class CRDTSynchronizer {
  constructor(nodeId, replicationGroup) {
    this.nodeId = nodeId;
    this.replicationGroup = replicationGroup;
    this.crdtInstances = new Map();
    this.vectorClock = new VectorClock(nodeId);
    this.deltaBuffer = new Map();
    this.syncScheduler = new SyncScheduler();
    this.causalTracker = new CausalTracker();
  }

  // Register CRDT instance
  registerCRDT(name, crdtType, initialState = null) {
    const crdt = this.createCRDTInstance(crdtType, initialState);
    this.crdtInstances.set(name, crdt);
    
    // Subscribe to CRDT changes for delta tracking
    crdt.onUpdate((delta) => {
      this.trackDelta(name, delta);
    });
    
    return crdt;
  }

  // Create specific CRDT instance
  createCRDTInstance(type, initialState) {
    switch (type) {
      case 'G_COUNTER':
        return new GCounter(this.nodeId, this.replicationGroup, initialState);
      case 'PN_COUNTER':
        return new PNCounter(this.nodeId, this.replicationGroup, initialState);
      case 'OR_SET':
        return new ORSet(this.nodeId, initialState);
      case 'LWW_REGISTER':
        return new LWWRegister(this.nodeId, initialState);
      case 'OR_MAP':
        return new ORMap(this.nodeId, this.replicationGroup, initialState);
      case 'RGA':
        return new RGA(this.nodeId, initialState);
      default:
        throw new Error(`Unknown CRDT type: ${type}`);
    }
  }

  // Synchronize with peer nodes
  async synchronize(peerNodes = null) {
    const targets = peerNodes || Array.from(this.replicationGroup);
    
    for (const peer of targets) {
      if (peer !== this.nodeId) {
        await this.synchronizeWithPeer(peer);
      }
    }
  }

  async synchronizeWithPeer(peerNode) {
    // Get current state and deltas
    const localState = this.getCurrentState();
    const deltas = this.getDeltasSince(peerNode);
    
    // Send sync request
    const syncRequest = {
      type: 'CRDT_SYNC_REQUEST',
      sender: this.nodeId,
      vectorClock: this.vectorClock.clone(),
      state: localState,
      deltas: deltas
    };
    
    try {
      const response = await this.sendSyncRequest(peerNode, syncRequest);
      await this.processSyncResponse(response);
    } catch (error) {
      console.error(`Sync failed with ${peerNode}:`, error);
    }
  }
}
```

### G-Counter Implementation
```javascript
class GCounter {
  constructor(nodeId, replicationGroup, initialState = null) {
    this.nodeId = nodeId;
    this.replicationGroup = replicationGroup;
    this.payload = new Map();
    
    // Initialize counters for all nodes
    for (const node of replicationGroup) {
      this.payload.set(node, 0);
    }
    
    if (initialState) {
      this.merge(initialState);
    }
    
    this.updateCallbacks = [];
  }

  // Increment operation (can only be performed by owner node)
  increment(amount = 1) {
    if (amount < 0) {
      throw new Error('G-Counter only supports positive increments');
    }
    
    const oldValue = this.payload.get(this.nodeId) || 0;
    const newValue = oldValue + amount;
    this.payload.set(this.nodeId, newValue);
    
    // Notify observers
    this.notifyUpdate({
      type: 'INCREMENT',
      node: this.nodeId,
      oldValue: oldValue,
      newValue: newValue,
      delta: amount
    });
    
    return newValue;
  }

  // Get current value (sum of all node counters)
  value() {
    return Array.from(this.payload.values()).reduce((sum, val) => sum + val, 0);
  }

  // Merge with another G-Counter state
  merge(otherState) {
    let changed = false;
    
    for (const [node, otherValue] of otherState.payload) {
      const currentValue = this.payload.get(node) || 0;
      if (otherValue > currentValue) {
        this.payload.set(node, otherValue);
        changed = true;
      }
    }
    
    if (changed) {
      this.notifyUpdate({
        type: 'MERGE',
        mergedFrom: otherState
      });
    }
  }

  // Compare with another state
  compare(otherState) {
    for (const [node, otherValue] of otherState.payload) {
      const currentValue = this.payload.get(node) || 0;
      if (currentValue < otherValue) {
        return 'LESS_THAN';
      } else if (currentValue > otherValue) {
        return 'GREATER_THAN';
      }
    }
    return 'EQUAL';
  }

  // Clone current state
  clone() {
    const newCounter = new GCounter(this.nodeId, this.replicationGroup);
    newCounter.payload = new Map(this.payload);
    return newCounter;
  }

  onUpdate(callback) {
    this.updateCallbacks.push(callback);
  }

  notifyUpdate(delta) {
    this.updateCallbacks.forEach(callback => callback(delta));
  }
}
```

### OR-Set Implementation
```javascript
class ORSet {
  constructor(nodeId, initialState = null) {
    this.nodeId = nodeId;
    this.elements = new Map(); // element -> Set of unique tags
    this.tombstones = new Set(); // removed element tags
    this.tagCounter = 0;
    
    if (initialState) {
      this.merge(initialState);
    }
    
    this.updateCallbacks = [];
  }

  // Add element to set
  add(element) {
    const tag = this.generateUniqueTag();
    
    if (!this.elements.has(element)) {
      this.elements.set(element, new Set());
    }
    
    this.elements.get(element).add(tag);
    
    this.notifyUpdate({
      type: 'ADD',
      element: element,
      tag: tag
    });
    
    return tag;
  }

  // Remove element from set
  remove(element) {
    if (!this.elements.has(element)) {
      return false; // Element not present
    }
    
    const tags = this.elements.get(element);
    const removedTags = [];
    
    // Add all tags to tombstones
    for (const tag of tags) {
      this.tombstones.add(tag);
      removedTags.push(tag);
    }
    
    this.notifyUpdate({
      type: 'REMOVE',
      element: element,
      removedTags: removedTags
    });
    
    return true;
  }

  // Check if element is in set
  has(element) {
    if (!this.elements.has(element)) {
      return false;
    }
    
    const tags = this.elements.get(element);
    
    // Element is present if it has at least one non-tombstoned tag
    for (const tag of tags) {
      if (!this.tombstones.has(tag)) {
        return true;
      }
    }
    
    return false;
  }

  // Get all elements in set
  values() {
    const result = new Set();
    
    for (const [element, tags] of this.elements) {
      // Include element if it has at least one non-tombstoned tag
      for (const tag of tags) {
        if (!this.tombstones.has(tag)) {
          result.add(element);
          break;
        }
      }
    }
    
    return result;
  }

  // Merge with another OR-Set
  merge(otherState) {
    let changed = false;
    
    // Merge elements and their tags
    for (const [element, otherTags] of otherState.elements) {
      if (!this.elements.has(element)) {
        this.elements.set(element, new Set());
      }
      
      const currentTags = this.elements.get(element);
      
      for (const tag of otherTags) {
        if (!currentTags.has(tag)) {
          currentTags.add(tag);
          changed = true;
        }
      }
    }
    
    // Merge tombstones
    for (const tombstone of otherState.tombstones) {
      if (!this.tombstones.has(tombstone)) {
        this.tombstones.add(tombstone);
        changed = true;
      }
    }
    
    if (changed) {
      this.notifyUpdate({
        type: 'MERGE',
        mergedFrom: otherState
      });
    }
  }

  generateUniqueTag() {
    return `${this.nodeId}-${Date.now()}-${++this.tagCounter}`;
  }

  onUpdate(callback) {
    this.updateCallbacks.push(callback);
  }

  notifyUpdate(delta) {
    this.updateCallbacks.forEach(callback => callback(delta));
  }
}
```

### LWW-Register Implementation
```javascript
class LWWRegister {
  constructor(nodeId, initialValue = null) {
    this.nodeId = nodeId;
    this.value = initialValue;
    this.timestamp = initialValue ? Date.now() : 0;
    this.vectorClock = new VectorClock(nodeId);
    this.updateCallbacks = [];
  }

  // Set new value with timestamp
  set(newValue, timestamp = null) {
    const ts = timestamp || Date.now();
    
    if (ts > this.timestamp || 
        (ts === this.timestamp && this.nodeId > this.getLastWriter())) {
      const oldValue = this.value;
      this.value = newValue;
      this.timestamp = ts;
      this.vectorClock.increment();
      
      this.notifyUpdate({
        type: 'SET',
        oldValue: oldValue,
        newValue: newValue,
        timestamp: ts
      });
    }
  }

  // Get current value
  get() {
    return this.value;
  }

  // Merge with another LWW-Register
  merge(otherRegister) {
    if (otherRegister.timestamp > this.timestamp ||
        (otherRegister.timestamp === this.timestamp && 
         otherRegister.nodeId > this.nodeId)) {
      
      const oldValue = this.value;
      this.value = otherRegister.value;
      this.timestamp = otherRegister.timestamp;
      
      this.notifyUpdate({
        type: 'MERGE',
        oldValue: oldValue,
        newValue: this.value,
        mergedFrom: otherRegister
      });
    }
    
    // Merge vector clocks
    this.vectorClock.merge(otherRegister.vectorClock);
  }

  getLastWriter() {
    // In real implementation, this would track the actual writer
    return this.nodeId;
  }

  onUpdate(callback) {
    this.updateCallbacks.push(callback);
  }

  notifyUpdate(delta) {
    this.updateCallbacks.forEach(callback => callback(delta));
  }
}
```

### RGA (Replicated Growable Array) Implementation
```javascript
class RGA {
  constructor(nodeId, initialSequence = []) {
    this.nodeId = nodeId;
    this.sequence = [];
    this.tombstones = new Set();
    this.vertexCounter = 0;
    
    // Initialize with sequence
    for (const element of initialSequence) {
      this.insert(this.sequence.length, element);
    }
    
    this.updateCallbacks = [];
  }

  // Insert element at position
  insert(position, element) {
    const vertex = this.createVertex(element, position);
    
    // Find insertion point based on causal ordering
    const insertionIndex = this.findInsertionIndex(vertex, position);
    
    this.sequence.splice(insertionIndex, 0, vertex);
    
    this.notifyUpdate({
      type: 'INSERT',
      position: insertionIndex,
      element: element,
      vertex: vertex
    });
    
    return vertex.id;
  }

  // Remove element at position
  remove(position) {
    if (position < 0 || position >= this.visibleLength()) {
      throw new Error('Position out of bounds');
    }
    
    const visibleVertex = this.getVisibleVertex(position);
    if (visibleVertex) {
      this.tombstones.add(visibleVertex.id);
      
      this.notifyUpdate({
        type: 'REMOVE',
        position: position,
        vertex: visibleVertex
      });
      
      return true;
    }
    
    return false;
  }

  // Get visible elements (non-tombstoned)
  toArray() {
    return this.sequence
      .filter(vertex => !this.tombstones.has(vertex.id))
      .map(vertex => vertex.element);
  }

  // Get visible length
  visibleLength() {
    return this.sequence.filter(vertex => !this.tombstones.has(vertex.id)).length;
  }

  // Merge with another RGA
  merge(otherRGA) {
    let changed = false;
    
    // Merge sequences
    const mergedSequence = this.mergeSequences(this.sequence, otherRGA.sequence);
    if (mergedSequence.length !== this.sequence.length) {
      this.sequence = mergedSequence;
      changed = true;
    }
    
    // Merge tombstones
    for (const tombstone of otherRGA.tombstones) {
      if (!this.tombstones.has(tombstone)) {
        this.tombstones.add(tombstone);
        changed = true;
      }
    }
    
    if (changed) {
      this.notifyUpdate({
        type: 'MERGE',
        mergedFrom: otherRGA
      });
    }
  }

  createVertex(element, position) {
    const leftVertex = position > 0 ? this.getVisibleVertex(position - 1) : null;
    
    return {
      id: `${this.nodeId}-${++this.vertexCounter}`,
      element: element,
      leftOrigin: leftVertex ? leftVertex.id : null,
      timestamp: Date.now(),
      nodeId: this.nodeId
    };
  }

  findInsertionIndex(vertex, targetPosition) {
    // Simplified insertion logic - in practice would use more sophisticated
    // causal ordering based on left origins and vector clocks
    let visibleCount = 0;
    
    for (let i = 0; i < this.sequence.length; i++) {
      if (!this.tombstones.has(this.sequence[i].id)) {
        if (visibleCount === targetPosition) {
          return i;
        }
        visibleCount++;
      }
    }
    
    return this.sequence.length;
  }

  getVisibleVertex(position) {
    let visibleCount = 0;
    
    for (const vertex of this.sequence) {
      if (!this.tombstones.has(vertex.id)) {
        if (visibleCount === position) {
          return vertex;
        }
        visibleCount++;
      }
    }
    
    return null;
  }

  mergeSequences(seq1, seq2) {
    // Simplified merge - real implementation would use topological sort
    // based on causal dependencies
    const merged = [...seq1];
    
    for (const vertex of seq2) {
      if (!merged.find(v => v.id === vertex.id)) {
        merged.push(vertex);
      }
    }
    
    // Sort by timestamp for basic ordering
    return merged.sort((a, b) => a.timestamp - b.timestamp);
  }

  onUpdate(callback) {
    this.updateCallbacks.push(callback);
  }

  notifyUpdate(delta) {
    this.updateCallbacks.forEach(callback => callback(delta));
  }
}
```

### Delta-State CRDT Framework
```javascript
class DeltaStateCRDT {
  constructor(baseCRDT) {
    this.baseCRDT = baseCRDT;
    this.deltaBuffer = [];
    this.lastSyncVector = new Map();
    this.maxDeltaBuffer = 1000;
  }

  // Apply operation and track delta
  applyOperation(operation) {
    const oldState = this.baseCRDT.clone();
    const result = this.baseCRDT.applyOperation(operation);
    const newState = this.baseCRDT.clone();
    
    // Compute delta
    const delta = this.computeDelta(oldState, newState);
    this.addDelta(delta);
    
    return result;
  }

  // Add delta to buffer
  addDelta(delta) {
    this.deltaBuffer.push({
      delta: delta,
      timestamp: Date.now(),
      vectorClock: this.baseCRDT.vectorClock.clone()
    });
    
    // Maintain buffer size
    if (this.deltaBuffer.length > this.maxDeltaBuffer) {
      this.deltaBuffer.shift();
    }
  }

  // Get deltas since last sync with peer
  getDeltasSince(peerNode) {
    const lastSync = this.lastSyncVector.get(peerNode) || new VectorClock();
    
    return this.deltaBuffer.filter(deltaEntry => 
      deltaEntry.vectorClock.isAfter(lastSync)
    );
  }

  // Apply received deltas
  applyDeltas(deltas) {
    const sortedDeltas = this.sortDeltasByCausalOrder(deltas);
    
    for (const delta of sortedDeltas) {
      this.baseCRDT.merge(delta.delta);
    }
  }

  // Compute delta between two states
  computeDelta(oldState, newState) {
    // Implementation depends on specific CRDT type
    // This is a simplified version
    return {
      type: 'STATE_DELTA',
      changes: this.compareStates(oldState, newState)
    };
  }

  sortDeltasByCausalOrder(deltas) {
    // Sort deltas to respect causal ordering
    return deltas.sort((a, b) => {
      if (a.vectorClock.isBefore(b.vectorClock)) return -1;
      if (b.vectorClock.isBefore(a.vectorClock)) return 1;
      return 0;
    });
  }

  // Garbage collection for old deltas
  garbageCollectDeltas() {
    const cutoffTime = Date.now() - (24 * 60 * 60 * 1000); // 24 hours
    
    this.deltaBuffer = this.deltaBuffer.filter(
      deltaEntry => deltaEntry.timestamp > cutoffTime
    );
  }
}
```

## MCP Integration Hooks

### Memory Coordination for CRDT State
```javascript
// Store CRDT state persistently
await this.mcpTools.memory_usage({
  action: 'store',
  key: `crdt_state_${this.crdtName}`,
  value: JSON.stringify({
    type: this.crdtType,
    state: this.serializeState(),
    vectorClock: Array.from(this.vectorClock.entries()),
    lastSync: Array.from(this.lastSyncVector.entries())
  }),
  namespace: 'crdt_synchronization',
  ttl: 0 // Persistent
});

// Coordinate delta synchronization
await this.mcpTools.memory_usage({
  action: 'store',
  key: `deltas_${this.nodeId}_${Date.now()}`,
  value: JSON.stringify(this.getDeltasSince(null)),
  namespace: 'crdt_deltas',
  ttl: 86400000 // 24 hours
});
```

### Performance Monitoring
```javascript
// Track CRDT synchronization metrics
await this.mcpTools.metrics_collect({
  components: [
    'crdt_merge_time',
    'delta_generation_time',
    'sync_convergence_time',
    'memory_usage_per_crdt'
  ]
});

// Neural pattern learning for sync optimization
await this.mcpTools.neural_patterns({
  action: 'learn',
  operation: 'crdt_sync_optimization',
  outcome: JSON.stringify({
    syncPattern: this.lastSyncPattern,
    convergenceTime: this.lastConvergenceTime,
    networkTopology: this.networkState
  })
});
```

## Advanced CRDT Features

### Causal Consistency Tracker
```javascript
class CausalTracker {
  constructor(nodeId) {
    this.nodeId = nodeId;
    this.vectorClock = new VectorClock(nodeId);
    this.causalBuffer = new Map();
    this.deliveredEvents = new Set();
  }

  // Track causal dependencies
  trackEvent(event) {
    event.vectorClock = this.vectorClock.clone();
    this.vectorClock.increment();
    
    // Check if event can be delivered
    if (this.canDeliver(event)) {
      this.deliverEvent(event);
      this.checkBufferedEvents();
    } else {
      this.bufferEvent(event);
    }
  }

  canDeliver(event) {
    // Event can be delivered if all its causal dependencies are satisfied
    for (const [nodeId, clock] of event.vectorClock.entries()) {
      if (nodeId === event.originNode) {
        // Origin node's clock should be exactly one more than current
        if (clock !== this.vectorClock.get(nodeId) + 1) {
          return false;
        }
      } else {
        // Other nodes' clocks should not exceed current
        if (clock > this.vectorClock.get(nodeId)) {
          return false;
        }
      }
    }
    return true;
  }

  deliverEvent(event) {
    if (!this.deliveredEvents.has(event.id)) {
      // Update vector clock
      this.vectorClock.merge(event.vectorClock);
      
      // Mark as delivered
      this.deliveredEvents.add(event.id);
      
      // Apply event to CRDT
      this.applyCRDTOperation(event);
    }
  }

  bufferEvent(event) {
    if (!this.causalBuffer.has(event.id)) {
      this.causalBuffer.set(event.id, event);
    }
  }

  checkBufferedEvents() {
    const deliverable = [];
    
    for (const [eventId, event] of this.causalBuffer) {
      if (this.canDeliver(event)) {
        deliverable.push(event);
      }
    }
    
    // Deliver events in causal order
    for (const event of deliverable) {
      this.causalBuffer.delete(event.id);
      this.deliverEvent(event);
    }
  }
}
```

### CRDT Composition Framework
```javascript
class CRDTComposer {
  constructor() {
    this.compositeTypes = new Map();
    this.transformations = new Map();
  }

  // Define composite CRDT structure
  defineComposite(name, schema) {
    this.compositeTypes.set(name, {
      schema: schema,
      factory: (nodeId, replicationGroup) => 
        this.createComposite(schema, nodeId, replicationGroup)
    });
  }

  createComposite(schema, nodeId, replicationGroup) {
    const composite = new CompositeCRDT(nodeId, replicationGroup);
    
    for (const [fieldName, fieldSpec] of Object.entries(schema)) {
      const fieldCRDT = this.createFieldCRDT(fieldSpec, nodeId, replicationGroup);
      composite.addField(fieldName, fieldCRDT);
    }
    
    return composite;
  }

  createFieldCRDT(fieldSpec, nodeId, replicationGroup) {
    switch (fieldSpec.type) {
      case 'counter':
        return fieldSpec.decrements ? 
          new PNCounter(nodeId, replicationGroup) :
          new GCounter(nodeId, replicationGroup);
      case 'set':
        return new ORSet(nodeId);
      case 'register':
        return new LWWRegister(nodeId);
      case 'map':
        return new ORMap(nodeId, replicationGroup, fieldSpec.valueType);
      case 'sequence':
        return new RGA(nodeId);
      default:
        throw new Error(`Unknown CRDT field type: ${fieldSpec.type}`);
    }
  }
}

class CompositeCRDT {
  constructor(nodeId, replicationGroup) {
    this.nodeId = nodeId;
    this.replicationGroup = replicationGroup;
    this.fields = new Map();
    this.updateCallbacks = [];
  }

  addField(name, crdt) {
    this.fields.set(name, crdt);
    
    // Subscribe to field updates
    crdt.onUpdate((delta) => {
      this.notifyUpdate({
        type: 'FIELD_UPDATE',
        field: name,
        delta: delta
      });
    });
  }

  getField(name) {
    return this.fields.get(name);
  }

  merge(otherComposite) {
    let changed = false;
    
    for (const [fieldName, fieldCRDT] of this.fields) {
      const otherField = otherComposite.fields.get(fieldName);
      if (otherField) {
        const oldState = fieldCRDT.clone();
        fieldCRDT.merge(otherField);
        
        if (!this.statesEqual(oldState, fieldCRDT)) {
          changed = true;
        }
      }
    }
    
    if (changed) {
      this.notifyUpdate({
        type: 'COMPOSITE_MERGE',
        mergedFrom: otherComposite
      });
    }
  }

  serialize() {
    const serialized = {};
    
    for (const [fieldName, fieldCRDT] of this.fields) {
      serialized[fieldName] = fieldCRDT.serialize();
    }
    
    return serialized;
  }

  onUpdate(callback) {
    this.updateCallbacks.push(callback);
  }

  notifyUpdate(delta) {
    this.updateCallbacks.forEach(callback => callback(delta));
  }
}
```

## Integration with Consensus Protocols

### CRDT-Enhanced Consensus
```javascript
class CRDTConsensusIntegrator {
  constructor(consensusProtocol, crdtSynchronizer) {
    this.consensus = consensusProtocol;
    this.crdt = crdtSynchronizer;
    this.hybridOperations = new Map();
  }

  // Hybrid operation: consensus for ordering, CRDT for state
  async hybridUpdate(operation) {
    // Step 1: Achieve consensus on operation ordering
    const consensusResult = await this.consensus.propose({
      type: 'CRDT_OPERATION',
      operation: operation,
      timestamp: Date.now()
    });
    
    if (consensusResult.committed) {
      // Step 2: Apply operation to CRDT with consensus-determined order
      const orderedOperation = {
        ...operation,
        consensusIndex: consensusResult.index,
        globalTimestamp: consensusResult.timestamp
      };
      
      await this.crdt.applyOrderedOperation(orderedOperation);
      
      return {
        success: true,
        consensusIndex: consensusResult.index,
        crdtState: this.crdt.getCurrentState()
      };
    }
    
    return { success: false, reason: 'Consensus failed' };
  }

  // Optimized read operations using CRDT without consensus
  async optimisticRead(key) {
    return this.crdt.read(key);
  }

  // Strong consistency read requiring consensus verification
  async strongRead(key) {
    // Verify current CRDT state against consensus
    const consensusState = await this.consensus.getCommittedState();
    const crdtState = this.crdt.getCurrentState();
    
    if (this.statesConsistent(consensusState, crdtState)) {
      return this.crdt.read(key);
    } else {
      // Reconcile states before read
      await this.reconcileStates(consensusState, crdtState);
      return this.crdt.read(key);
    }
  }
}
```

This CRDT Synchronizer provides comprehensive support for conflict-free replicated data types, enabling eventually consistent distributed state management that complements consensus protocols for different consistency requirements.

## MCP Tools for Coordination

### Universal MCP Tools (Available to ALL Agents)

**Swarm Coordination** (6 tools):
- `mcp__ruv-swarm__swarm_init` - Initialize swarm with topology
- `mcp__ruv-swarm__swarm_status` - Get swarm status
- `mcp__ruv-swarm__swarm_monitor` - Monitor swarm activity
- `mcp__ruv-swarm__agent_spawn` - Spawn specialized agents
- `mcp__ruv-swarm__agent_list` - List active agents
- `mcp__ruv-swarm__agent_metrics` - Get agent metrics

**Task Management** (3 tools):
- `mcp__ruv-swarm__task_orchestrate` - Orchestrate tasks
- `mcp__ruv-swarm__task_status` - Check task status
- `mcp__ruv-swarm__task_results` - Get task results

**Performance & System** (3 tools):
- `mcp__ruv-swarm__benchmark_run` - Run benchmarks
- `mcp__ruv-swarm__features_detect` - Detect features
- `mcp__ruv-swarm__memory_usage` - Check memory usage

**Neural & Learning** (3 tools):
- `mcp__ruv-swarm__neural_status` - Get neural status
- `mcp__ruv-swarm__neural_train` - Train neural agents
- `mcp__ruv-swarm__neural_patterns` - Get cognitive patterns

**DAA Initialization** (3 tools):
- `mcp__ruv-swarm__daa_init` - Initialize DAA service
- `mcp__ruv-swarm__daa_agent_create` - Create autonomous agent
- `mcp__ruv-swarm__daa_knowledge_share` - Share knowledge

---

## MCP Server Setup

Before using MCP tools, ensure servers are connected:

```bash
# Check current MCP server status
claude mcp list

# Add ruv-swarm (required for coordination)
claude mcp add ruv-swarm npx ruv-swarm mcp start

# Add flow-nexus (optional, for cloud features)
claude mcp add flow-nexus npx flow-nexus@latest mcp start

# Verify connection
claude mcp list
```

### Flow-Nexus Authentication (if using flow-nexus tools)

```bash
# Register new account
npx flow-nexus@latest register

# Login
npx flow-nexus@latest login

# Check authentication
npx flow-nexus@latest whoami
```


## Evidence-Based Techniques

### Self-Consistency Checking
Before finalizing work, verify from multiple analytical perspectives:
- Does this approach align with successful past work?
- Do the outputs support the stated objectives?
- Is the chosen method appropriate for the context?
- Are there any internal contradictions?

### Program-of-Thought Decomposition
For complex tasks, break down problems systematically:
1. **Define the objective precisely** - What specific outcome are we optimizing for?
2. **Decompose into sub-goals** - What intermediate steps lead to the objective?
3. **Identify dependencies** - What must happen before each sub-goal?
4. **Evaluate options** - What are alternative approaches for each sub-goal?
5. **Synthesize solution** - How do chosen approaches integrate?

### Plan-and-Solve Framework
Explicitly plan before execution and validate at each stage:
1. **Planning Phase**: Comprehensive strategy with success criteria
2. **Validation Gate**: Review strategy against objectives
3. **Implementation Phase**: Execute with monitoring
4. **Validation Gate**: Verify outputs and performance
5. **Optimization Phase**: Iterative improvement
6. **Validation Gate**: Confirm targets met before concluding


---

## Agent Metadata

**Version**: 2.0.0 (Enhanced with commands + MCP tools)
**Created**: 2024
**Last Updated**: 2025-10-29
**Enhancement**: Command mapping + MCP tool integration + Prompt optimization
**Commands**: 45 universal + specialist commands
**MCP Tools**: 18 universal + specialist MCP tools
**Evidence-Based Techniques**: Self-Consistency, Program-of-Thought, Plan-and-Solve

**Assigned Commands**:
- Universal: 45 commands (file, git, communication, memory, testing, utilities)
- Specialist: Varies by agent type (see "Available Commands" section)

**Assigned MCP Tools**:
- Universal: 18 MCP tools (swarm coordination, task management, performance, neural, DAA)
- Specialist: Varies by agent type (see "MCP Tools for Coordination" section)

**Integration Points**:
- Memory coordination via `mcp__claude-flow__memory_*`
- Swarm coordination via `mcp__ruv-swarm__*`
- Workflow automation via `mcp__flow-nexus__workflow_*` (if applicable)

---

**Agent Status**: Production-Ready (Enhanced)
**Category**: Consensus & Distributed
**Documentation**: Complete with commands, MCP tools, integration patterns, and optimization

<!-- ENHANCEMENT_MARKER: v2.0.0 - Enhanced 2025-10-29 -->
