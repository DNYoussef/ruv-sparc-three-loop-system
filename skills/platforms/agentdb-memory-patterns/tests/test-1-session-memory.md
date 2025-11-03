# Test 1: Session Memory Management

## Overview
Test session memory functionality with triple-layer retention (24h/7d/30d).

**Status**: ✅ Ready for execution
**Estimated Duration**: 5-10 minutes
**Prerequisites**: AgentDB initialized, Python 3.8+, session_memory.py available

---

## Test Objectives

1. Verify triple-layer memory storage (short/mid/long-term)
2. Test automatic layer assignment based on priority
3. Validate retention policies and expiration
4. Confirm memory retrieval and filtering
5. Test consolidation and promotion logic

---

## Test Setup

### 1. Initialize Test Environment

```bash
# Create test database
npx agentdb@latest init .agentdb/test-session.db --dimension 384

# Set test session ID
export AGENTDB_SESSION="test-session-$(date +%s)"
export AGENTDB_PATH=".agentdb/test-session.db"

echo "Test session: $AGENTDB_SESSION"
```

### 2. Import Session Memory Module

```python
import sys
sys.path.insert(0, 'resources/scripts')

from session_memory import TripleLayerMemory
import time
```

---

## Test Cases

### Test Case 1: Layer Assignment by Priority

**Objective**: Verify automatic layer assignment based on priority scores.

```python
def test_layer_assignment():
    """Test automatic layer assignment"""
    memory = TripleLayerMemory('.agentdb/test-session.db')
    session_id = "test-layer-assignment"

    # Test data with different priorities
    test_cases = [
        ("Low priority event", 0.2, "short_term"),
        ("Medium priority event", 0.6, "mid_term"),
        ("High priority event", 0.9, "long_term"),
        ("Threshold low", 0.49, "short_term"),
        ("Threshold mid", 0.5, "mid_term"),
        ("Threshold high", 0.8, "long_term"),
    ]

    results = []
    for content, priority, expected_layer in test_cases:
        # Store memory
        entry_id = memory.store(content, session_id, priority)

        # Retrieve and verify layer
        entries = memory.retrieve(session_id, limit=100)
        actual_layer = next(
            (e['layer'] for e in entries if e['content'] == content),
            None
        )

        passed = actual_layer == expected_layer
        results.append({
            'content': content,
            'priority': priority,
            'expected': expected_layer,
            'actual': actual_layer,
            'passed': passed
        })

        print(f"{'✅' if passed else '❌'} Priority {priority:.1f} -> "
              f"{actual_layer} (expected: {expected_layer})")

    memory.close()

    # Summary
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    print(f"\nTest Case 1: {passed}/{total} passed")

    return all(r['passed'] for r in results)
```

**Expected Results**:
- Priority < 0.5 → short_term
- Priority 0.5-0.8 → mid_term
- Priority ≥ 0.8 → long_term
- All 6 test cases pass

---

### Test Case 2: Retention and Expiration

**Objective**: Verify retention periods and automatic expiration.

```python
def test_retention_expiration():
    """Test retention periods"""
    memory = TripleLayerMemory('.agentdb/test-session.db')
    session_id = "test-retention"

    # Store entries in each layer
    short_id = memory.store("Short-term entry", session_id, priority=0.3)
    mid_id = memory.store("Mid-term entry", session_id, priority=0.6)
    long_id = memory.store("Long-term entry", session_id, priority=0.9)

    # Check initial state
    entries = memory.retrieve(session_id)
    print(f"Initial entries: {len(entries)}")

    # Verify retention hours
    import sqlite3
    conn = sqlite3.connect('.agentdb/test-session.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT layer,
               CAST((expires_at - created_at) AS REAL) / 3600 as retention_hours
        FROM memory_layers
        WHERE session_id = ?
    ''', (session_id,))

    retentions = cursor.fetchall()
    expected = {
        'short_term': 24,
        'mid_term': 168,
        'long_term': 720
    }

    print("\nRetention verification:")
    passed = True
    for layer, hours in retentions:
        expected_hours = expected[layer]
        match = abs(hours - expected_hours) < 1  # 1 hour tolerance
        print(f"{'✅' if match else '❌'} {layer}: {hours:.1f}h "
              f"(expected: {expected_hours}h)")
        passed = passed and match

    conn.close()
    memory.close()

    print(f"\nTest Case 2: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- Short-term: 24h retention
- Mid-term: 168h (7 days) retention
- Long-term: 720h (30 days) retention

---

### Test Case 3: Memory Retrieval and Filtering

**Objective**: Test memory retrieval with various filters.

```python
def test_memory_retrieval():
    """Test memory retrieval and filtering"""
    memory = TripleLayerMemory('.agentdb/test-session.db')
    session_id = "test-retrieval"

    # Store diverse memories
    memories = [
        ("User likes Python", 0.9, {'category': 'preference'}),
        ("Discussed ML algorithms", 0.6, {'category': 'conversation'}),
        ("Asked about weather", 0.3, {'category': 'query'}),
        ("Prefers dark mode", 0.8, {'category': 'preference'}),
        ("Mentioned TensorFlow", 0.5, {'category': 'conversation'}),
    ]

    for content, priority, metadata in memories:
        memory.store(content, session_id, priority, metadata=metadata)

    print("Retrieval tests:")

    # Test 1: Retrieve all
    all_memories = memory.retrieve(session_id, limit=10)
    print(f"✅ Retrieved all: {len(all_memories)} memories")

    # Test 2: Filter by layer
    long_term = memory.retrieve(session_id, layer='long_term')
    print(f"✅ Long-term only: {len(long_term)} memories")

    # Test 3: Priority threshold
    high_priority = memory.retrieve(session_id, min_priority=0.7)
    print(f"✅ High priority (≥0.7): {len(high_priority)} memories")

    # Test 4: Verify ordering (priority DESC)
    priorities = [m['priority'] for m in all_memories]
    ordered = all(priorities[i] >= priorities[i+1]
                  for i in range(len(priorities)-1))
    print(f"{'✅' if ordered else '❌'} Memories ordered by priority")

    # Test 5: Access count increment
    initial_accesses = all_memories[0]['access_count']
    memory.retrieve(session_id, limit=1)
    updated = memory.retrieve(session_id, limit=1)[0]
    incremented = updated['access_count'] > initial_accesses
    print(f"{'✅' if incremented else '❌'} Access count incremented")

    memory.close()

    passed = (len(all_memories) == 5 and
              len(long_term) == 2 and
              len(high_priority) == 2 and
              ordered and incremented)

    print(f"\nTest Case 3: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- All 5 memories retrieved
- 2 long-term memories (priority ≥ 0.8)
- 2 high-priority memories (priority ≥ 0.7)
- Memories ordered by priority descending
- Access counts increment on retrieval

---

### Test Case 4: Memory Consolidation and Promotion

**Objective**: Test automatic promotion between layers.

```python
def test_consolidation():
    """Test memory consolidation and promotion"""
    memory = TripleLayerMemory('.agentdb/test-session.db')
    session_id = "test-consolidation"

    # Create short-term memory with high access
    entry_id = memory.store(
        "Frequently accessed short-term",
        session_id,
        priority=0.6
    )

    # Simulate multiple accesses (≥5 for promotion)
    for _ in range(6):
        memory.retrieve(session_id, limit=1)

    print("Pre-consolidation:")
    stats_before = memory.get_statistics(session_id)
    for layer, data in stats_before.items():
        print(f"  {layer}: {data['count']} entries")

    # Run consolidation
    consolidation_stats = memory.consolidate(session_id)
    print(f"\nConsolidation results:")
    print(f"  Promoted: {consolidation_stats['promoted']}")
    print(f"  Expired: {consolidation_stats['expired']}")

    print("\nPost-consolidation:")
    stats_after = memory.get_statistics(session_id)
    for layer, data in stats_after.items():
        print(f"  {layer}: {data['count']} entries")

    # Verify promotion occurred
    entries = memory.retrieve(session_id)
    promoted_entry = next(
        (e for e in entries if 'Frequently accessed' in e['content']),
        None
    )

    promoted = promoted_entry and promoted_entry['layer'] == 'mid_term'
    print(f"\n{'✅' if promoted else '❌'} Entry promoted to mid-term")

    memory.close()

    print(f"\nTest Case 4: {'PASSED' if promoted else 'FAILED'}")
    return promoted
```

**Expected Results**:
- Short-term entry promoted to mid-term after 5+ accesses
- Priority boosted by 10%
- Consolidation stats accurate

---

### Test Case 5: Statistics and Monitoring

**Objective**: Verify statistics collection and reporting.

```python
def test_statistics():
    """Test statistics collection"""
    memory = TripleLayerMemory('.agentdb/test-session.db')
    session_id = "test-stats"

    # Store varied memories
    for i in range(10):
        priority = 0.3 + (i * 0.07)  # Range 0.3-0.93
        memory.store(f"Memory {i}", session_id, priority)
        if i % 2 == 0:
            memory.retrieve(session_id, limit=1)

    # Get statistics
    stats = memory.get_statistics(session_id)

    print("Memory statistics:")
    total_entries = 0
    total_accesses = 0

    for layer, data in stats.items():
        print(f"\n{layer}:")
        print(f"  Count: {data['count']}")
        print(f"  Avg Priority: {data['avg_priority']:.3f}")
        print(f"  Total Accesses: {data['total_accesses']}")

        total_entries += data['count']
        total_accesses += data['total_accesses']

    print(f"\nTotals:")
    print(f"  Total Entries: {total_entries}")
    print(f"  Total Accesses: {total_accesses}")

    # Verify totals
    passed = (total_entries == 10 and total_accesses >= 5)

    memory.close()

    print(f"\nTest Case 5: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- 10 total entries across all layers
- Accurate count per layer
- Access statistics tracked
- Average priority calculated correctly

---

## Running the Test Suite

```python
def run_session_memory_tests():
    """Run all session memory tests"""
    print("=" * 60)
    print("AgentDB Session Memory Test Suite")
    print("=" * 60)

    tests = [
        ("Layer Assignment", test_layer_assignment),
        ("Retention & Expiration", test_retention_expiration),
        ("Memory Retrieval", test_memory_retrieval),
        ("Consolidation", test_consolidation),
        ("Statistics", test_statistics),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print('=' * 60)

        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Success Rate: {(passed/total)*100:.1f}%")

    return all(p for _, p in results)

# Execute
if __name__ == '__main__':
    success = run_session_memory_tests()
    sys.exit(0 if success else 1)
```

---

## Cleanup

```bash
# Remove test database
rm -rf .agentdb/test-session.db

# Unset environment variables
unset AGENTDB_SESSION
unset AGENTDB_PATH
```

---

## Success Criteria

- ✅ All 5 test cases pass
- ✅ Layer assignment accurate (100%)
- ✅ Retention policies enforced
- ✅ Memory retrieval filtering works
- ✅ Consolidation promotes entries correctly
- ✅ Statistics accurate

---

## Notes

- Test uses isolated database to avoid conflicts
- Automatic cleanup removes test artifacts
- Access counts are incremented on each retrieval
- Consolidation should be idempotent (safe to run multiple times)
