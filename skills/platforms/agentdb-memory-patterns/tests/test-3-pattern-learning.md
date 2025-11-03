# Test 3: Pattern Learning and Recognition

## Overview
Test pattern learning, recognition, and recommendation using AgentDB.

**Status**: ✅ Ready for execution
**Estimated Duration**: 10-15 minutes
**Prerequisites**: AgentDB initialized, pattern_learning.py available

---

## Test Objectives

1. Verify pattern learning from successful interactions
2. Test pattern matching and recognition
3. Validate confidence scoring and updates
4. Test pattern discovery from logs
5. Verify pattern recommendations

---

## Test Setup

### 1. Initialize Test Environment

```bash
# Create test database
export PATTERN_DB=".agentdb/test-patterns.db"
npx agentdb@latest init "$PATTERN_DB" --dimension 384

echo "Pattern database: $PATTERN_DB"
```

### 2. Import Pattern Learning Module

```python
import sys
sys.path.insert(0, 'resources/scripts')

from pattern_learning import PatternLearner, Pattern
import time
import json
```

---

## Test Cases

### Test Case 1: Pattern Learning and Confidence

**Objective**: Test pattern learning with confidence calculation.

```python
def test_pattern_learning():
    """Test pattern learning and confidence updates"""
    learner = PatternLearner('.agentdb/test-patterns.db')

    # Define test pattern
    trigger = "user_asks_time"
    response = "provide_formatted_time"

    print("Learning pattern through multiple executions...")

    # Execution 1: Success
    pattern_id_1 = learner.learn_pattern(trigger, response, success=True)
    print(f"  Execution 1 (success): Pattern ID {pattern_id_1}")

    # Execution 2: Success (same pattern)
    pattern_id_2 = learner.learn_pattern(trigger, response, success=True)
    print(f"  Execution 2 (success): Pattern ID {pattern_id_2}")

    # Verify same pattern ID
    same_id = pattern_id_1 == pattern_id_2
    print(f"{'✅' if same_id else '❌'} Same pattern ID: {same_id}")

    # Execution 3: Failure
    pattern_id_3 = learner.learn_pattern(trigger, response, success=False)
    print(f"  Execution 3 (failure): Pattern ID {pattern_id_3}")

    # Check confidence after 2 successes, 1 failure
    pattern = learner.match_pattern(trigger, min_confidence=0.0)

    expected_confidence = 2/3  # 2 successes out of 3 attempts
    actual_confidence = pattern.confidence
    confidence_ok = abs(actual_confidence - expected_confidence) < 0.01

    print(f"\nPattern statistics:")
    print(f"  Usage count: {pattern.usage_count}")
    print(f"  Success count: {pattern.success_count}")
    print(f"  Confidence: {actual_confidence:.3f} (expected: {expected_confidence:.3f})")
    print(f"{'✅' if confidence_ok else '❌'} Confidence calculation correct")

    # Verify usage and success counts
    counts_ok = (pattern.usage_count == 3 and pattern.success_count == 2)
    print(f"{'✅' if counts_ok else '❌'} Usage and success counts correct")

    learner.close()

    passed = same_id and confidence_ok and counts_ok
    print(f"\nTest Case 1: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- Pattern ID consistent across executions
- Confidence = 2/3 = 0.667
- Usage count = 3
- Success count = 2

---

### Test Case 2: Pattern Matching

**Objective**: Test pattern matching with exact and fuzzy matching.

```python
def test_pattern_matching():
    """Test pattern matching (exact and fuzzy)"""
    learner = PatternLearner('.agentdb/test-patterns.db')

    # Learn diverse patterns
    patterns = [
        ("user_greets", "respond_greeting", 0.9),
        ("user_asks_help", "provide_assistance", 0.8),
        ("user_says_goodbye", "respond_farewell", 0.95),
        ("weather_query", "fetch_weather_data", 0.7),
    ]

    print("Learning patterns...")
    for trigger, response, confidence in patterns:
        # Simulate executions to achieve target confidence
        successes = int(confidence * 10)
        failures = 10 - successes

        for _ in range(successes):
            learner.learn_pattern(trigger, response, success=True)
        for _ in range(failures):
            learner.learn_pattern(trigger, response, success=False)

    # Test 1: Exact match
    print("\nTest 1: Exact matching")
    exact_match = learner.match_pattern("user_greets", min_confidence=0.5)
    exact_ok = exact_match and exact_match.trigger == "user_greets"
    print(f"{'✅' if exact_ok else '❌'} Exact match: {exact_match.trigger if exact_match else 'None'}")

    # Test 2: Fuzzy match (contains)
    print("\nTest 2: Fuzzy matching")
    fuzzy_match = learner.match_pattern("user asks for help", min_confidence=0.5)
    fuzzy_ok = fuzzy_match and "help" in fuzzy_match.trigger
    print(f"{'✅' if fuzzy_ok else '❌'} Fuzzy match: {fuzzy_match.trigger if fuzzy_match else 'None'}")

    # Test 3: No match (low confidence threshold)
    print("\nTest 3: Confidence threshold")
    no_match = learner.match_pattern("weather_query", min_confidence=0.9)
    threshold_ok = no_match is None
    print(f"{'✅' if threshold_ok else '❌'} No match below threshold: {no_match is None}")

    # Test 4: Match with appropriate threshold
    with_match = learner.match_pattern("weather_query", min_confidence=0.6)
    threshold_match_ok = with_match and with_match.trigger == "weather_query"
    print(f"{'✅' if threshold_match_ok else '❌'} Match above threshold: {with_match is not None}")

    learner.close()

    passed = exact_ok and fuzzy_ok and threshold_ok and threshold_match_ok
    print(f"\nTest Case 2: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- Exact match finds "user_greets"
- Fuzzy match finds pattern containing "help"
- High threshold filters low-confidence patterns
- Appropriate threshold returns matches

---

### Test Case 3: Top Patterns and Filtering

**Objective**: Test retrieval of top patterns with filtering.

```python
def test_top_patterns():
    """Test top pattern retrieval and filtering"""
    learner = PatternLearner('.agentdb/test-patterns.db')

    # Get top patterns
    print("Retrieving top patterns...")
    top_patterns = learner.get_top_patterns(limit=5, min_usage=1)

    print(f"\nTop {len(top_patterns)} patterns:")
    for i, pattern in enumerate(top_patterns, 1):
        print(f"  {i}. {pattern.trigger} -> {pattern.response}")
        print(f"     Confidence: {pattern.confidence:.3f}, "
              f"Usage: {pattern.usage_count}")

    # Verify ordering (confidence DESC)
    confidences = [p.confidence for p in top_patterns]
    ordered = all(confidences[i] >= confidences[i+1]
                  for i in range(len(confidences)-1))
    print(f"\n{'✅' if ordered else '❌'} Patterns ordered by confidence")

    # Test filtering by usage
    high_usage = learner.get_top_patterns(limit=10, min_usage=5)
    usage_filtered = all(p.usage_count >= 5 for p in high_usage)
    print(f"{'✅' if usage_filtered else '❌'} Usage filtering works "
          f"({len(high_usage)} patterns with usage ≥ 5)")

    learner.close()

    passed = len(top_patterns) > 0 and ordered and usage_filtered
    print(f"\nTest Case 3: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- Top patterns retrieved successfully
- Patterns ordered by confidence descending
- Usage filter applied correctly
- All returned patterns meet criteria

---

### Test Case 4: Pattern Discovery

**Objective**: Test pattern discovery from session logs.

```python
def test_pattern_discovery():
    """Test pattern discovery from logs"""
    learner = PatternLearner('.agentdb/test-patterns.db')

    # Simulate session logs
    print("Simulating session logs...")
    session_logs = [
        {'trigger': 'error_occurred', 'response': 'show_error_message', 'success': True},
        {'trigger': 'error_occurred', 'response': 'show_error_message', 'success': True},
        {'trigger': 'error_occurred', 'response': 'show_error_message', 'success': True},
        {'trigger': 'data_loaded', 'response': 'display_data', 'success': True},
        {'trigger': 'data_loaded', 'response': 'display_data', 'success': True},
        {'trigger': 'user_clicks_button', 'response': 'handle_click', 'success': True},
        {'trigger': 'user_clicks_button', 'response': 'handle_click', 'success': True},
        {'trigger': 'user_clicks_button', 'response': 'handle_click', 'success': True},
        {'trigger': 'user_clicks_button', 'response': 'handle_click', 'success': True},
        {'trigger': 'rare_event', 'response': 'handle_rare', 'success': True},
    ]

    # Discover patterns (min frequency = 3)
    discovered = learner.discover_patterns(session_logs, min_frequency=3)

    print(f"\nDiscovered {len(discovered)} patterns (min frequency: 3):")
    for trigger, response, count in discovered:
        print(f"  {trigger} -> {response} (freq: {count})")

    # Verify discovered patterns
    expected_patterns = {
        ('error_occurred', 'show_error_message', 3),
        ('user_clicks_button', 'handle_click', 4),
    }

    discovered_set = set(discovered)
    patterns_match = expected_patterns == discovered_set
    print(f"\n{'✅' if patterns_match else '❌'} Discovered patterns match expected")

    # Verify frequency ordering
    frequencies = [count for _, _, count in discovered]
    freq_ordered = all(frequencies[i] >= frequencies[i+1]
                       for i in range(len(frequencies)-1))
    print(f"{'✅' if freq_ordered else '❌'} Patterns ordered by frequency")

    learner.close()

    passed = len(discovered) == 2 and patterns_match and freq_ordered
    print(f"\nTest Case 4: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- 2 patterns discovered (frequency ≥ 3)
- "error_occurred" pattern (3 occurrences)
- "user_clicks_button" pattern (4 occurrences)
- Patterns ordered by frequency

---

### Test Case 5: Pattern Recommendations

**Objective**: Test pattern recommendations based on context.

```python
def test_pattern_recommendations():
    """Test pattern recommendations"""
    learner = PatternLearner('.agentdb/test-patterns.db')

    # Learn patterns with context
    print("Learning patterns with context...")

    patterns_with_context = [
        ("fetch_user_data", "query_database",
         {'domain': 'data', 'operation': 'read'}),
        ("update_user_profile", "write_database",
         {'domain': 'data', 'operation': 'write'}),
        ("send_email", "smtp_send",
         {'domain': 'communication', 'operation': 'send'}),
        ("generate_report", "compile_data",
         {'domain': 'reporting', 'operation': 'generate'}),
    ]

    for trigger, response, context in patterns_with_context:
        # Learn with high success rate
        for _ in range(8):
            learner.learn_pattern(trigger, response, success=True, context=context)
        for _ in range(2):
            learner.learn_pattern(trigger, response, success=False, context=context)

    # Test 1: Recommend based on matching context
    print("\nTest 1: Context-based recommendation")
    current_context = {'domain': 'data', 'operation': 'read'}
    recommendations = learner.recommend_patterns(current_context, limit=3)

    print(f"Recommendations for context {current_context}:")
    for i, pattern in enumerate(recommendations, 1):
        print(f"  {i}. {pattern.trigger} -> {pattern.response}")
        print(f"     Context: {pattern.context}")

    # Verify data domain pattern recommended
    data_patterns = [p for p in recommendations
                     if p.context.get('domain') == 'data']
    data_recommended = len(data_patterns) > 0
    print(f"\n{'✅' if data_recommended else '❌'} Data domain patterns recommended")

    # Test 2: Different context
    print("\nTest 2: Different context recommendation")
    comm_context = {'domain': 'communication', 'operation': 'send'}
    comm_recommendations = learner.recommend_patterns(comm_context, limit=2)

    print(f"Recommendations for context {comm_context}:")
    for pattern in comm_recommendations:
        print(f"  {pattern.trigger} (domain: {pattern.context.get('domain')})")

    # Verify communication pattern recommended
    comm_patterns = [p for p in comm_recommendations
                     if p.context.get('domain') == 'communication']
    comm_recommended = len(comm_patterns) > 0
    print(f"{'✅' if comm_recommended else '❌'} Communication patterns recommended")

    learner.close()

    passed = data_recommended and comm_recommended
    print(f"\nTest Case 5: {'PASSED' if passed else 'FAILED'}")
    return passed
```

**Expected Results**:
- Recommendations match context
- Data domain patterns recommended for data context
- Communication patterns recommended for comm context
- Top recommendations are high-confidence

---

## Running the Test Suite

```python
def run_pattern_learning_tests():
    """Run all pattern learning tests"""
    print("=" * 60)
    print("AgentDB Pattern Learning Test Suite")
    print("=" * 60)

    tests = [
        ("Pattern Learning & Confidence", test_pattern_learning),
        ("Pattern Matching", test_pattern_matching),
        ("Top Patterns", test_top_patterns),
        ("Pattern Discovery", test_pattern_discovery),
        ("Pattern Recommendations", test_pattern_recommendations),
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
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print(f"\nOverall: {passed_count}/{total} tests passed")
    print(f"Success Rate: {(passed_count/total)*100:.1f}%")

    return all(p for _, p in results)

# Execute
if __name__ == '__main__':
    success = run_pattern_learning_tests()
    sys.exit(0 if success else 1)
```

---

## Cleanup

```bash
# Remove test database
rm -rf .agentdb/test-patterns.db

# Unset environment variables
unset PATTERN_DB
```

---

## Success Criteria

- ✅ All 5 test cases pass
- ✅ Confidence calculated correctly
- ✅ Pattern matching works (exact and fuzzy)
- ✅ Top patterns retrieved with proper ordering
- ✅ Pattern discovery identifies frequent patterns
- ✅ Recommendations match context

---

## Performance Benchmarks

Expected performance characteristics:

- **Pattern Learning**: < 5ms per pattern
- **Pattern Matching**: < 10ms
- **Top Patterns**: < 20ms for 100 patterns
- **Discovery**: < 50ms for 1000 log entries
- **Recommendations**: < 30ms for 5 recommendations

---

## Notes

- Pattern IDs are deterministic (SHA256 hash of trigger+response)
- Confidence updates automatically on each execution
- Fuzzy matching uses SQL LIKE operator
- Context similarity uses key overlap metric
- Patterns can have tags for categorization
