# Example 2: Debugging Race Condition (Intermediate)

## Overview

This example demonstrates debugging a complex race condition in asynchronous code that causes intermittent data corruption.

**Difficulty**: Intermediate
**Estimated Time**: 45-60 minutes
**Agents Used**: code-analyzer, coder, tester

## The Problem

### Initial Symptom

**User Report**: "Shopping cart items randomly disappear when adding products quickly. Sometimes duplicate items appear."

**Error Logs**:
```
[WARN] Cart version mismatch for user_123
[ERROR] Failed to update cart: data integrity violation
[INFO] Cart save conflict - retry attempt 1
```

**Reproduction**: Intermittent - occurs ~30% of the time during load testing with concurrent requests.

## Phase 1: Symptom Identification

### Agent: code-analyzer

**Investigation Steps**:

1. **Collect data from production monitoring**:
   - Error rate spikes during high traffic periods
   - Multiple concurrent requests to same user's cart
   - Data inconsistencies between cart reads

2. **Reproduce locally**:
```javascript
// Test that triggers the issue
async function testConcurrentCartUpdates() {
  const userId = 'test_user_123';
  await createEmptyCart(userId);

  // Simulate concurrent add-to-cart requests
  const results = await Promise.all([
    addToCart(userId, { id: 'item1', name: 'Product 1' }),
    addToCart(userId, { id: 'item2', name: 'Product 2' }),
    addToCart(userId, { id: 'item3', name: 'Product 3' })
  ]);

  const cart = await getCart(userId);
  console.log('Expected: 3 items');
  console.log('Actual:', cart.items.length);
  // Output: Actual: 1 or 2 (items lost!)
}
```

3. **Document failure patterns**:
   - Items lost when adding to cart concurrently
   - Last write wins - earlier updates overwritten
   - No error thrown - silent data loss

**Symptom Report**:
```
Issue ID: BUG-2024-042
Type: Race Condition / Data Corruption
Severity: Critical
Frequency: 30% under concurrent load
Root Location: cart-service.js:updateCart()
Expected: All concurrent cart additions should be preserved
Actual: Last update overwrites previous updates (lost data)
Impact: Customer cart items disappear → Lost sales
```

## Phase 2: Root Cause Analysis

### Agent: code-analyzer + coder

**Current Implementation**:
```javascript
// cart-service.js - PROBLEMATIC CODE
async function addToCart(userId, item) {
  // Step 1: Read current cart
  const cart = await db.getCart(userId);

  // Step 2: Modify cart
  cart.items.push(item);

  // Step 3: Write back to database
  await db.saveCart(userId, cart);

  return cart;
}
```

**Execution Trace (Timeline Analysis)**:

```
Time  | Request A           | Request B           | Request C
------|---------------------|---------------------|---------------------
T0    | Read cart (empty)   |                     |
T1    |                     | Read cart (empty)   |
T2    | Modify: [item1]     |                     | Read cart (empty)
T3    |                     | Modify: [item2]     |
T4    | Save cart: [item1]  |                     |
T5    |                     | Save cart: [item2]  | Modify: [item3]
T6    |                     |                     | Save cart: [item3]
------|---------------------|---------------------|---------------------
Final state: [item3] only
Lost items: item1, item2
```

**Why This Occurs (Root Cause)**:

> The `addToCart` function uses a **read-modify-write** pattern without any concurrency control. When multiple requests execute simultaneously, they all read the same initial state, make independent modifications, and then overwrite each other's changes. This is a classic **lost update** race condition.

**Key Issues**:
1. **No locking mechanism**: Multiple requests can read/write simultaneously
2. **No version tracking**: Database doesn't detect conflicting updates
3. **No retry logic**: Failed updates are not retried
4. **Silent failures**: No error indication when data is lost

**Related Code Affected**:
- `removeFromCart()` - Same race condition pattern
- `updateQuantity()` - Same race condition pattern
- `clearCart()` - Could conflict with concurrent adds

## Phase 3: Fix Generation

### Agent: coder

**Solution Approaches Evaluated**:

### Option 1: Pessimistic Locking (Database Row Lock)
```javascript
async function addToCart(userId, item) {
  const transaction = await db.beginTransaction();
  try {
    // Lock the cart row
    const cart = await db.getCartForUpdate(userId, { lock: true });
    cart.items.push(item);
    await db.saveCart(userId, cart);
    await transaction.commit();
    return cart;
  } catch (error) {
    await transaction.rollback();
    throw error;
  }
}
```
**Pros**: Simple, guaranteed consistency
**Cons**: Poor scalability, blocking, potential deadlocks, longer lock times
**Trade-offs**: Not ideal for high-traffic e-commerce

### Option 2: Optimistic Locking with Version Field (CHOSEN)
```javascript
async function addToCart(userId, item) {
  let attempts = 0;
  const maxAttempts = 5;

  while (attempts < maxAttempts) {
    try {
      // Read cart with version
      const cart = await db.getCart(userId);
      const currentVersion = cart._version || 0;

      // Modify cart
      cart.items.push(item);
      cart._version = currentVersion + 1;

      // Save only if version matches (no concurrent updates)
      const saved = await db.saveCartWithVersion(userId, cart, currentVersion);

      if (saved) {
        return cart;
      }
      // Version mismatch - another update happened, retry
      attempts++;
      await sleep(50 * attempts); // Exponential backoff

    } catch (error) {
      if (error.code === 'VERSION_CONFLICT' && attempts < maxAttempts - 1) {
        attempts++;
        continue;
      }
      throw error;
    }
  }

  throw new Error(`Failed to update cart after ${maxAttempts} attempts`);
}
```
**Pros**: High throughput, no blocking, automatic retry, scales well
**Cons**: Requires version field, retry logic complexity
**Trade-offs**: Best for high-concurrency scenarios

### Option 3: Queue-Based (Serialize Updates)
```javascript
const cartQueues = new Map();

async function addToCart(userId, item) {
  if (!cartQueues.has(userId)) {
    cartQueues.set(userId, Promise.resolve());
  }

  const queue = cartQueues.get(userId);
  const promise = queue.then(async () => {
    const cart = await db.getCart(userId);
    cart.items.push(item);
    await db.saveCart(userId, cart);
    return cart;
  });

  cartQueues.set(userId, promise);
  return promise;
}
```
**Pros**: Simple logic, no database changes needed
**Cons**: In-memory state, not cluster-safe, memory leaks if not cleaned
**Trade-offs**: Only works for single-server deployments

**Selected Solution**: **Option 2 (Optimistic Locking)** - Best balance of performance and correctness

### Implementation

```javascript
// cart-service.js - FIXED VERSION

/**
 * Add item to user's shopping cart with optimistic locking
 *
 * Uses version-based concurrency control to handle simultaneous updates.
 * Automatically retries on conflicts with exponential backoff.
 *
 * @param {string} userId - User identifier
 * @param {Object} item - Item to add to cart
 * @returns {Promise<Object>} Updated cart
 * @throws {Error} If max retry attempts exceeded
 */
async function addToCart(userId, item) {
  const MAX_ATTEMPTS = 5;
  const BASE_DELAY_MS = 50;

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      // Read current cart state with version
      const cart = await db.getCart(userId);
      const expectedVersion = cart._version || 0;

      // Validate item before adding
      if (!item || !item.id) {
        throw new Error('Invalid item: must have an id');
      }

      // Modify cart state
      cart.items = cart.items || [];
      cart.items.push(item);
      cart.updatedAt = new Date().toISOString();
      cart._version = expectedVersion + 1;

      // Attempt to save with version check
      const result = await db.saveCartWithVersion(userId, cart, expectedVersion);

      if (result.success) {
        logger.info(`Cart updated successfully for user ${userId}`, {
          itemsCount: cart.items.length,
          version: cart._version,
          attempt
        });
        return cart;
      }

      // Version conflict - another update happened concurrently
      logger.warn(`Cart version conflict for user ${userId}`, {
        expectedVersion,
        actualVersion: result.currentVersion,
        attempt
      });

      // Exponential backoff before retry
      if (attempt < MAX_ATTEMPTS) {
        const delay = BASE_DELAY_MS * Math.pow(2, attempt - 1);
        await sleep(delay);
      }

    } catch (error) {
      if (error.code !== 'VERSION_CONFLICT') {
        logger.error(`Unexpected error updating cart for user ${userId}`, error);
        throw error;
      }

      if (attempt === MAX_ATTEMPTS) {
        throw new Error(
          `Failed to update cart after ${MAX_ATTEMPTS} attempts due to conflicts. ` +
          'Please try again.'
        );
      }
    }
  }
}

// database.js - Database layer implementation
async function saveCartWithVersion(userId, cart, expectedVersion) {
  // SQL with optimistic locking
  const result = await db.query(`
    UPDATE carts
    SET
      items = $1,
      updated_at = $2,
      _version = $3
    WHERE
      user_id = $4
      AND _version = $5
    RETURNING *
  `, [
    JSON.stringify(cart.items),
    cart.updatedAt,
    cart._version,
    userId,
    expectedVersion
  ]);

  if (result.rowCount === 0) {
    // No rows updated - version mismatch
    const current = await db.query('SELECT _version FROM carts WHERE user_id = $1', [userId]);
    return {
      success: false,
      currentVersion: current.rows[0]?._version
    };
  }

  return { success: true };
}
```

**Migration for Version Field**:
```sql
-- Add version column to existing carts table
ALTER TABLE carts
ADD COLUMN IF NOT EXISTS _version INTEGER DEFAULT 0;

-- Ensure all existing carts have version 0
UPDATE carts SET _version = 0 WHERE _version IS NULL;

-- Add index for version checks
CREATE INDEX idx_carts_user_version ON carts(user_id, _version);
```

**Why This Approach**:
1. **Non-blocking**: Requests don't wait for locks
2. **High throughput**: Handles concurrent requests efficiently
3. **Automatic retry**: Resolves conflicts transparently
4. **Observable**: Logs conflicts and retries for monitoring
5. **Scalable**: Works across multiple servers/containers

## Phase 4: Validation Testing

### Agent: tester

**Test Suite**:

```javascript
// tests/cart-concurrency.test.js
const { addToCart, getCart, createCart } = require('../cart-service');
const db = require('../database');

describe('Cart Concurrency Tests', () => {
  let userId;

  beforeEach(async () => {
    userId = `test_user_${Date.now()}`;
    await createCart(userId);
  });

  afterEach(async () => {
    await db.deleteCart(userId);
  });

  describe('Concurrent cart updates', () => {
    it('should handle 3 simultaneous additions correctly', async () => {
      // Add 3 items concurrently
      const items = [
        { id: 'item1', name: 'Product 1', price: 10 },
        { id: 'item2', name: 'Product 2', price: 20 },
        { id: 'item3', name: 'Product 3', price: 30 }
      ];

      await Promise.all(
        items.map(item => addToCart(userId, item))
      );

      // Verify all items present
      const cart = await getCart(userId);
      expect(cart.items).toHaveLength(3);
      expect(cart._version).toBe(3);

      // Verify exact items
      const itemIds = cart.items.map(i => i.id).sort();
      expect(itemIds).toEqual(['item1', 'item2', 'item3']);
    });

    it('should handle 10 concurrent additions without data loss', async () => {
      const itemCount = 10;
      const items = Array.from({ length: itemCount }, (_, i) => ({
        id: `item${i}`,
        name: `Product ${i}`
      }));

      await Promise.all(
        items.map(item => addToCart(userId, item))
      );

      const cart = await getCart(userId);
      expect(cart.items).toHaveLength(itemCount);
      expect(cart._version).toBe(itemCount);
    });

    it('should retry on version conflicts', async () => {
      const spy = jest.spyOn(db, 'saveCartWithVersion');

      // Force conflict by updating version manually
      await db.query('UPDATE carts SET _version = 5 WHERE user_id = $1', [userId]);

      await addToCart(userId, { id: 'item1', name: 'Product 1' });

      // Should have retried at least once
      expect(spy).toHaveBeenCalledTimes(2);
    });

    it('should fail after max retry attempts', async () => {
      // Simulate persistent conflicts
      jest.spyOn(db, 'saveCartWithVersion').mockResolvedValue({ success: false });

      await expect(
        addToCart(userId, { id: 'item1', name: 'Product 1' })
      ).rejects.toThrow('Failed to update cart after 5 attempts');
    });
  });

  describe('Version tracking', () => {
    it('should increment version on each update', async () => {
      let cart = await getCart(userId);
      expect(cart._version).toBe(0);

      await addToCart(userId, { id: 'item1' });
      cart = await getCart(userId);
      expect(cart._version).toBe(1);

      await addToCart(userId, { id: 'item2' });
      cart = await getCart(userId);
      expect(cart._version).toBe(2);
    });

    it('should detect stale reads', async () => {
      const cart1 = await getCart(userId);
      const cart2 = await getCart(userId);

      // Both reads get version 0
      expect(cart1._version).toBe(0);
      expect(cart2._version).toBe(0);

      // First update succeeds
      await addToCart(userId, { id: 'item1' });

      // Second update with stale version should retry
      const result = await addToCart(userId, { id: 'item2' });
      expect(result._version).toBe(2);
    });
  });

  describe('Load testing', () => {
    it('should handle 100 concurrent requests', async () => {
      const concurrentRequests = 100;
      const items = Array.from({ length: concurrentRequests }, (_, i) => ({
        id: `item${i}`,
        name: `Product ${i}`
      }));

      const startTime = Date.now();
      await Promise.all(
        items.map(item => addToCart(userId, item))
      );
      const duration = Date.now() - startTime;

      const cart = await getCart(userId);
      expect(cart.items).toHaveLength(concurrentRequests);
      expect(duration).toBeLessThan(5000); // Should complete in < 5 seconds
    });
  });
});
```

**Test Results**:
```
PASS tests/cart-concurrency.test.js
  Cart Concurrency Tests
    Concurrent cart updates
      ✓ should handle 3 simultaneous additions correctly (245ms)
      ✓ should handle 10 concurrent additions without data loss (512ms)
      ✓ should retry on version conflicts (89ms)
      ✓ should fail after max retry attempts (126ms)
    Version tracking
      ✓ should increment version on each update (67ms)
      ✓ should detect stale reads (134ms)
    Load testing
      ✓ should handle 100 concurrent requests (2847ms)

Test Suites: 1 passed, 1 total
Tests:       7 passed, 7 total
Time:        4.123s
```

**Performance Benchmarking**:
```
Before fix:
- Data loss rate: 30% under concurrent load
- Average latency: 45ms
- P99 latency: 120ms

After fix:
- Data loss rate: 0% ✅
- Average latency: 52ms (+7ms overhead)
- P99 latency: 340ms (retries add latency)
- Retry rate: ~8% of requests (acceptable)
```

## Phase 5: Regression Prevention

### Agent: tester + coder

**Permanent Test Integration**:
- Concurrency tests added to CI/CD pipeline
- Load tests run nightly against staging environment
- Performance regression alerts set up

**Monitoring Enhancements**:
```javascript
// metrics.js - Added monitoring for race conditions
metrics.recordCartUpdate({
  userId,
  success: true,
  attempts,
  latency: duration,
  versionConflicts: attempts - 1
});

// Alert on high conflict rates
alerts.defineRule({
  name: 'high_cart_version_conflicts',
  condition: 'cart_version_conflicts > 20% over 5 minutes',
  action: 'notify-team',
  message: 'High rate of cart update conflicts detected'
});
```

**Documentation Updates**:
```markdown
## Cart Service - Concurrency Control

**Implementation**: Optimistic locking with automatic retry

**Version Field**: `_version` column tracks update count
- Initial value: 0
- Incremented on every successful update
- Used to detect concurrent modifications

**Retry Logic**:
- Max 5 attempts with exponential backoff
- Base delay: 50ms
- Backoff: 50ms, 100ms, 200ms, 400ms, 800ms

**Monitoring**:
- Track retry rates (alert if > 20%)
- Log version conflicts for analysis
- Monitor P99 latency (retries increase tail latency)

**Future Improvements**:
- Consider Redis for high-volume users
- Add circuit breaker for persistent conflicts
- Implement conflict resolution strategies
```

**Code Review Checklist Updated**:
```markdown
## Concurrency Review Checklist

- [ ] All database read-modify-write operations use version locking
- [ ] Retry logic implemented for transient conflicts
- [ ] Logging in place for conflict monitoring
- [ ] Tests cover concurrent execution scenarios
- [ ] Performance impact measured and acceptable
```

## Summary

**Issue**: Race condition causing lost cart items
**Root Cause**: Read-modify-write without concurrency control
**Fix**: Optimistic locking with version field and retry logic
**Tests**: 7 comprehensive concurrency tests
**Impact**: 0% data loss, 7ms latency increase, 8% retry rate
**Time to Resolution**: 52 minutes

## Key Takeaways

1. **Identify patterns**: Read-modify-write is a common race condition source
2. **Version everything**: Add version fields to prevent lost updates
3. **Retry with backoff**: Handle conflicts transparently
4. **Test concurrency**: Use `Promise.all()` to simulate concurrent requests
5. **Monitor in production**: Track conflict rates and retry patterns
6. **Document trade-offs**: Optimistic locking increases tail latency

## Related Examples

- [Example 1: Null Pointer](example-1-null-pointer.md) - Basic null safety debugging
- [Example 3: Memory Leak](example-3-memory-leak.md) - Advanced memory leak investigation
