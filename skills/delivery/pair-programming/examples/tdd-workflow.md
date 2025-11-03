# TDD Workflow Example
Complete Test-Driven Development workflow using pair programming

## Scenario: Building a Shopping Cart System

This example demonstrates a full TDD session implementing shopping cart functionality with the Red-Green-Refactor cycle.

---

## Session Setup

```bash
# Start TDD pair programming session
claude-flow pair --start \
  --mode tdd \
  --agent tdd-specialist \
  --focus testing \
  --test-first \
  --coverage 90 \
  --verify
```

**TDD Configuration:**
- Test-first approach (write failing tests before code)
- Red-Green-Refactor cycle enforcement
- Minimum 90% code coverage requirement
- Continuous testing enabled
- Truth score verification on commits

---

## Cycle 1: Add Item to Cart

### 🔴 RED PHASE: Write Failing Test

**AI Pair Partner Guidance:**

```
🤖 TDD SPECIALIST:

Let's start with the most basic functionality: adding an item to the cart.

Test Strategy:
1. Test adding a single item
2. Test adding multiple items
3. Test adding the same item (should increase quantity)
4. Test invalid inputs

Let's begin with test #1. I'll help you structure a good test.
```

**Your Code (test/cart.test.js):**

```javascript
const Cart = require('../src/cart');

describe('ShoppingCart', () => {
  let cart;

  beforeEach(() => {
    cart = new Cart();
  });

  describe('addItem', () => {
    it('should add a single item to empty cart', () => {
      const item = {
        id: 'item1',
        name: 'Laptop',
        price: 999.99,
        quantity: 1
      };

      cart.addItem(item);

      expect(cart.getItemCount()).toBe(1);
      expect(cart.getItems()).toHaveLength(1);
      expect(cart.getItems()[0]).toEqual(item);
    });

    it('should add multiple different items', () => {
      const laptop = { id: 'item1', name: 'Laptop', price: 999.99, quantity: 1 };
      const mouse = { id: 'item2', name: 'Mouse', price: 29.99, quantity: 1 };

      cart.addItem(laptop);
      cart.addItem(mouse);

      expect(cart.getItemCount()).toBe(2);
      expect(cart.getItems()).toHaveLength(2);
    });

    it('should increase quantity when adding existing item', () => {
      const item = { id: 'item1', name: 'Laptop', price: 999.99, quantity: 1 };

      cart.addItem(item);
      cart.addItem(item);

      expect(cart.getItemCount()).toBe(1); // Still 1 unique item
      expect(cart.getItems()[0].quantity).toBe(2); // But quantity is 2
    });

    it('should throw error for invalid item', () => {
      expect(() => cart.addItem(null)).toThrow('Invalid item');
      expect(() => cart.addItem({})).toThrow('Item must have id, name, and price');
      expect(() => cart.addItem({ id: '1', name: 'Item' })).toThrow('Item must have id, name, and price');
    });
  });
});
```

**Run Tests:**

```bash
npm test
```

**Test Results (Expected Failures):**

```
 FAIL  test/cart.test.js
  ShoppingCart
    addItem
      ✗ should add a single item to empty cart (2ms)
      ✗ should add multiple different items (1ms)
      ✗ should increase quantity when adding existing item (1ms)
      ✗ should throw error for invalid item (1ms)

  ● ShoppingCart › addItem › should add a single item to empty cart

    TypeError: Cart is not a constructor

Test Suites: 1 failed, 1 total
Tests:       4 failed, 4 total
```

**AI Feedback:**

```
✅ RED PHASE SUCCESSFUL

All tests are failing as expected. This is GOOD in TDD!

Your tests are well-structured:
✓ Clear test descriptions
✓ Testing multiple scenarios
✓ Edge cases included (invalid inputs)
✓ Good use of beforeEach for setup

Now let's move to the GREEN phase and write the minimal code
to make these tests pass.
```

---

### 🟢 GREEN PHASE: Make Tests Pass

**AI Pair Partner:**

```
Now I'll implement the Cart class with just enough code to pass the tests.
No more, no less. This is the TDD discipline.
```

**AI Generates (src/cart.js):**

```javascript
class Cart {
  constructor() {
    this.items = [];
  }

  addItem(item) {
    // Validation
    if (!item) {
      throw new Error('Invalid item');
    }

    if (!item.id || !item.name || item.price === undefined) {
      throw new Error('Item must have id, name, and price');
    }

    // Check if item already exists
    const existingItem = this.items.find(i => i.id === item.id);

    if (existingItem) {
      // Item exists, increase quantity
      existingItem.quantity += item.quantity || 1;
    } else {
      // New item, add to cart
      this.items.push({
        ...item,
        quantity: item.quantity || 1
      });
    }
  }

  getItems() {
    return this.items;
  }

  getItemCount() {
    return this.items.length;
  }
}

module.exports = Cart;
```

**Run Tests:**

```bash
npm test
```

**Test Results:**

```
 PASS  test/cart.test.js
  ShoppingCart
    addItem
      ✓ should add a single item to empty cart (3ms)
      ✓ should add multiple different items (2ms)
      ✓ should increase quantity when adding existing item (2ms)
      ✓ should throw error for invalid item (2ms)

Test Suites: 1 passed, 1 total
Tests:       4 passed, 4 total
Coverage:    100% lines, 100% branches
```

**AI Feedback:**

```
✅ GREEN PHASE SUCCESSFUL

All tests are now passing! Coverage: 100%

The implementation is minimal but sufficient. We didn't add
any features that aren't tested. This is TDD discipline.

Ready for REFACTOR phase.
```

---

### 🔧 REFACTOR PHASE: Improve Code Quality

**Your Turn as Navigator:**

```
/review --complexity
```

**Complexity Analysis:**

```
📊 COMPLEXITY ANALYSIS:

Cart.addItem():
- Cyclomatic Complexity: 4
- Lines: 18
- Parameters: 1
- Nesting Depth: 2

Status: ✅ GOOD (within thresholds)

Suggestions for refactoring:
1. Extract validation into separate method
2. Extract item merging logic into separate method
3. Consider using a Map instead of array for O(1) lookup

These changes won't affect functionality but improve maintainability.
```

**You Refactor:**

```javascript
class Cart {
  constructor() {
    this.items = [];
  }

  addItem(item) {
    this._validateItem(item);

    const existingItem = this._findItem(item.id);

    if (existingItem) {
      this._increaseQuantity(existingItem, item.quantity || 1);
    } else {
      this._addNewItem(item);
    }
  }

  _validateItem(item) {
    if (!item) {
      throw new Error('Invalid item');
    }

    if (!item.id || !item.name || item.price === undefined) {
      throw new Error('Item must have id, name, and price');
    }
  }

  _findItem(itemId) {
    return this.items.find(i => i.id === itemId);
  }

  _increaseQuantity(item, quantity) {
    item.quantity += quantity;
  }

  _addNewItem(item) {
    this.items.push({
      ...item,
      quantity: item.quantity || 1
    });
  }

  getItems() {
    return this.items;
  }

  getItemCount() {
    return this.items.length;
  }
}

module.exports = Cart;
```

**Run Tests (Verify Refactoring):**

```bash
npm test
```

**Test Results:**

```
 PASS  test/cart.test.js
  ShoppingCart
    addItem
      ✓ should add a single item to empty cart (3ms)
      ✓ should add multiple different items (2ms)
      ✓ should increase quantity when adding existing item (1ms)
      ✓ should throw error for invalid item (2ms)

Test Suites: 1 passed, 1 total
Tests:       4 passed, 4 total
Coverage:    100% lines, 100% branches

Truth Score: 0.985 ✅
```

**AI Feedback:**

```
✅ REFACTOR PHASE SUCCESSFUL

Tests still passing after refactoring!

Improvements:
✓ Better separation of concerns
✓ Single Responsibility Principle
✓ More readable code
✓ Easier to test individual pieces

Truth Score: 0.985 (Excellent!)

Ready for Cycle 2?
```

---

## Cycle 2: Calculate Cart Total

### 🔴 RED: Write Test for Total Calculation

```javascript
describe('calculateTotal', () => {
  it('should return 0 for empty cart', () => {
    expect(cart.calculateTotal()).toBe(0);
  });

  it('should calculate total for single item', () => {
    cart.addItem({ id: '1', name: 'Item', price: 29.99, quantity: 1 });
    expect(cart.calculateTotal()).toBe(29.99);
  });

  it('should calculate total with quantities', () => {
    cart.addItem({ id: '1', name: 'Item', price: 10.00, quantity: 3 });
    expect(cart.calculateTotal()).toBe(30.00);
  });

  it('should calculate total for multiple items', () => {
    cart.addItem({ id: '1', name: 'Laptop', price: 999.99, quantity: 1 });
    cart.addItem({ id: '2', name: 'Mouse', price: 29.99, quantity: 2 });
    expect(cart.calculateTotal()).toBe(1059.97);
  });

  it('should handle decimal precision correctly', () => {
    cart.addItem({ id: '1', name: 'Item', price: 10.99, quantity: 3 });
    expect(cart.calculateTotal()).toBeCloseTo(32.97, 2);
  });
});
```

**Run Tests:**

```
 FAIL  test/cart.test.js
  ShoppingCart
    calculateTotal
      ✗ should return 0 for empty cart
      ✗ should calculate total for single item
      ...

  ● ShoppingCart › calculateTotal › should return 0 for empty cart

    TypeError: cart.calculateTotal is not a function
```

### 🟢 GREEN: Implement Total Calculation

```javascript
class Cart {
  // ... previous methods ...

  calculateTotal() {
    return this.items.reduce((total, item) => {
      return total + (item.price * item.quantity);
    }, 0);
  }
}
```

**Tests Pass:**

```
 PASS  test/cart.test.js
  ✓ All 9 tests passing
  Coverage: 100%
```

### 🔧 REFACTOR: Add Precision Handling

```javascript
calculateTotal() {
  const total = this.items.reduce((sum, item) => {
    return sum + (item.price * item.quantity);
  }, 0);

  // Round to 2 decimal places to avoid floating point errors
  return Math.round(total * 100) / 100;
}
```

---

## Cycle 3: Remove Item from Cart

### 🔴 RED: Write Tests

```javascript
describe('removeItem', () => {
  it('should remove item from cart', () => {
    cart.addItem({ id: '1', name: 'Item', price: 10, quantity: 1 });
    cart.removeItem('1');

    expect(cart.getItemCount()).toBe(0);
    expect(cart.calculateTotal()).toBe(0);
  });

  it('should decrease quantity if multiple items', () => {
    cart.addItem({ id: '1', name: 'Item', price: 10, quantity: 3 });
    cart.removeItem('1', 1); // Remove 1

    const item = cart.getItems()[0];
    expect(item.quantity).toBe(2);
  });

  it('should remove item completely if quantity becomes 0', () => {
    cart.addItem({ id: '1', name: 'Item', price: 10, quantity: 2 });
    cart.removeItem('1', 2);

    expect(cart.getItemCount()).toBe(0);
  });

  it('should throw error for non-existent item', () => {
    expect(() => cart.removeItem('nonexistent')).toThrow('Item not found');
  });
});
```

### 🟢 GREEN: Implement Removal

```javascript
removeItem(itemId, quantity = null) {
  const itemIndex = this.items.findIndex(i => i.id === itemId);

  if (itemIndex === -1) {
    throw new Error('Item not found');
  }

  if (quantity === null || this.items[itemIndex].quantity <= quantity) {
    // Remove completely
    this.items.splice(itemIndex, 1);
  } else {
    // Decrease quantity
    this.items[itemIndex].quantity -= quantity;
  }
}
```

### 🔧 REFACTOR: Extract Methods

```javascript
removeItem(itemId, quantity = null) {
  const itemIndex = this._findItemIndex(itemId);

  if (itemIndex === -1) {
    throw new Error('Item not found');
  }

  if (this._shouldRemoveCompletely(itemIndex, quantity)) {
    this._removeItemCompletely(itemIndex);
  } else {
    this._decreaseItemQuantity(itemIndex, quantity);
  }
}

_findItemIndex(itemId) {
  return this.items.findIndex(i => i.id === itemId);
}

_shouldRemoveCompletely(itemIndex, quantity) {
  return quantity === null || this.items[itemIndex].quantity <= quantity;
}

_removeItemCompletely(itemIndex) {
  this.items.splice(itemIndex, 1);
}

_decreaseItemQuantity(itemIndex, quantity) {
  this.items[itemIndex].quantity -= quantity;
}
```

---

## Final Session Summary

### TDD Metrics

```json
{
  "tdd": {
    "cycles": 3,
    "redPhaseDuration": 420,
    "greenPhaseDuration": 380,
    "refactorPhaseDuration": 340,
    "totalDuration": 1140
  },

  "testing": {
    "totalTests": 13,
    "passing": 13,
    "failing": 0,
    "coverage": {
      "lines": 100,
      "branches": 100,
      "functions": 100,
      "statements": 100
    }
  },

  "quality": {
    "avgComplexity": 2.3,
    "maxComplexity": 4,
    "avgFunctionLength": 8,
    "truthScore": 0.987
  }
}
```

### Complete Cart Implementation

**Final Code (src/cart.js):**

```javascript
class Cart {
  constructor() {
    this.items = [];
  }

  // Add item to cart
  addItem(item) {
    this._validateItem(item);
    const existingItem = this._findItem(item.id);

    if (existingItem) {
      this._increaseQuantity(existingItem, item.quantity || 1);
    } else {
      this._addNewItem(item);
    }
  }

  // Remove item from cart
  removeItem(itemId, quantity = null) {
    const itemIndex = this._findItemIndex(itemId);

    if (itemIndex === -1) {
      throw new Error('Item not found');
    }

    if (this._shouldRemoveCompletely(itemIndex, quantity)) {
      this._removeItemCompletely(itemIndex);
    } else {
      this._decreaseItemQuantity(itemIndex, quantity);
    }
  }

  // Calculate cart total
  calculateTotal() {
    const total = this.items.reduce((sum, item) => {
      return sum + (item.price * item.quantity);
    }, 0);

    return Math.round(total * 100) / 100;
  }

  // Get all items
  getItems() {
    return this.items;
  }

  // Get unique item count
  getItemCount() {
    return this.items.length;
  }

  // Get total quantity of all items
  getTotalQuantity() {
    return this.items.reduce((sum, item) => sum + item.quantity, 0);
  }

  // Clear cart
  clear() {
    this.items = [];
  }

  // Private methods
  _validateItem(item) {
    if (!item) {
      throw new Error('Invalid item');
    }
    if (!item.id || !item.name || item.price === undefined) {
      throw new Error('Item must have id, name, and price');
    }
  }

  _findItem(itemId) {
    return this.items.find(i => i.id === itemId);
  }

  _findItemIndex(itemId) {
    return this.items.findIndex(i => i.id === itemId);
  }

  _increaseQuantity(item, quantity) {
    item.quantity += quantity;
  }

  _addNewItem(item) {
    this.items.push({
      ...item,
      quantity: item.quantity || 1
    });
  }

  _shouldRemoveCompletely(itemIndex, quantity) {
    return quantity === null || this.items[itemIndex].quantity <= quantity;
  }

  _removeItemCompletely(itemIndex) {
    this.items.splice(itemIndex, 1);
  }

  _decreaseItemQuantity(itemIndex, quantity) {
    this.items[itemIndex].quantity -= quantity;
  }
}

module.exports = Cart;
```

### TDD Benefits Demonstrated

✅ **Comprehensive Test Coverage**: 100% coverage achieved
✅ **Better Design**: Tests drove clean architecture
✅ **No Over-Engineering**: Only built what was needed
✅ **Continuous Verification**: Tests ensure correctness
✅ **Easy Refactoring**: Safe to improve with test safety net
✅ **Documentation**: Tests serve as usage examples
✅ **High Confidence**: 13 tests all passing

### Key TDD Lessons

1. **Always Red First**: Write failing test before any code
2. **Minimal Green**: Write only enough code to pass
3. **Refactor Safely**: Tests protect during improvement
4. **Test Behavior**: Focus on what, not how
5. **Small Cycles**: Keep cycles short (10-15 minutes)
6. **Commit Often**: After each successful refactor

---

**Session Complete: Shopping Cart with TDD** ✅
- 3 TDD Cycles completed
- 13 tests written
- 100% coverage achieved
- Clean, maintainable code
- Truth Score: 0.987
