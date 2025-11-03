# Example 2: Multi-Step Intent - "Research, Recommend, and Implement"

## Scenario Overview

A user submits a complex multi-step request: **"Evaluate different state management solutions for React, recommend the best one for our e-commerce app, and show me how to set it up"**. This example demonstrates Intent Analyzer's ability to recognize and structure responses for multi-phase workflows with sequential intent dependencies.

## Initial User Request

```
User: Evaluate different state management solutions for React, recommend the best one for our e-commerce app, and show me how to set it up
```

## Phase 1: Deep Analysis (Internal Processing)

### Intent Archaeology

**What is explicitly stated?**
- Three distinct phases:
  1. **Evaluate** (analytical): Compare state management solutions
  2. **Recommend** (decision): Pick best one for e-commerce
  3. **Show setup** (technical): Implementation guide

**What is implied?**
- User making architecture decision for production app
- E-commerce context suggests specific requirements (shopping cart, user sessions, product catalog)
- "Our app" suggests team context (not solo learning project)
- Sequential dependency: evaluation → recommendation → implementation

**What domain knowledge is assumed?**
- User knows React (specific to React ecosystem)
- Aware state management is a solved problem (wants comparison, not explanation of concept)
- Has context for why this matters (building e-commerce)

**What expertise level does phrasing suggest?**
- **Intermediate-to-Advanced React developer:**
  - Knows "state management solutions" (not "how to manage state")
  - Specific terminology ("e-commerce app")
  - Wants evaluation (not tutorial)
- NOT beginner (would ask "how to manage state in React")

### Goal Extrapolation

**Immediate goals:**
1. **Analytical (Phase 1):** Understand options and tradeoffs
2. **Decision (Phase 2):** Make informed choice for specific use case
3. **Technical (Phase 3):** Learn implementation details

**Higher-order goals:**
- Make correct architecture decision early (costly to change later)
- Avoid analysis paralysis (wants recommendation, not just comparison)
- Get to implementation quickly (wants setup guide)

**Why this structure?**
User wants **decision support** (evaluate + recommend) **and** **implementation guidance** (setup). This is a classic "I need to decide AND do it" pattern.

### Constraint Detection

**Stated requirements:**
- Technology: React (explicit)
- Domain: E-commerce (explicit)
- Scope: State management solutions (explicit)

**Contextual constraints:**
- Production context ("our app" suggests team/business context)
- E-commerce requirements:
  - Shopping cart persistence
  - User authentication state
  - Product catalog data
  - Possibly inventory, checkout flow
- Likely needs scalability, performance, developer experience

**Domain requirements:**
- Must work with React ecosystem
- E-commerce-appropriate (good for complex, shared state)
- Team-friendly (not overly complex for collaboration)

### Pattern Recognition

Run **intent-classifier.py**:

```bash
$ python intent-classifier.py "Evaluate different state management solutions for React, recommend the best one for our e-commerce app, and show me how to set it up"

{
  "primary_category": "analytical",
  "confidence": 0.45,
  "categories": {
    "analytical": 0.45,
    "decision": 0.35,
    "technical": 0.20
  },
  "multi_intent": true,
  "ambiguous": false,
  "requires_clarification": false,
  "signals": [
    "evaluate (verb)",
    "solutions (noun)",
    "recommend (verb)",
    "show (verb)",
    "setup (noun)"
  ],
  "reasoning": "Multiple intent categories detected: analytical, decision, technical. Low-moderate confidence (0.45) across three distinct phases. Three-phase request: analysis → decision → implementation"
}
```

### Ambiguity Assessment

**Confidence levels:**
- Analytical: 45%
- Decision: 35%
- Technical: 20%

**Overall:** Low-moderate confidence, but **no ambiguity about structure**

**Decision:**
- Multi-intent is CLEAR (explicit three-phase structure)
- No clarification needed (all three intents are unambiguous)
- **Path chosen:** Proceed directly with three-phase response

**Why no clarification despite low confidence?**
- Low confidence is from **equal distribution** across 3 intents, not ambiguity
- User explicitly structured request in sequence
- All three phases are clear and well-defined
- Clarification would be redundant

## Phase 2: Decision Point

**Path chosen:** Path A - High Confidence Interpretation (proceed directly)

**Reasoning:**
- Multi-phase structure is explicit and clear
- Each phase has clear intent (evaluate → recommend → implement)
- Sequential dependency is logical
- E-commerce context provides sufficient requirements

**No clarification needed, but will acknowledge interpretation:**
- "I'll provide a three-phase response: evaluation, recommendation, implementation"
- Surface key assumptions about e-commerce requirements

## Phase 3: No Clarification (Skip to Phase 4)

Since intent is clear, skip clarification and proceed directly to interpretation reconstruction.

## Phase 4: Interpretation Reconstruction

### Intent Synthesis

**Three-phase workflow identified:**

**Phase 1 - Analytical:**
- Compare major React state management solutions
- Evaluate against e-commerce criteria
- Present tradeoffs objectively

**Phase 2 - Decision:**
- Recommend specific solution
- Justify choice for e-commerce context
- Explain why this over alternatives

**Phase 3 - Technical:**
- Setup instructions for recommended solution
- E-commerce-specific implementation patterns
- Code examples with cart, products, user state

### Assumption Surfacing

```
I'm interpreting this as a request for:

1. Comparison of React state management options (Redux, MobX, Zustand, Recoil, Context API)
2. Recommendation tailored to e-commerce requirements (shopping cart, user sessions, complex state)
3. Setup guide for the recommended solution with e-commerce examples

I'm assuming:
- You're building a production e-commerce application
- You need to handle complex shared state (cart, products, user)
- You want a solution that scales and has good DX
- Your team has React experience but hasn't committed to a state solution yet

Let me know if I should adjust focus on any of these areas!
```

### Approach Signaling

```
I'll provide:
1. **Evaluation**: Compare 5 state management solutions against e-commerce criteria
2. **Recommendation**: Best choice for e-commerce (with justification)
3. **Setup Guide**: Step-by-step implementation with shopping cart example
```

## Final Response (Multi-Phase Structure)

```markdown
# React State Management for E-Commerce: Evaluation, Recommendation & Setup

I'll address this in three phases: evaluation, recommendation, and implementation.

---

## Phase 1: Evaluation - State Management Solutions

### Solutions Compared

| Solution | Boilerplate | Learning Curve | DevTools | Performance | Async Support | Community |
|----------|-------------|----------------|----------|-------------|---------------|-----------|
| **Redux Toolkit** | Low (modern API) | Moderate | Excellent | Excellent | Built-in (RTK Query) | Huge |
| **Zustand** | Very Low | Low | Good | Excellent | Simple | Growing |
| **Recoil** | Low | Moderate | Good | Good | Excellent | Medium |
| **MobX** | Low | Low-Moderate | Good | Excellent | Good | Large |
| **Context API** | Very Low | Low | Built-in | Poor (for large apps) | Manual | Built-in |

### E-Commerce Criteria Analysis

**1. Complex Shared State** (shopping cart, products, user, inventory)
- ✅ **Best**: Redux Toolkit, MobX (designed for complex state)
- ⚠️ **Good**: Zustand, Recoil (simple but scalable)
- ❌ **Poor**: Context API (re-render issues at scale)

**2. Async Data Fetching** (API calls for products, orders, user data)
- ✅ **Best**: Redux Toolkit (RTK Query built-in)
- ⚠️ **Good**: Recoil (async selectors), Zustand (middleware)
- ❌ **Manual**: MobX, Context API

**3. Developer Experience**
- ✅ **Best**: Zustand (simplest), Redux Toolkit (modern, opinionated)
- ⚠️ **Good**: Recoil, MobX
- ❌ **Verbose**: Context API (for complex state)

**4. Team Collaboration**
- ✅ **Best**: Redux Toolkit (strict patterns, predictable)
- ⚠️ **Good**: Recoil, MobX (less strict but clear)
- ❌ **Poor**: Context API (no enforced patterns)

**5. Performance at Scale**
- ✅ **Best**: Zustand, MobX (fine-grained reactivity)
- ⚠️ **Good**: Redux Toolkit (with proper selectors)
- ❌ **Poor**: Context API (broad re-renders)

**6. Persistence** (cart across sessions)
- ✅ **Best**: Redux Toolkit (redux-persist ecosystem)
- ⚠️ **Manual**: Zustand, Recoil, MobX (custom solutions)
- ❌ **Manual**: Context API

---

## Phase 2: Recommendation - Best for E-Commerce

### Recommendation: **Redux Toolkit (RTK)**

**Why Redux Toolkit for E-Commerce:**

1. **Complex State Management** ✅
   - Shopping cart with multiple items, quantities, variants
   - Product catalog with filters, sorting, pagination
   - User authentication, preferences, order history
   - RTK handles this complexity elegantly

2. **Async Data Built-In** ✅
   - RTK Query eliminates need for separate data fetching
   - Automatic caching, invalidation, refetching
   - Perfect for product API, user API, order API

3. **Developer Experience** ✅
   - Modern, opinionated API (no Redux boilerplate)
   - Excellent DevTools (time-travel debugging)
   - TypeScript support out-of-the-box

4. **Team Collaboration** ✅
   - Strict patterns reduce "creative freedom" bugs
   - Predictable state updates
   - Large community = easy onboarding

5. **Persistence** ✅
   - `redux-persist` for cart across sessions
   - Server-side rendering compatible

6. **Ecosystem** ✅
   - Payment integrations (Stripe, PayPal) have Redux examples
   - Analytics, A/B testing tools integrate well

**When NOT to use Redux Toolkit:**
- Simple apps with minimal state (Zustand better)
- Rapid prototyping (Zustand faster to start)
- Very small teams who want maximum flexibility (Zustand, Recoil)

**For e-commerce specifically, RTK's structure and ecosystem win.**

---

## Phase 3: Setup Guide - Implementing Redux Toolkit

### Step 1: Install Dependencies

```bash
npm install @reduxjs/toolkit react-redux
npm install redux-persist  # For cart persistence
```

### Step 2: Create Store Structure

```
src/
├── app/
│   └── store.js           # Redux store configuration
├── features/
│   ├── cart/
│   │   └── cartSlice.js   # Shopping cart logic
│   ├── products/
│   │   ├── productsSlice.js
│   │   └── productsAPI.js  # RTK Query API
│   └── user/
│       └── userSlice.js    # User authentication
```

### Step 3: Configure Store

**`src/app/store.js`:**
```javascript
import { configureStore } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage';
import cartReducer from '../features/cart/cartSlice';
import { productsApi } from '../features/products/productsAPI';
import userReducer from '../features/user/userSlice';

// Persist cart across sessions
const cartPersistConfig = {
  key: 'cart',
  storage,
};

const persistedCartReducer = persistReducer(cartPersistConfig, cartReducer);

export const store = configureStore({
  reducer: {
    cart: persistedCartReducer,
    user: userReducer,
    [productsApi.reducerPath]: productsApi.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }).concat(productsApi.middleware),
});

export const persistor = persistStore(store);
```

### Step 4: Shopping Cart Slice

**`src/features/cart/cartSlice.js`:**
```javascript
import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  items: [],
  totalQuantity: 0,
  totalAmount: 0,
};

const cartSlice = createSlice({
  name: 'cart',
  initialState,
  reducers: {
    addToCart: (state, action) => {
      const { id, name, price, quantity = 1 } = action.payload;
      const existingItem = state.items.find(item => item.id === id);

      if (existingItem) {
        existingItem.quantity += quantity;
      } else {
        state.items.push({ id, name, price, quantity });
      }

      state.totalQuantity += quantity;
      state.totalAmount += price * quantity;
    },

    removeFromCart: (state, action) => {
      const id = action.payload;
      const existingItem = state.items.find(item => item.id === id);

      if (existingItem) {
        state.totalQuantity -= existingItem.quantity;
        state.totalAmount -= existingItem.price * existingItem.quantity;
        state.items = state.items.filter(item => item.id !== id);
      }
    },

    updateQuantity: (state, action) => {
      const { id, quantity } = action.payload;
      const existingItem = state.items.find(item => item.id === id);

      if (existingItem) {
        const quantityDiff = quantity - existingItem.quantity;
        existingItem.quantity = quantity;
        state.totalQuantity += quantityDiff;
        state.totalAmount += existingItem.price * quantityDiff;
      }
    },

    clearCart: (state) => {
      state.items = [];
      state.totalQuantity = 0;
      state.totalAmount = 0;
    },
  },
});

export const { addToCart, removeFromCart, updateQuantity, clearCart } = cartSlice.actions;
export default cartSlice.reducer;

// Selectors
export const selectCartItems = (state) => state.cart.items;
export const selectCartTotal = (state) => state.cart.totalAmount;
export const selectCartQuantity = (state) => state.cart.totalQuantity;
```

### Step 5: Products API (RTK Query)

**`src/features/products/productsAPI.js`:**
```javascript
import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export const productsApi = createApi({
  reducerPath: 'productsApi',
  baseQuery: fetchBaseQuery({ baseUrl: '/api' }),
  tagTypes: ['Products'],
  endpoints: (builder) => ({
    getProducts: builder.query({
      query: ({ category, page = 1, limit = 20 }) =>
        `/products?category=${category}&page=${page}&limit=${limit}`,
      providesTags: ['Products'],
    }),

    getProductById: builder.query({
      query: (id) => `/products/${id}`,
      providesTags: (result, error, id) => [{ type: 'Products', id }],
    }),

    searchProducts: builder.query({
      query: (searchTerm) => `/products/search?q=${searchTerm}`,
    }),
  }),
});

export const {
  useGetProductsQuery,
  useGetProductByIdQuery,
  useSearchProductsQuery,
} = productsApi;
```

### Step 6: Connect React App

**`src/index.js`:**
```javascript
import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { store, persistor } from './app/store';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <Provider store={store}>
    <PersistGate loading={null} persistor={persistor}>
      <App />
    </PersistGate>
  </Provider>
);
```

### Step 7: Use in Components

**Product List Component:**
```javascript
import React from 'react';
import { useDispatch } from 'react-redux';
import { useGetProductsQuery } from './features/products/productsAPI';
import { addToCart } from './features/cart/cartSlice';

function ProductList() {
  const dispatch = useDispatch();
  const { data: products, isLoading, error } = useGetProductsQuery({
    category: 'electronics',
    page: 1,
    limit: 20,
  });

  if (isLoading) return <div>Loading products...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div className="product-grid">
      {products.map((product) => (
        <div key={product.id} className="product-card">
          <h3>{product.name}</h3>
          <p>${product.price}</p>
          <button
            onClick={() => dispatch(addToCart({
              id: product.id,
              name: product.name,
              price: product.price,
              quantity: 1,
            }))}
          >
            Add to Cart
          </button>
        </div>
      ))}
    </div>
  );
}
```

**Shopping Cart Component:**
```javascript
import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {
  selectCartItems,
  selectCartTotal,
  removeFromCart,
  updateQuantity,
} from './features/cart/cartSlice';

function ShoppingCart() {
  const cartItems = useSelector(selectCartItems);
  const cartTotal = useSelector(selectCartTotal);
  const dispatch = useDispatch();

  return (
    <div className="cart">
      <h2>Shopping Cart</h2>
      {cartItems.length === 0 ? (
        <p>Your cart is empty</p>
      ) : (
        <>
          {cartItems.map((item) => (
            <div key={item.id} className="cart-item">
              <h4>{item.name}</h4>
              <p>${item.price}</p>
              <input
                type="number"
                value={item.quantity}
                onChange={(e) => dispatch(updateQuantity({
                  id: item.id,
                  quantity: parseInt(e.target.value, 10),
                }))}
                min="1"
              />
              <button onClick={() => dispatch(removeFromCart(item.id))}>
                Remove
              </button>
            </div>
          ))}
          <div className="cart-total">
            <h3>Total: ${cartTotal.toFixed(2)}</h3>
            <button className="checkout-btn">Proceed to Checkout</button>
          </div>
        </>
      )}
    </div>
  );
}
```

### Step 8: DevTools Setup

Redux Toolkit includes Redux DevTools by default. Install browser extension:

- Chrome: Redux DevTools Extension
- Firefox: Redux DevTools Extension

Features you get:
- Time-travel debugging
- Action history
- State diff viewer
- Action replay

### Step 9: Testing

**Test Cart Slice:**
```javascript
import cartReducer, { addToCart, removeFromCart } from './cartSlice';

describe('cart reducer', () => {
  const initialState = {
    items: [],
    totalQuantity: 0,
    totalAmount: 0,
  };

  it('should handle addToCart', () => {
    const actual = cartReducer(initialState, addToCart({
      id: 1,
      name: 'Test Product',
      price: 29.99,
      quantity: 1,
    }));

    expect(actual.items).toHaveLength(1);
    expect(actual.totalQuantity).toBe(1);
    expect(actual.totalAmount).toBe(29.99);
  });

  it('should handle removeFromCart', () => {
    const stateWithItem = {
      items: [{ id: 1, name: 'Test', price: 29.99, quantity: 1 }],
      totalQuantity: 1,
      totalAmount: 29.99,
    };

    const actual = cartReducer(stateWithItem, removeFromCart(1));

    expect(actual.items).toHaveLength(0);
    expect(actual.totalQuantity).toBe(0);
    expect(actual.totalAmount).toBe(0);
  });
});
```

---

## Next Steps

1. **Implement User Authentication**: Create `userSlice` for login/logout
2. **Add Product Filters**: Extend `productsApi` with filter endpoints
3. **Checkout Flow**: Create order slice and payment integration
4. **Optimize Performance**: Use `reselect` for complex selectors
5. **Analytics**: Add Redux middleware for tracking cart events

## Resources

- Redux Toolkit Docs: https://redux-toolkit.js.org/
- RTK Query Tutorial: https://redux-toolkit.js.org/tutorials/rtk-query
- E-Commerce Example: https://github.com/reduxjs/redux-toolkit/tree/master/examples

Let me know if you need help with any specific part of the implementation!
```

## Key Outcomes

### What Made This Multi-Phase Response Successful?

1. **Clear Phase Recognition:**
   - Detected three distinct intents (analytical → decision → technical)
   - Understood sequential dependency (must evaluate before recommending)
   - Structured response to match request structure

2. **No Over-Clarification:**
   - Recognized low confidence was from intent distribution, not ambiguity
   - Didn't ask redundant questions when intent was clear
   - Proceeded confidently with explicit three-phase structure

3. **E-Commerce Context Integration:**
   - Evaluation criteria tailored to e-commerce needs
   - Recommendation justified for specific use case
   - Implementation examples use shopping cart, products, user state

4. **Progressive Depth:**
   - Phase 1: High-level comparison (helps understand options)
   - Phase 2: Specific recommendation with justification (makes decision)
   - Phase 3: Detailed implementation (enables action)

5. **Production-Ready Focus:**
   - Recognized "our app" implies team/business context
   - Included persistence, testing, DevTools setup
   - Addressed scalability and team collaboration

### Metrics

- **Phases delivered:** 3/3 (evaluation, recommendation, implementation)
- **Clarification questions:** 0 (not needed)
- **Response coherence:** Sequential workflow maintained throughout
- **Context adaptation:** E-commerce focus in all three phases

## Lessons Learned

1. **Multi-intent ≠ Ambiguous:** Multiple clear intents don't require clarification
2. **Respect request structure:** User's sequencing signals their thought process
3. **Context is key:** "E-commerce" drastically changes evaluation criteria
4. **Complete the loop:** All three phases must be addressed for completeness
5. **Progressive detail:** Early phases inform later phases (evaluation → recommendation → implementation)

## Comparison: What Would Have Been WRONG

**Anti-Pattern 1: Ignoring Multi-Intent**
```
❌ User: "Evaluate, recommend, and show setup"
❌ Response: [Only provides evaluation, stops there]
```

**Anti-Pattern 2: Over-Clarification**
```
❌ User: "Evaluate, recommend, and show setup for e-commerce"
❌ Response: "What do you mean by evaluate? What specific aspects of e-commerce?"
[User already specified everything clearly!]
```

**Anti-Pattern 3: Wrong Order**
```
❌ User: "Evaluate, recommend, and show setup"
❌ Response: [Gives setup first, then recommendation, then evaluation]
[Backwards logic - can't set up without choosing, can't choose without evaluating]
```

**Anti-Pattern 4: Generic (Ignoring Context)**
```
❌ User: "For our e-commerce app"
❌ Response: [Generic React state comparison, no e-commerce specifics]
```

## Success Pattern

✅ **Respect structure** → ✅ **Tailor to context** → ✅ **Complete all phases** → ✅ **Maintain coherence**

This is how multi-step intents should be handled!
