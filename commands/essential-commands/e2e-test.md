---
name: e2e-test
category: testing
version: 1.0.0
---

# /e2e-test

End-to-end testing for critical user journeys and business workflows.

## Usage
```bash
/e2e-test [suite] [options]
```

## Parameters
- `suite` - Test suite: all|critical|smoke|full (default: critical)
- `--browser` - Browser engine: chromium|firefox|webkit|all (default: chromium)
- `--headless` - Headless mode (default: true)
- `--record-video` - Record test execution (default: on-failure)
- `--screenshot` - Screenshot on failure (default: true)
- `--parallel` - Run tests in parallel (default: false)
- `--retry` - Retry failed tests (default: 2)
- `--trace` - Enable Playwright trace (default: on-failure)

## What It Does

**Complete User Journey Testing**:
1. 🌐 **Browser Automation**: Chromium, Firefox, WebKit
2. 👤 **User Flows**: Registration, login, checkout
3. 📱 **Multi-Device**: Desktop, tablet, mobile viewports
4. 🎨 **Visual Validation**: Screenshot comparison
5. 🔗 **API Interaction**: Network request validation
6. 📊 **Performance**: Page load, interaction timing
7. 🎥 **Video Recording**: Test execution replay
8. 🐛 **Debugging**: Traces, screenshots, logs

**Critical User Journeys**:
- **Authentication**: Sign up, login, logout, password reset
- **E-commerce**: Browse, search, add to cart, checkout, payment
- **Content Management**: Create, edit, delete, publish
- **User Profile**: View, edit profile, upload avatar
- **Notifications**: Real-time updates, email, push

## Examples

```bash
# Run critical E2E tests
/e2e-test critical

# Full test suite across all browsers
/e2e-test full --browser all

# Smoke tests with video recording
/e2e-test smoke --record-video always --headless false

# Parallel execution with retries
/e2e-test --parallel true --retry 3

# Debug mode with traces
/e2e-test --headless false --trace always

# Mobile viewport testing
/e2e-test --browser webkit --device "iPhone 12"
```

## Output

```
🎭 End-to-End Test Suite Started

Configuration:
  Suite: critical
  Browser: chromium
  Headless: true
  Video: on-failure
  Screenshot: on-failure
  Retry: 2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Critical User Journey: Authentication
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [01] User Registration Flow
       ✅ Navigate to /signup
       ✅ Fill registration form
          Email: test-user-1234@example.com
          Password: ••••••••
          Name: Test User
       ✅ Submit form
       ✅ Verify success message displayed
       ✅ Check confirmation email sent
       ✅ User redirected to dashboard
       Duration: 3.2s

  [02] User Login Flow
       ✅ Navigate to /login
       ✅ Enter credentials
       ✅ Submit login form
       ✅ Verify JWT token in localStorage
       ✅ Dashboard loaded successfully
       ✅ User avatar displayed
       Duration: 2.1s

  [03] Logout Flow
       ✅ Click logout button
       ✅ Confirm logout modal
       ✅ Token removed from localStorage
       ✅ Redirected to home page
       ✅ Protected routes inaccessible
       Duration: 1.4s

  [04] Password Reset Flow
       ✅ Navigate to /forgot-password
       ✅ Enter email address
       ✅ Submit request
       ✅ Reset email sent (verified)
       ✅ Click reset link
       ✅ Enter new password
       ✅ Password updated successfully
       ✅ Login with new password works
       Duration: 4.7s

  Result: 4/4 flows passed ✅
  Total Duration: 11.4s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Critical User Journey: E-Commerce
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [05] Product Browse & Search
       ✅ Navigate to /products
       ✅ Product grid loaded (24 items)
       ✅ Filter by category: Electronics
       ✅ Results filtered (12 items)
       ✅ Search: "laptop"
       ✅ Search results displayed (5 items)
       ✅ Sort by price (low to high)
       ✅ Products sorted correctly
       Duration: 5.6s

  [06] Add to Cart
       ✅ Click product: "MacBook Pro 16"
       ✅ Product details page loaded
       ✅ Select quantity: 2
       ✅ Click "Add to Cart"
       ✅ Cart badge updated (2 items)
       ✅ Toast notification: "Added to cart"
       ✅ Verify cart API request: POST /api/cart
       Duration: 3.8s

  [07] Shopping Cart Management
       ✅ Navigate to /cart
       ✅ Cart items displayed correctly
       ✅ Update quantity: 2 → 1
       ✅ Cart total recalculated
       ✅ Remove item from cart
       ✅ "Cart is empty" not shown (1 item remains)
       ✅ Apply coupon code: SAVE10
       ✅ Discount applied: -$149.90
       Duration: 4.2s

  [08] Checkout & Payment
       ✅ Click "Proceed to Checkout"
       ✅ Shipping address form loaded
       ✅ Fill shipping details
       ✅ Select shipping method: Express
       ✅ Shipping cost calculated: $19.99
       ✅ Payment form loaded (Stripe iframe)
       ✅ Enter payment details
          Card: 4242 4242 4242 4242
          Expiry: 12/25
          CVC: 123
       ✅ Submit payment
       ✅ Payment processing... ⏳
       ✅ Payment successful! 💳
       ✅ Order confirmation page loaded
       ✅ Order number: #ORD-2025-1234
       ✅ Confirmation email sent
       ✅ Inventory decremented (verified)
       Duration: 12.7s

  ❌ [09] Order History
       ✅ Navigate to /account/orders
       ✅ Orders list loaded
       ❌ Order #ORD-2025-1234 not found
          Error: 404 Not Found
          Expected: Order visible immediately
          Actual: Database replication lag (eventual consistency)
          Retry 1/2... ⏳
       ✅ Retry successful (order appeared after 2s)
       Duration: 5.9s (with retry)

  Result: 5/5 flows passed ✅
  Total Duration: 32.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Critical User Journey: User Profile
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [10] View Profile
       ✅ Navigate to /account/profile
       ✅ Profile data loaded
       ✅ Avatar image displayed
       ✅ User details correct
       Duration: 2.3s

  [11] Edit Profile
       ✅ Click "Edit Profile"
       ✅ Form populated with existing data
       ✅ Update name: "Test User Updated"
       ✅ Update bio
       ✅ Save changes
       ✅ Success message displayed
       ✅ Profile updated (verified API)
       ✅ Page reloaded with new data
       Duration: 4.1s

  [12] Upload Avatar
       ✅ Click avatar upload
       ✅ Select file: avatar.jpg (2.3 MB)
       ✅ Upload started
       ✅ Progress bar: 0% → 100%
       ✅ Upload complete
       ✅ Image preview updated
       ✅ Avatar URL updated in database
       Duration: 6.8s

  Result: 3/3 flows passed ✅
  Total Duration: 13.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Performance Metrics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Page Load Times:
    /login: 0.89s (First Contentful Paint: 0.45s)
    /products: 1.23s (First Contentful Paint: 0.67s)
    /cart: 0.78s (First Contentful Paint: 0.41s)
    /checkout: 1.45s (First Contentful Paint: 0.89s)

  Interaction Times:
    Add to cart: 156ms
    Update quantity: 89ms
    Apply coupon: 234ms
    Submit payment: 2,345ms

  Network Requests:
    Total: 234 requests
    Avg response time: 187ms
    Slowest: GET /api/products (567ms)
    Failed: 0 (0%)

  Bundle Sizes:
    main.js: 342 KB (gzip: 98 KB)
    vendor.js: 1.2 MB (gzip: 387 KB)
    CSS: 145 KB (gzip: 32 KB)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Visual Validation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Screenshot Comparisons:
    ✅ Login page: No visual regressions
    ✅ Product listing: No visual regressions
    ✅ Cart page: No visual regressions
    ⚠️  Checkout page: Minor difference (2.1%)
       → Button color slightly different (#007bff vs #0056b3)
       → Acceptable within threshold

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
E2E Test Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ✅ SUCCESS
Total Flows: 12
Passed: 12 (100%)
Failed: 0
Retried: 1 (successful)

Test Breakdown:
  ✅ Authentication: 4/4 flows
  ✅ E-Commerce: 5/5 flows
  ✅ User Profile: 3/3 flows

Performance Summary:
  Average page load: 1.09s
  Average interaction: 706ms
  Total requests: 234
  Network errors: 0

Artifacts:
  📊 HTML Report: reports/e2e-test-report.html
  🎥 Videos: 0 (none recorded, all tests passed)
  📸 Screenshots: 0 (none taken, all tests passed)
  🔍 Traces: 1 (checkout flow for analysis)

Test Duration: 56.8s

Recommendations:
  ✅ All critical paths working
  ✅ Performance within acceptable range
  ⚠️  Consider optimizing /api/products response time
  ✅ Ready for deployment

✅ E2E Tests Complete!
```

## Chains With

```bash
# E2E → regression → deploy
/e2e-test && /regression-test && /k8s-deploy

# Full test pipeline
/integration-test && /e2e-test && /load-test

# Pre-production validation
/e2e-test critical && /smoke-test && /production-readiness

# Debug failed test
/e2e-test --headless false --trace always --record-video always
```

## See Also
- `/integration-test` - Integration testing
- `/regression-test` - Regression testing
- `/smoke-test` - Quick smoke tests
- `/load-test` - Load and stress testing
- `/production-readiness` - Pre-deployment validation
