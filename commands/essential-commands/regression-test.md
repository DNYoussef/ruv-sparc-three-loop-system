---
name: regression-test
category: testing
version: 1.0.0
---

# /regression-test

Comprehensive regression testing to detect unintended changes and breaking regressions.

## Usage
```bash
/regression-test [test_suite] [options]
```

## Parameters
- `test_suite` - Test suite to run: all|critical|ui|api|db (default: all)
- `--baseline` - Baseline version for comparison (default: previous release)
- `--parallel` - Run tests in parallel (default: true)
- `--workers` - Number of parallel workers (default: 4)
- `--fail-fast` - Stop on first failure (default: false)
- `--screenshot-diff` - Enable visual regression testing (default: true)
- `--report-format` - Output format: html|json|junit (default: html)
- `--coverage` - Include code coverage (default: false)

## What It Does

**Multi-Layer Regression Testing**:
1. 🧪 **Unit Tests**: Component-level regression
2. 🔗 **Integration Tests**: API contract validation
3. 🎨 **Visual Regression**: Screenshot comparison
4. 📊 **Performance Tests**: Response time baselines
5. 🗄️ **Database Tests**: Schema and query validation
6. 🌐 **E2E Tests**: Critical user flows
7. 📈 **Metrics Comparison**: Against baseline
8. 📝 **Report Generation**: Detailed regression report

**Detection Categories**:
- **Breaking Changes**: API contracts, function signatures
- **UI Regressions**: Layout shifts, visual bugs
- **Performance Degradation**: Slower response times
- **Data Integrity**: Database schema changes
- **Security Regressions**: New vulnerabilities
- **Behavior Changes**: Altered business logic

## Examples

```bash
# Full regression test suite
/regression-test

# Critical path tests only
/regression-test critical --fail-fast true

# API regression tests
/regression-test api --baseline v1.2.0

# UI regression with screenshots
/regression-test ui --screenshot-diff true

# Parallel execution with 8 workers
/regression-test --parallel true --workers 8

# With coverage report
/regression-test --coverage true --report-format html

# Database regression tests
/regression-test db --baseline v1.2.9
```

## Output

```
🧪 Regression Test Suite Started

Configuration:
  Suite: all
  Baseline: v1.2.9
  Current: v1.3.0
  Parallel: 4 workers
  Visual Regression: Enabled

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit Tests (342 tests)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ AuthService: 45 passed
  ✅ UserController: 38 passed
  ✅ DataValidator: 52 passed
  ✅ CacheManager: 27 passed
  ❌ PaymentService: 12 passed, 3 failed
     → processRefund() breaking change detected
     → Expected: Promise<Refund>
     → Actual: Promise<RefundResult>
  ✅ NotificationService: 41 passed
  ✅ Utilities: 124 passed

  Result: 339/342 passed (99.1%)
  Duration: 8.2s (parallel)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Integration Tests (87 tests)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  API Contract Validation:
    ✅ GET /api/users: Schema valid
    ✅ POST /api/auth/login: Response unchanged
    ❌ GET /api/orders/{id}: Breaking change
       → Response field 'customer' renamed to 'buyer'
       → Affected clients: Web App, Mobile App
    ✅ PUT /api/profile: Backward compatible
    ✅ DELETE /api/sessions: Unchanged

  Database Queries:
    ✅ User queries: Performance stable
    ⚠️  Order queries: 15% slower (87ms → 100ms)
       → Investigate missing index
    ✅ Product queries: 5% faster (optimized)

  External Services:
    ✅ Stripe API: Integration stable
    ✅ SendGrid: Email sending works
    ✅ S3: File upload unchanged

  Result: 84/87 passed (96.6%)
  Duration: 12.4s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Visual Regression (23 screens)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Comparing screenshots with baseline v1.2.9...

  ✅ Login page: No differences (0%)
  ✅ Dashboard: No differences (0%)
  ❌ User profile: Visual regression detected (12.3%)
     → Layout shift: Avatar moved 15px down
     → Color change: Button #007bff → #0056b3
     → Screenshot diff: reports/visual-diff/profile.png
  ✅ Settings: No differences (0%)
  ✅ Products list: No differences (0%)
  ⚠️  Checkout flow: Minor difference (2.1%)
     → Font weight change: 400 → 500 (acceptable)

  Result: 21/23 passed (91.3%)
  Duration: 15.7s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Performance Regression (45 endpoints)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Comparing response times with baseline v1.2.9...

  ✅ GET /api/users: 45ms (baseline: 43ms, +4.7%)
  ✅ POST /api/orders: 89ms (baseline: 92ms, -3.3%)
  ❌ GET /api/analytics: 2,456ms (baseline: 187ms, +1,213%)
     → CRITICAL: 13x slower, investigate immediately
     → Likely cause: Missing database index
  ✅ GET /api/products: 67ms (baseline: 71ms, -5.6%)
  ⚠️  POST /api/checkout: 234ms (baseline: 198ms, +18.2%)
     → Slower but within threshold (<20%)

  Metrics Summary:
    - Improved: 28 endpoints (62%)
    - Stable: 12 endpoints (27%)
    - Degraded: 4 endpoints (9%)
    - Critical: 1 endpoint (2%)

  Result: 44/45 within SLA (97.8%)
  Duration: 34.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Database Regression (12 schemas)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Schema Validation:
    ✅ users: No breaking changes
    ✅ orders: Backward compatible (added optional fields)
    ❌ payments: Breaking change detected
       → Column 'status' type changed: varchar(50) → enum
       → Migration required for existing data
    ✅ products: No changes
    ✅ sessions: No changes

  Index Coverage:
    ✅ users.email: Index present
    ❌ orders.created_at: Missing index (used in queries)
       → Add index to improve performance
    ✅ products.category_id: Index present

  Data Integrity:
    ✅ Foreign key constraints: Valid
    ✅ Unique constraints: No violations
    ✅ Check constraints: All passing

  Result: 10/12 schemas valid (83.3%)
  Duration: 5.8s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
E2E Critical Paths (18 flows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ User Registration: Success (3.2s)
  ✅ Login Flow: Success (2.1s)
  ✅ Browse Products: Success (4.5s)
  ❌ Checkout Process: Failed
     → Error: "buyer is not defined"
     → Related to API breaking change (customer → buyer)
  ✅ Payment: Success (6.7s)
  ✅ Order History: Success (2.8s)
  ✅ Profile Update: Success (3.1s)

  Result: 17/18 passed (94.4%)
  Duration: 45.6s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Regression Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: ⚠️  REGRESSIONS DETECTED
Total Tests: 527
Passed: 495 (93.9%)
Failed: 32 (6.1%)

Critical Issues (3):
  1. ❌ API Breaking Change: orders endpoint
     Impact: Web app, mobile app
     Fix: Restore 'customer' field or provide migration

  2. ❌ Performance Regression: /api/analytics (13x slower)
     Impact: Dashboard load time
     Fix: Add database index on analytics.timestamp

  3. ❌ E2E Failure: Checkout flow broken
     Impact: Revenue loss (checkout broken)
     Fix: Update frontend to use 'buyer' field

High Priority (2):
  4. ⚠️  Visual Regression: Profile page layout
  5. ⚠️  Database: Missing index on orders.created_at

Medium Priority (27):
  - Various minor UI differences
  - Small performance variations

Comparison with Baseline (v1.2.9):
  Breaking Changes: 3
  Performance Regressions: 5
  Visual Changes: 2
  Schema Changes: 2

Recommended Actions:
  1. Fix critical API breaking change
  2. Add missing database index
  3. Update checkout flow frontend code
  4. Review visual regression for profile page
  5. Re-run regression tests after fixes

Test Duration: 2m 1s
Report: reports/regression-v1.3.0-vs-v1.2.9.html

⚠️  RECOMMENDATION: DO NOT DEPLOY to production
Fix critical regressions first.
```

## Chains With

```bash
# Regression test → fix → re-test
/regression-test && /fix-bug && /regression-test

# Full quality pipeline
/regression-test && /integration-test && /e2e-test

# Pre-deployment validation
/regression-test --fail-fast true && /production-readiness

# Test → analyze → report
/regression-test && /bottleneck-detect && /performance-report
```

## See Also
- `/integration-test` - Integration testing
- `/e2e-test` - End-to-end testing
- `/load-test` - Load testing
- `/smoke-test` - Quick smoke tests
- `/production-readiness` - Pre-deployment validation
- `/performance-report` - Performance analysis
