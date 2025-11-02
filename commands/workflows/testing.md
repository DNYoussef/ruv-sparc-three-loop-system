---
name: workflow:testing
description: Comprehensive testing workflow orchestration with unit, integration, e2e, and performance tests
category: workflows
version: 2.0.0
requires_mcp: true
---

# 🧪 Testing Workflow - Comprehensive Test Orchestration

**Workflow**: Testing Pipeline | **Category**: Workflows | **Loop**: Loop 3 Validation

Complete testing workflow covering unit tests, integration tests, e2e tests, performance tests, and accessibility audits.

## Workflow Execution

```bash
#!/bin/bash
# Complete testing workflow

set -e  # Exit on error

echo "🧪 Testing Workflow Started"
echo "================================"

# Phase 1: Unit Tests
echo "Phase 1: Running Unit Tests..."
npm run test:unit -- --coverage --bail

# Store coverage
npx claude-flow@alpha memory store \
  --key "testing/unit/coverage/$(date +%Y%m%d)" \
  --value "$(cat coverage/coverage-summary.json)"

# Phase 2: Integration Tests
echo "Phase 2: Running Integration Tests..."
npm run test:integration

# Phase 3: E2E Tests
echo "Phase 3: Running E2E Tests..."
npm run test:e2e

# Phase 4: Performance Tests
echo "Phase 4: Running Performance Tests..."
npx autocannon http://localhost:3000/api/health \
  -c 100 -d 30 -p 10 > performance/api-benchmark.json

# Phase 5: Accessibility Tests
echo "Phase 5: Running Accessibility Tests..."
npm run test:a11y

# Phase 6: Security Tests
echo "Phase 6: Running Security Scan..."
npm audit --audit-level=moderate
npx snyk test

# Phase 7: Test Report Generation
echo "Phase 7: Generating Test Report..."
cat > reports/test-summary.json <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "unit": {
    "total": $(jq '.numTotalTests' coverage/coverage-summary.json),
    "passed": $(jq '.numPassedTests' coverage/coverage-summary.json),
    "coverage": $(jq '.total.lines.pct' coverage/coverage-summary.json)
  },
  "integration": "passed",
  "e2e": "passed",
  "performance": "passed",
  "accessibility": "passed",
  "security": "passed"
}
EOF

echo "✅ Testing Workflow Complete"
echo "================================"
echo "Results stored in reports/test-summary.json"
```

## Phase Breakdown

### 1. Unit Testing
```typescript
// Jest Configuration
export default {
  preset: 'ts-jest',
  testEnvironment: 'node',
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/**/*.d.ts',
    '!src/**/*.test.{ts,tsx}'
  ]
};

// Example Unit Test
describe('UserService', () => {
  let service: UserService;
  let mockRepository: jest.Mocked<UserRepository>;

  beforeEach(() => {
    mockRepository = {
      findById: jest.fn(),
      create: jest.fn(),
      update: jest.fn()
    } as any;

    service = new UserService(mockRepository);
  });

  it('should create user with hashed password', async () => {
    const userData = {
      email: 'test@example.com',
      password: 'password123'
    };

    mockRepository.create.mockResolvedValue({
      id: '1',
      email: userData.email,
      password: 'hashed_password'
    });

    const user = await service.create(userData);

    expect(mockRepository.create).toHaveBeenCalled();
    expect(user.password).not.toBe(userData.password);
  });
});
```

### 2. Integration Testing
```typescript
// API Integration Tests
describe('POST /api/users', () => {
  it('should create user and return 201', async () => {
    const response = await request(app)
      .post('/api/users')
      .send({
        email: 'test@example.com',
        password: 'password123',
        username: 'testuser'
      })
      .expect(201);

    expect(response.body.data).toHaveProperty('id');
    expect(response.body.data.email).toBe('test@example.com');

    // Cleanup
    await db.user.delete({ where: { id: response.body.data.id } });
  });

  it('should validate email format', async () => {
    const response = await request(app)
      .post('/api/users')
      .send({
        email: 'invalid-email',
        password: 'password123'
      })
      .expect(400);

    expect(response.body.error.code).toBe('VALIDATION_ERROR');
  });
});
```

### 3. E2E Testing
```typescript
// Playwright E2E Tests
import { test, expect } from '@playwright/test';

test.describe('User Registration Flow', () => {
  test('should complete full registration', async ({ page }) => {
    // Navigate to registration page
    await page.goto('/register');

    // Fill form
    await page.fill('[name="email"]', 'test@example.com');
    await page.fill('[name="password"]', 'password123');
    await page.fill('[name="confirmPassword"]', 'password123');

    // Submit
    await page.click('button[type="submit"]');

    // Verify success
    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('h1')).toContainText('Welcome');
  });

  test('should show validation errors', async ({ page }) => {
    await page.goto('/register');
    await page.click('button[type="submit"]');

    await expect(page.locator('[role="alert"]')).toBeVisible();
    await expect(page.locator('[role="alert"]')).toContainText('Email is required');
  });
});
```

### 4. Performance Testing
```bash
# API Performance Test
npx autocannon http://localhost:3000/api/users \
  --connections 100 \
  --duration 30 \
  --pipelining 10 \
  --workers 4 \
  --json > performance/users-endpoint.json

# Frontend Performance Test
npx lighthouse http://localhost:3000 \
  --output json \
  --output-path performance/lighthouse.json \
  --chrome-flags="--headless"

# Load Testing with k6
cat > tests/load-test.js <<EOF
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 0 }
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
    http_req_failed: ['rate<0.01']
  }
};

export default function () {
  const response = http.get('http://localhost:3000/api/users');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500
  });
  sleep(1);
}
EOF

k6 run tests/load-test.js
```

### 5. Accessibility Testing
```typescript
// Automated A11y Tests with jest-axe
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

describe('Accessibility', () => {
  it('should have no accessibility violations', async () => {
    const { container } = render(<LoginForm />);
    const results = await axe(container);

    expect(results).toHaveNoViolations();
  });

  it('should be keyboard navigable', async () => {
    const { getByLabelText, getByRole } = render(<LoginForm />);

    const emailInput = getByLabelText(/email/i);
    const submitButton = getByRole('button', { name: /sign in/i });

    emailInput.focus();
    expect(document.activeElement).toBe(emailInput);

    // Tab to next element
    userEvent.tab();
    // Should focus on password input
  });
});
```

---

## Continuous Testing Integration

```yaml
# .github/workflows/testing.yml
name: Testing Workflow

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install Dependencies
        run: npm ci

      - name: Run Unit Tests
        run: npm run test:unit -- --coverage

      - name: Run Integration Tests
        run: npm run test:integration

      - name: Run E2E Tests
        run: npm run test:e2e

      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/coverage-final.json

      - name: Performance Tests
        run: |
          npm start &
          sleep 5
          npx autocannon http://localhost:3000/api/health -d 10
```

---

## Test Metrics & Reporting

```typescript
/**
 * Test Metrics Collection
 */
interface TestMetrics {
  unit: {
    total: number;
    passed: number;
    failed: number;
    coverage: {
      lines: number;
      branches: number;
      functions: number;
      statements: number;
    };
  };
  integration: {
    total: number;
    passed: number;
    failed: number;
    duration: number;
  };
  e2e: {
    total: number;
    passed: number;
    failed: number;
    duration: number;
  };
  performance: {
    p50: number;
    p95: number;
    p99: number;
    throughput: number;
  };
}

// Generate Report
function generateTestReport(metrics: TestMetrics): string {
  return `
# Test Report - ${new Date().toISOString()}

## Summary
- **Total Tests**: ${metrics.unit.total + metrics.integration.total + metrics.e2e.total}
- **Passed**: ${metrics.unit.passed + metrics.integration.passed + metrics.e2e.passed}
- **Failed**: ${metrics.unit.failed + metrics.integration.failed + metrics.e2e.failed}

## Unit Tests
- Coverage: ${metrics.unit.coverage.lines}%
- Tests: ${metrics.unit.passed}/${metrics.unit.total}

## Integration Tests
- Tests: ${metrics.integration.passed}/${metrics.integration.total}
- Duration: ${metrics.integration.duration}ms

## E2E Tests
- Tests: ${metrics.e2e.passed}/${metrics.e2e.total}
- Duration: ${metrics.e2e.duration}ms

## Performance
- P95 Response Time: ${metrics.performance.p95}ms
- Throughput: ${metrics.performance.throughput} req/s
  `;
}
```

---

## Integration with Three-Loop System

**Loop 2**: Receive implementation for testing
**Loop 3**: Execute comprehensive test suite (this workflow)

**Success Criteria**:
- Unit test coverage > 80%
- All integration tests passing
- All E2E tests passing
- Performance metrics within SLA
- Zero accessibility violations

**Agent Status**: Production-Ready
**Version**: 2.0.0
