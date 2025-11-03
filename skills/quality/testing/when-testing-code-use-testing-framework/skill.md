# Testing Framework Skill - Complete SOP

## Metadata
- **Skill ID**: when-testing-code-use-testing-framework
- **Version**: 1.0.0
- **Category**: Testing
- **Complexity**: HIGH
- **Agents Required**: tester, coder, code-analyzer
- **Estimated Time**: 2-4 hours

## Purpose

Implement comprehensive testing infrastructure following industry best practices with proper test strategy, unit tests, integration tests, E2E tests, coverage analysis, and CI/CD integration.

## Trigger Conditions

Activate this skill when:
- User explicitly requests "test this code" or "add tests"
- New feature implementation requires test coverage
- Code changes need test validation
- Setting up testing infrastructure for a project
- Improving test coverage metrics
- Implementing TDD workflow
- Preparing code for production deployment

## 5-Phase Testing Framework SOP

### Phase 1: Test Strategy Design (20 minutes)

**Objective**: Analyze codebase and design comprehensive test strategy

**Agent**: `code-analyzer`

**Actions**:
1. Analyze codebase structure and identify testable units
2. Determine appropriate testing frameworks (Jest, Vitest, Playwright)
3. Define coverage goals (minimum 80% for critical paths)
4. Identify test categories needed:
   - Unit tests (isolated functions/methods)
   - Integration tests (component interactions)
   - E2E tests (user workflows)
   - Performance tests (if applicable)
5. Create test plan document with priorities

**Hooks**:
```bash
npx claude-flow@alpha hooks pre-task --description "Test strategy design"
npx claude-flow@alpha hooks session-restore --session-id "testing-framework"
```

**Outputs**:
- Test strategy document (`docs/TEST_STRATEGY.md`)
- Coverage goals and metrics
- Testing framework selections
- Test file structure plan

**Memory Storage**:
```bash
npx claude-flow@alpha memory store \
  --key "testing/strategy" \
  --value "$(cat docs/TEST_STRATEGY.md)" \
  --tags "testing,strategy,phase-1"
```

---

### Phase 2: Unit Test Implementation (60 minutes)

**Objective**: Create isolated unit tests for all functions and methods

**Agent**: `tester`

**Actions**:
1. Set up testing framework configuration
2. Create test utilities and helpers
3. Write unit tests for each module:
   - Test happy paths
   - Test edge cases
   - Test error handling
   - Mock external dependencies
4. Organize tests to mirror source structure
5. Ensure tests are atomic and independent

**Test Patterns**:

**Jest/Vitest Unit Test Example**:
```javascript
// tests/unit/utils/calculator.test.js
import { describe, it, expect, beforeEach } from 'vitest';
import { Calculator } from '@/utils/calculator';

describe('Calculator', () => {
  let calculator;

  beforeEach(() => {
    calculator = new Calculator();
  });

  describe('add', () => {
    it('should add two positive numbers correctly', () => {
      expect(calculator.add(2, 3)).toBe(5);
    });

    it('should handle negative numbers', () => {
      expect(calculator.add(-2, 3)).toBe(1);
      expect(calculator.add(-2, -3)).toBe(-5);
    });

    it('should handle zero values', () => {
      expect(calculator.add(0, 5)).toBe(5);
      expect(calculator.add(5, 0)).toBe(5);
    });

    it('should throw error for non-numeric inputs', () => {
      expect(() => calculator.add('a', 2)).toThrow('Invalid input');
    });
  });

  describe('divide', () => {
    it('should divide numbers correctly', () => {
      expect(calculator.divide(10, 2)).toBe(5);
    });

    it('should handle decimal results', () => {
      expect(calculator.divide(5, 2)).toBe(2.5);
    });

    it('should throw error when dividing by zero', () => {
      expect(() => calculator.divide(10, 0)).toThrow('Division by zero');
    });
  });
});
```

**Hooks**:
```bash
npx claude-flow@alpha hooks post-edit \
  --file "tests/unit/calculator.test.js" \
  --memory-key "testing/unit-tests"
```

**Outputs**:
- Unit test files in `tests/unit/`
- Test configuration (`vitest.config.js` or `jest.config.js`)
- Test utilities (`tests/helpers/`)
- Mock files (`tests/mocks/`)

**File Structure**:
```
tests/
├── unit/
│   ├── utils/
│   │   ├── calculator.test.js
│   │   └── validator.test.js
│   ├── services/
│   │   └── userService.test.js
│   └── models/
│       └── User.test.js
├── helpers/
│   ├── testUtils.js
│   └── mockData.js
└── mocks/
    ├── api.js
    └── database.js
```

---

### Phase 3: Integration Test Creation (60 minutes)

**Objective**: Test component interactions and data flow

**Agent**: `tester`

**Actions**:
1. Identify integration points (API calls, database operations, services)
2. Set up test databases or mock services
3. Create integration tests for:
   - API endpoints
   - Service layer interactions
   - Database operations
   - External service integrations
4. Use real dependencies where feasible
5. Implement proper setup/teardown

**Integration Test Example**:
```javascript
// tests/integration/api/users.test.js
import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import request from 'supertest';
import { app } from '@/app';
import { database } from '@/config/database';
import { createTestUser, clearDatabase } from '@/tests/helpers/testUtils';

describe('User API Integration Tests', () => {
  beforeAll(async () => {
    await database.connect();
  });

  afterAll(async () => {
    await database.disconnect();
  });

  beforeEach(async () => {
    await clearDatabase();
  });

  describe('POST /api/users', () => {
    it('should create a new user successfully', async () => {
      const userData = {
        email: 'test@example.com',
        name: 'Test User',
        password: 'SecurePass123!'
      };

      const response = await request(app)
        .post('/api/users')
        .send(userData)
        .expect(201);

      expect(response.body).toMatchObject({
        id: expect.any(String),
        email: userData.email,
        name: userData.name
      });
      expect(response.body).not.toHaveProperty('password');
    });

    it('should return 400 for invalid email format', async () => {
      const userData = {
        email: 'invalid-email',
        name: 'Test User',
        password: 'SecurePass123!'
      };

      const response = await request(app)
        .post('/api/users')
        .send(userData)
        .expect(400);

      expect(response.body.error).toMatch(/invalid email/i);
    });

    it('should return 409 for duplicate email', async () => {
      const userData = {
        email: 'test@example.com',
        name: 'Test User',
        password: 'SecurePass123!'
      };

      await createTestUser(userData);

      const response = await request(app)
        .post('/api/users')
        .send(userData)
        .expect(409);

      expect(response.body.error).toMatch(/email already exists/i);
    });
  });

  describe('GET /api/users/:id', () => {
    it('should retrieve user by id', async () => {
      const user = await createTestUser({
        email: 'test@example.com',
        name: 'Test User'
      });

      const response = await request(app)
        .get(`/api/users/${user.id}`)
        .expect(200);

      expect(response.body).toMatchObject({
        id: user.id,
        email: user.email,
        name: user.name
      });
    });

    it('should return 404 for non-existent user', async () => {
      const response = await request(app)
        .get('/api/users/non-existent-id')
        .expect(404);

      expect(response.body.error).toMatch(/user not found/i);
    });
  });
});
```

**Service Integration Test Example**:
```javascript
// tests/integration/services/orderService.test.js
import { describe, it, expect, beforeEach } from 'vitest';
import { OrderService } from '@/services/OrderService';
import { UserService } from '@/services/UserService';
import { ProductService } from '@/services/ProductService';
import { database } from '@/config/database';
import { clearDatabase, createTestUser, createTestProduct } from '@/tests/helpers/testUtils';

describe('OrderService Integration', () => {
  let orderService;
  let userService;
  let productService;

  beforeEach(async () => {
    await clearDatabase();
    orderService = new OrderService();
    userService = new UserService();
    productService = new ProductService();
  });

  it('should create order with valid user and products', async () => {
    const user = await createTestUser();
    const product1 = await createTestProduct({ price: 10.00 });
    const product2 = await createTestProduct({ price: 20.00 });

    const order = await orderService.createOrder({
      userId: user.id,
      items: [
        { productId: product1.id, quantity: 2 },
        { productId: product2.id, quantity: 1 }
      ]
    });

    expect(order).toMatchObject({
      id: expect.any(String),
      userId: user.id,
      total: 40.00,
      status: 'pending'
    });
    expect(order.items).toHaveLength(2);
  });

  it('should update inventory when order is created', async () => {
    const user = await createTestUser();
    const product = await createTestProduct({
      price: 10.00,
      inventory: 100
    });

    await orderService.createOrder({
      userId: user.id,
      items: [{ productId: product.id, quantity: 5 }]
    });

    const updatedProduct = await productService.getById(product.id);
    expect(updatedProduct.inventory).toBe(95);
  });

  it('should fail when product inventory is insufficient', async () => {
    const user = await createTestUser();
    const product = await createTestProduct({
      price: 10.00,
      inventory: 2
    });

    await expect(
      orderService.createOrder({
        userId: user.id,
        items: [{ productId: product.id, quantity: 5 }]
      })
    ).rejects.toThrow(/insufficient inventory/i);
  });
});
```

**Outputs**:
- Integration test files in `tests/integration/`
- Test database configurations
- API test utilities
- Service mocks and stubs

---

### Phase 4: E2E Test Setup (60 minutes)

**Objective**: Create end-to-end tests simulating real user workflows

**Agent**: `tester`

**Actions**:
1. Set up E2E testing framework (Playwright, Cypress)
2. Configure test environments
3. Create page object models
4. Write E2E tests for critical user journeys:
   - Authentication flows
   - Core feature workflows
   - Error scenarios
5. Implement visual regression tests (if applicable)

**Playwright E2E Test Example**:
```javascript
// tests/e2e/user-registration.spec.js
import { test, expect } from '@playwright/test';

test.describe('User Registration Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/register');
  });

  test('should register new user successfully', async ({ page }) => {
    // Fill registration form
    await page.fill('input[name="email"]', 'newuser@example.com');
    await page.fill('input[name="name"]', 'New User');
    await page.fill('input[name="password"]', 'SecurePass123!');
    await page.fill('input[name="confirmPassword"]', 'SecurePass123!');

    // Submit form
    await page.click('button[type="submit"]');

    // Verify success
    await expect(page.locator('.success-message')).toContainText(
      'Registration successful'
    );
    await expect(page).toHaveURL('/dashboard');

    // Verify user is logged in
    await expect(page.locator('.user-menu')).toContainText('New User');
  });

  test('should show validation errors for invalid input', async ({ page }) => {
    await page.fill('input[name="email"]', 'invalid-email');
    await page.fill('input[name="password"]', '123');
    await page.click('button[type="submit"]');

    await expect(page.locator('.error-message')).toContainText(
      'Invalid email format'
    );
    await expect(page.locator('.error-message')).toContainText(
      'Password must be at least 8 characters'
    );
  });

  test('should show error for duplicate email', async ({ page }) => {
    // First registration
    await page.fill('input[name="email"]', 'existing@example.com');
    await page.fill('input[name="name"]', 'Existing User');
    await page.fill('input[name="password"]', 'SecurePass123!');
    await page.fill('input[name="confirmPassword"]', 'SecurePass123!');
    await page.click('button[type="submit"]');

    // Logout and try again with same email
    await page.click('.logout-button');
    await page.goto('/register');

    await page.fill('input[name="email"]', 'existing@example.com');
    await page.fill('input[name="name"]', 'Another User');
    await page.fill('input[name="password"]', 'SecurePass123!');
    await page.fill('input[name="confirmPassword"]', 'SecurePass123!');
    await page.click('button[type="submit"]');

    await expect(page.locator('.error-message')).toContainText(
      'Email already registered'
    );
  });
});

test.describe('User Login Flow', () => {
  test('should login existing user successfully', async ({ page }) => {
    await page.goto('/login');

    await page.fill('input[name="email"]', 'testuser@example.com');
    await page.fill('input[name="password"]', 'TestPass123!');
    await page.click('button[type="submit"]');

    await expect(page).toHaveURL('/dashboard');
    await expect(page.locator('.welcome-message')).toContainText('Welcome back');
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');

    await page.fill('input[name="email"]', 'wrong@example.com');
    await page.fill('input[name="password"]', 'WrongPassword');
    await page.click('button[type="submit"]');

    await expect(page.locator('.error-message')).toContainText(
      'Invalid email or password'
    );
    await expect(page).toHaveURL('/login');
  });
});
```

**Page Object Model Example**:
```javascript
// tests/e2e/pages/RegistrationPage.js
export class RegistrationPage {
  constructor(page) {
    this.page = page;
    this.emailInput = page.locator('input[name="email"]');
    this.nameInput = page.locator('input[name="name"]');
    this.passwordInput = page.locator('input[name="password"]');
    this.confirmPasswordInput = page.locator('input[name="confirmPassword"]');
    this.submitButton = page.locator('button[type="submit"]');
    this.successMessage = page.locator('.success-message');
    this.errorMessage = page.locator('.error-message');
  }

  async goto() {
    await this.page.goto('/register');
  }

  async register(email, name, password, confirmPassword = password) {
    await this.emailInput.fill(email);
    await this.nameInput.fill(name);
    await this.passwordInput.fill(password);
    await this.confirmPasswordInput.fill(confirmPassword);
    await this.submitButton.click();
  }

  async expectSuccess() {
    await expect(this.successMessage).toBeVisible();
    await expect(this.page).toHaveURL('/dashboard');
  }

  async expectError(message) {
    await expect(this.errorMessage).toContainText(message);
  }
}
```

**Outputs**:
- E2E test files in `tests/e2e/`
- Page object models in `tests/e2e/pages/`
- Playwright/Cypress configuration
- Test fixtures and utilities

---

### Phase 5: Coverage Analysis & CI/CD Integration (40 minutes)

**Objective**: Analyze test coverage, generate reports, and integrate with CI/CD

**Agent**: `coder`

**Actions**:
1. Configure coverage collection
2. Generate coverage reports (HTML, JSON, LCOV)
3. Identify untested code paths
4. Set up CI/CD pipeline for automated testing
5. Configure coverage thresholds
6. Create pre-commit hooks for testing

**Coverage Configuration**:
```javascript
// vitest.config.js
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'json', 'lcov'],
      all: true,
      include: ['src/**/*.{js,ts,jsx,tsx}'],
      exclude: [
        'src/**/*.test.{js,ts,jsx,tsx}',
        'src/**/*.spec.{js,ts,jsx,tsx}',
        'src/**/index.{js,ts}',
        'src/**/*.d.ts',
        'src/tests/**'
      ],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 75,
        statements: 80
      }
    },
    setupFiles: ['./tests/setup.js'],
    testTimeout: 10000
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  }
});
```

**GitHub Actions CI Configuration**:
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [18.x, 20.x]

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run unit tests
        run: npm run test:unit

      - name: Run integration tests
        run: npm run test:integration
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/testdb

      - name: Run E2E tests
        run: npm run test:e2e

      - name: Generate coverage report
        run: npm run test:coverage

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
          flags: unittests
          name: codecov-umbrella

      - name: Check coverage thresholds
        run: npm run test:coverage:check

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20.x'
      - run: npm ci
      - run: npm run lint
      - run: npm run typecheck
```

**Package.json Scripts**:
```json
{
  "scripts": {
    "test": "vitest",
    "test:unit": "vitest run --config vitest.unit.config.js",
    "test:integration": "vitest run --config vitest.integration.config.js",
    "test:e2e": "playwright test",
    "test:coverage": "vitest run --coverage",
    "test:coverage:check": "vitest run --coverage --run",
    "test:watch": "vitest watch",
    "test:ui": "vitest --ui",
    "test:e2e:ui": "playwright test --ui",
    "test:e2e:debug": "playwright test --debug"
  }
}
```

**Pre-commit Hook**:
```bash
#!/bin/bash
# .husky/pre-commit

echo "Running tests before commit..."

# Run unit tests
npm run test:unit
if [ $? -ne 0 ]; then
  echo "Unit tests failed. Commit aborted."
  exit 1
fi

# Run linting
npm run lint
if [ $? -ne 0 ]; then
  echo "Linting failed. Commit aborted."
  exit 1
fi

echo "All checks passed. Proceeding with commit."
```

**Hooks**:
```bash
npx claude-flow@alpha hooks post-task --task-id "testing-framework"
npx claude-flow@alpha hooks session-end --export-metrics true
```

**Outputs**:
- Coverage reports in `coverage/`
- CI/CD configuration files
- Pre-commit hooks
- Test documentation

**Memory Storage**:
```bash
npx claude-flow@alpha memory store \
  --key "testing/coverage" \
  --value "$(cat coverage/coverage-summary.json)" \
  --tags "testing,coverage,metrics"
```

---

## Success Criteria

- [ ] All phases completed successfully
- [ ] Unit test coverage ≥ 80% for critical paths
- [ ] Integration tests cover all API endpoints
- [ ] E2E tests cover critical user journeys
- [ ] All tests pass in CI/CD pipeline
- [ ] Coverage reports generated and accessible
- [ ] Pre-commit hooks configured
- [ ] Test documentation complete

## Quality Gates

**Phase 1**: Test strategy document approved
**Phase 2**: Unit tests pass with >80% coverage
**Phase 3**: Integration tests pass with real dependencies
**Phase 4**: E2E tests pass in headless mode
**Phase 5**: CI/CD pipeline executes successfully

## Rollback Strategy

If any phase fails:
1. Document failure reason in memory
2. Rollback to previous working state
3. Notify orchestrator for replanning
4. Store lessons learned for future attempts

## Performance Metrics

- Test execution time: <5 minutes for full suite
- Coverage collection overhead: <10% additional time
- CI/CD pipeline duration: <10 minutes
- Test flakiness rate: <1%

## Documentation Requirements

- Test strategy document
- Test coverage report
- Testing best practices guide
- Troubleshooting guide
- CI/CD setup instructions
