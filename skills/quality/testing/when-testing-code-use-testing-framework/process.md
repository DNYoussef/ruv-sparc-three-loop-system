# Testing Framework Process - Detailed Workflow

## Table of Contents

1. [Test Strategy Development](#test-strategy-development)
2. [Unit Testing Workflow](#unit-testing-workflow)
3. [Integration Testing Workflow](#integration-testing-workflow)
4. [E2E Testing Workflow](#e2e-testing-workflow)
5. [Coverage Analysis](#coverage-analysis)
6. [CI/CD Integration](#cicd-integration)
7. [Test Maintenance](#test-maintenance)

---

## Test Strategy Development

### Step 1: Codebase Analysis

**Objective**: Understand the codebase structure and identify testable units

**Actions**:

1. **Inventory Source Files**
   ```bash
   # List all source files
   find src -type f \( -name "*.js" -o -name "*.ts" \) | wc -l

   # Analyze file complexity
   npx complexity-report src/
   ```

2. **Identify Module Types**
   - Utilities (pure functions)
   - Services (business logic)
   - Controllers (API handlers)
   - Models (data structures)
   - Components (UI elements)

3. **Map Dependencies**
   ```bash
   # Visualize dependency graph
   npx madge --image graph.png src/
   ```

4. **Identify Testing Priorities**
   - High priority: Business logic, authentication, payment processing
   - Medium priority: API endpoints, data validation, utilities
   - Low priority: UI components, configuration, static content

### Step 2: Framework Selection

**Decision Matrix**:

| Framework | Best For | Pros | Cons |
|-----------|----------|------|------|
| Vitest | Modern projects, Vite users | Fast, modern, great DX | Smaller ecosystem |
| Jest | Large projects, mature codebases | Mature, extensive plugins | Slower, older tech |
| Playwright | E2E testing | Cross-browser, reliable | Steeper learning curve |
| Cypress | E2E testing, visual testing | Great DX, time-travel debugging | Browser limitations |

**Selection Criteria**:
- Project type (frontend, backend, full-stack)
- Build tool (Vite, webpack, Rollup)
- Team familiarity
- Performance requirements
- CI/CD integration needs

### Step 3: Define Coverage Goals

**Coverage Targets by Module Type**:

```javascript
// vitest.config.js
export default {
  test: {
    coverage: {
      thresholds: {
        // Global thresholds
        lines: 80,
        functions: 80,
        branches: 75,
        statements: 80,

        // Per-file thresholds
        'src/services/**/*.js': {
          lines: 90,
          functions: 90,
          branches: 85,
          statements: 90
        },
        'src/utils/**/*.js': {
          lines: 95,
          functions: 95,
          branches: 90,
          statements: 95
        },
        'src/controllers/**/*.js': {
          lines: 80,
          functions: 85,
          branches: 75,
          statements: 80
        }
      }
    }
  }
};
```

### Step 4: Create Test Plan Document

**Template**:

```markdown
# Test Plan - [Project Name]

## Testing Strategy

### Scope
- In scope: [List features to test]
- Out of scope: [List what won't be tested]

### Test Levels
1. Unit Tests: [Coverage goal: 85%]
2. Integration Tests: [Coverage goal: 80%]
3. E2E Tests: [Critical user journeys]

### Testing Frameworks
- Unit/Integration: Vitest
- E2E: Playwright
- API: Supertest

### Test Environment
- Local: Development database
- CI: Test database (Docker)
- Staging: Staging environment

### Success Criteria
- All tests pass
- Coverage > 80%
- No critical bugs
- Performance < 5min

### Risk Assessment
- High Risk: Payment processing, authentication
- Medium Risk: Data validation, API endpoints
- Low Risk: UI components, static pages
```

---

## Unit Testing Workflow

### Step 1: Setup Testing Environment

**Install Dependencies**:
```bash
npm install -D vitest @vitest/ui @vitest/coverage-v8
npm install -D @faker-js/faker
npm install -D msw  # For API mocking
```

**Configure Vitest**:
```javascript
// vitest.config.js
import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    setupFiles: ['./tests/setup.js'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'json', 'lcov'],
      include: ['src/**/*.{js,ts}'],
      exclude: [
        'src/**/*.test.{js,ts}',
        'src/**/*.spec.{js,ts}',
        'src/**/index.{js,ts}'
      ]
    },
    testTimeout: 10000,
    hookTimeout: 10000
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  }
});
```

### Step 2: Create Test Utilities

**Test Setup File**:
```javascript
// tests/setup.js
import { beforeAll, afterAll, beforeEach, afterEach } from 'vitest';

// Global test setup
beforeAll(async () => {
  console.log('Starting test suite');
});

afterAll(async () => {
  console.log('Test suite completed');
});

beforeEach(() => {
  // Reset mocks before each test
  vi.clearAllMocks();
});

afterEach(() => {
  // Cleanup after each test
});
```

**Mock Data Factories**:
```javascript
// tests/helpers/factories.js
import { faker } from '@faker-js/faker';

export const UserFactory = {
  build: (overrides = {}) => ({
    id: faker.string.uuid(),
    email: faker.internet.email(),
    name: faker.person.fullName(),
    createdAt: faker.date.past(),
    ...overrides
  }),

  buildList: (count, overrides = {}) => {
    return Array.from({ length: count }, () => UserFactory.build(overrides));
  }
};

export const OrderFactory = {
  build: (overrides = {}) => ({
    id: faker.string.uuid(),
    userId: faker.string.uuid(),
    total: faker.number.float({ min: 10, max: 1000, precision: 0.01 }),
    status: faker.helpers.arrayElement(['pending', 'completed', 'cancelled']),
    items: [],
    createdAt: faker.date.past(),
    ...overrides
  })
};
```

### Step 3: Write Unit Tests

**Test Structure Pattern**:

```javascript
// tests/unit/services/UserService.test.js
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { UserService } from '@/services/UserService';
import { UserRepository } from '@/repositories/UserRepository';
import { UserFactory } from '@/tests/helpers/factories';

// Mock dependencies
vi.mock('@/repositories/UserRepository');

describe('UserService', () => {
  let userService;
  let mockUserRepository;

  beforeEach(() => {
    mockUserRepository = new UserRepository();
    userService = new UserService(mockUserRepository);
  });

  describe('createUser', () => {
    it('should create a new user with valid data', async () => {
      const userData = {
        email: 'test@example.com',
        name: 'Test User',
        password: 'SecurePass123!'
      };

      const expectedUser = UserFactory.build(userData);
      mockUserRepository.create.mockResolvedValue(expectedUser);

      const result = await userService.createUser(userData);

      expect(mockUserRepository.create).toHaveBeenCalledWith(
        expect.objectContaining({
          email: userData.email,
          name: userData.name,
          password: expect.any(String) // hashed password
        })
      );
      expect(result).toEqual(expectedUser);
    });

    it('should hash password before storing', async () => {
      const userData = {
        email: 'test@example.com',
        name: 'Test User',
        password: 'PlainPassword'
      };

      await userService.createUser(userData);

      const callArgs = mockUserRepository.create.mock.calls[0][0];
      expect(callArgs.password).not.toBe(userData.password);
      expect(callArgs.password.length).toBeGreaterThan(20); // bcrypt hash
    });

    it('should validate email format', async () => {
      const userData = {
        email: 'invalid-email',
        name: 'Test User',
        password: 'SecurePass123!'
      };

      await expect(userService.createUser(userData))
        .rejects.toThrow('Invalid email format');

      expect(mockUserRepository.create).not.toHaveBeenCalled();
    });

    it('should throw error for duplicate email', async () => {
      const userData = UserFactory.build();
      mockUserRepository.create.mockRejectedValue(
        new Error('Email already exists')
      );

      await expect(userService.createUser(userData))
        .rejects.toThrow('Email already exists');
    });

    it('should trim whitespace from email and name', async () => {
      const userData = {
        email: '  test@example.com  ',
        name: '  Test User  ',
        password: 'SecurePass123!'
      };

      await userService.createUser(userData);

      expect(mockUserRepository.create).toHaveBeenCalledWith(
        expect.objectContaining({
          email: 'test@example.com',
          name: 'Test User'
        })
      );
    });
  });

  describe('getUserById', () => {
    it('should return user when found', async () => {
      const user = UserFactory.build();
      mockUserRepository.findById.mockResolvedValue(user);

      const result = await userService.getUserById(user.id);

      expect(result).toEqual(user);
      expect(mockUserRepository.findById).toHaveBeenCalledWith(user.id);
    });

    it('should throw error when user not found', async () => {
      mockUserRepository.findById.mockResolvedValue(null);

      await expect(userService.getUserById('non-existent-id'))
        .rejects.toThrow('User not found');
    });

    it('should validate id format', async () => {
      await expect(userService.getUserById('invalid-uuid'))
        .rejects.toThrow('Invalid user ID format');

      expect(mockUserRepository.findById).not.toHaveBeenCalled();
    });
  });
});
```

### Step 4: Test Edge Cases and Error Handling

**Comprehensive Test Coverage**:

```javascript
describe('calculateDiscount', () => {
  it('should calculate percentage discount correctly', () => {
    expect(calculateDiscount(100, 10)).toBe(90);
  });

  it('should handle zero discount', () => {
    expect(calculateDiscount(100, 0)).toBe(100);
  });

  it('should handle 100% discount', () => {
    expect(calculateDiscount(100, 100)).toBe(0);
  });

  it('should round to 2 decimal places', () => {
    expect(calculateDiscount(100, 33.333)).toBe(66.67);
  });

  it('should throw error for negative amounts', () => {
    expect(() => calculateDiscount(-100, 10))
      .toThrow('Amount must be positive');
  });

  it('should throw error for discount > 100%', () => {
    expect(() => calculateDiscount(100, 101))
      .toThrow('Discount cannot exceed 100%');
  });

  it('should throw error for negative discount', () => {
    expect(() => calculateDiscount(100, -10))
      .toThrow('Discount must be non-negative');
  });

  it('should handle very large numbers', () => {
    expect(calculateDiscount(1000000, 50)).toBe(500000);
  });

  it('should handle very small numbers', () => {
    expect(calculateDiscount(0.01, 50)).toBe(0.01);
  });
});
```

---

## Integration Testing Workflow

### Step 1: Setup Test Environment

**Test Database Configuration**:
```javascript
// tests/helpers/database.js
import { PrismaClient } from '@prisma/client';

let prisma;

export async function setupTestDatabase() {
  prisma = new PrismaClient({
    datasources: {
      db: {
        url: process.env.TEST_DATABASE_URL
      }
    }
  });

  await prisma.$connect();
  return prisma;
}

export async function cleanupTestDatabase() {
  const tables = await prisma.$queryRaw`
    SELECT tablename FROM pg_tables
    WHERE schemaname = 'public'
  `;

  for (const { tablename } of tables) {
    if (tablename !== '_prisma_migrations') {
      await prisma.$executeRawUnsafe(
        `TRUNCATE TABLE "public"."${tablename}" CASCADE;`
      );
    }
  }
}

export async function closeTestDatabase() {
  await prisma.$disconnect();
}

export { prisma };
```

### Step 2: Write API Integration Tests

**API Test Pattern**:

```javascript
// tests/integration/api/users.test.js
import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import request from 'supertest';
import { app } from '@/app';
import { setupTestDatabase, cleanupTestDatabase, closeTestDatabase } from '@/tests/helpers/database';
import { UserFactory } from '@/tests/helpers/factories';

describe('User API Integration Tests', () => {
  beforeAll(async () => {
    await setupTestDatabase();
  });

  afterAll(async () => {
    await closeTestDatabase();
  });

  beforeEach(async () => {
    await cleanupTestDatabase();
  });

  describe('POST /api/users', () => {
    it('should create user and return 201', async () => {
      const userData = {
        email: 'test@example.com',
        name: 'Test User',
        password: 'SecurePass123!'
      };

      const response = await request(app)
        .post('/api/users')
        .send(userData)
        .expect(201)
        .expect('Content-Type', /json/);

      expect(response.body).toMatchObject({
        id: expect.any(String),
        email: userData.email,
        name: userData.name,
        createdAt: expect.any(String)
      });
      expect(response.body).not.toHaveProperty('password');
    });

    it('should validate required fields', async () => {
      const response = await request(app)
        .post('/api/users')
        .send({})
        .expect(400);

      expect(response.body.errors).toContainEqual(
        expect.objectContaining({
          field: 'email',
          message: expect.stringMatching(/required/i)
        })
      );
    });
  });

  describe('GET /api/users/:id', () => {
    it('should retrieve existing user', async () => {
      const user = await createTestUser();

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
        .get('/api/users/00000000-0000-0000-0000-000000000000')
        .expect(404);

      expect(response.body.error).toMatch(/not found/i);
    });
  });
});
```

---

## E2E Testing Workflow

### Step 1: Setup Playwright

**Configuration**:
```javascript
// playwright.config.js
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',

  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure'
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] }
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] }
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] }
    }
  ],

  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI
  }
});
```

### Step 2: Create Page Objects

**Page Object Pattern**:
```javascript
// tests/e2e/pages/LoginPage.js
export class LoginPage {
  constructor(page) {
    this.page = page;
    this.emailInput = page.locator('input[name="email"]');
    this.passwordInput = page.locator('input[name="password"]');
    this.submitButton = page.locator('button[type="submit"]');
    this.errorMessage = page.locator('.error-message');
  }

  async goto() {
    await this.page.goto('/login');
  }

  async login(email, password) {
    await this.emailInput.fill(email);
    await this.passwordInput.fill(password);
    await this.submitButton.click();
  }

  async expectError(message) {
    await expect(this.errorMessage).toContainText(message);
  }
}
```

---

## Coverage Analysis

### Generate Reports

```bash
# Run tests with coverage
npm run test:coverage

# View HTML report
open coverage/index.html

# Check coverage thresholds
npm run test:coverage:check
```

### Identify Gaps

```bash
# Find uncovered files
npx vitest run --coverage --reporter=json | jq '.coverage.uncoveredFiles'

# Analyze coverage by directory
npx nyc report --reporter=text-summary
```

---

## CI/CD Integration

### GitHub Actions Configuration

See SKILL.md Phase 5 for complete CI/CD setup.

---

## Test Maintenance

### Regular Tasks

1. **Review Flaky Tests**: Investigate and fix tests with intermittent failures
2. **Update Test Data**: Keep factories and fixtures current
3. **Refactor Tests**: Follow DRY principles, extract common patterns
4. **Monitor Performance**: Track test execution time
5. **Update Documentation**: Keep test documentation synchronized

### Performance Optimization

- Use `test.concurrent` for independent tests
- Implement test data caching
- Optimize database operations
- Run tests in parallel
- Use coverage caching in CI
