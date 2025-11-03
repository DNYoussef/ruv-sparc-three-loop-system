# Testing Framework Skill - Quick Start Guide

## Overview

This skill provides a comprehensive testing framework implementation following industry best practices. It covers unit testing, integration testing, E2E testing, coverage analysis, and CI/CD integration.

## Quick Start

### Trigger Phrases

- "test this code"
- "add tests for [feature]"
- "create test suite"
- "implement testing framework"
- "run tests with coverage"

### Prerequisites

```bash
# Install testing dependencies
npm install -D vitest @vitest/ui @vitest/coverage-v8
npm install -D @playwright/test
npm install -D supertest
```

## Usage

### Basic Test Generation

```bash
# Via slash command
/test-generate --file src/utils/calculator.js --type unit

# Via MCP tool
mcp__testing-framework__generate_tests --file "calculator.js" --type "unit"
```

### Run Tests

```bash
# Run all tests
npm run test

# Run unit tests only
npm run test:unit

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

## Framework Support

### Unit & Integration Testing

- **Vitest** (recommended) - Fast, modern, Vite-native
- **Jest** - Popular, mature, extensive ecosystem

### E2E Testing

- **Playwright** (recommended) - Cross-browser, modern, fast
- **Cypress** - Developer-friendly, great DX

## Test Structure

```
tests/
├── unit/               # Isolated function tests
│   ├── utils/
│   ├── services/
│   └── models/
├── integration/        # Component interaction tests
│   ├── api/
│   ├── services/
│   └── database/
├── e2e/               # End-to-end user workflows
│   ├── pages/         # Page object models
│   └── fixtures/      # Test data
├── helpers/           # Test utilities
└── mocks/            # Mock data and services
```

## Coverage Goals

- **Critical Paths**: 90%+ coverage
- **Business Logic**: 80%+ coverage
- **Utilities**: 85%+ coverage
- **Overall Project**: 80%+ coverage

## Configuration Files

### Vitest Configuration

```javascript
// vitest.config.js
export default {
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'json'],
      thresholds: {
        lines: 80,
        functions: 80,
        branches: 75,
        statements: 80
      }
    }
  }
};
```

### Playwright Configuration

```javascript
// playwright.config.js
export default {
  testDir: './tests/e2e',
  use: {
    baseURL: 'http://localhost:3000',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure'
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } }
  ]
};
```

## Best Practices

### Unit Tests

- Test one thing per test
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Keep tests independent

### Integration Tests

- Test real component interactions
- Use test databases/services
- Implement proper setup/teardown
- Test happy and error paths
- Verify data flow

### E2E Tests

- Test critical user journeys
- Use page object models
- Handle async operations properly
- Implement retry logic
- Run in CI/CD pipeline

## Common Patterns

### Mock Data Factory

```javascript
// tests/helpers/factories.js
export function createTestUser(overrides = {}) {
  return {
    id: faker.string.uuid(),
    email: faker.internet.email(),
    name: faker.person.fullName(),
    ...overrides
  };
}
```

### Test Utilities

```javascript
// tests/helpers/testUtils.js
export async function clearDatabase() {
  await db.user.deleteMany();
  await db.order.deleteMany();
}

export async function createAuthenticatedRequest(app, user) {
  const token = generateAuthToken(user);
  return request(app).set('Authorization', `Bearer ${token}`);
}
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci
      - run: npm run test:coverage
      - uses: codecov/codecov-action@v3
```

## Troubleshooting

### Tests Running Slowly

- Use `test.concurrent` for independent tests
- Optimize database operations
- Use fixtures instead of factories
- Run unit and integration tests separately

### Flaky Tests

- Add proper wait conditions in E2E tests
- Avoid hardcoded timeouts
- Clear state between tests
- Use deterministic data

### Low Coverage

- Identify untested paths with `coverage/index.html`
- Prioritize critical business logic
- Test error scenarios
- Add integration tests for complex flows

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Best Practices](https://testingjavascript.com/)
- [Test-Driven Development Guide](https://martinfowler.com/bliki/TestDrivenDevelopment.html)

## Support

For issues or questions:
1. Check PROCESS.md for detailed workflows
2. Review test examples in skill directory
3. Consult testing framework documentation
4. Open issue in project repository
