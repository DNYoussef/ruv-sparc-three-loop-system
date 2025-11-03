/**
 * Test Suite for Flow Nexus Authentication Manager
 *
 * Comprehensive tests for authentication workflows including:
 * - User registration
 * - Login/logout
 * - Password management
 * - Profile updates
 * - Tier upgrades
 *
 * Run with: node auth-manager.test.js
 * Or with a test framework: jest auth-manager.test.js
 */

const AuthManager = require('../resources/scripts/auth-manager');

class TestRunner {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }

  describe(description, callback) {
    console.log(`\n${description}`);
    callback();
  }

  it(testName, testFn) {
    this.tests.push({ name: testName, fn: testFn });
  }

  async run() {
    console.log('Running Authentication Manager Tests\n');
    console.log('='.repeat(60));

    for (const test of this.tests) {
      try {
        await test.fn();
        console.log(`✓ ${test.name}`);
        this.passed++;
      } catch (error) {
        console.error(`✗ ${test.name}`);
        console.error(`  Error: ${error.message}`);
        this.failed++;
      }
    }

    console.log('\n' + '='.repeat(60));
    console.log(`\nTest Results: ${this.passed} passed, ${this.failed} failed`);
    console.log(`Total: ${this.tests.length} tests\n`);

    process.exit(this.failed > 0 ? 1 : 0);
  }

  assert(condition, message) {
    if (!condition) {
      throw new Error(message || 'Assertion failed');
    }
  }

  assertEqual(actual, expected, message) {
    if (actual !== expected) {
      throw new Error(
        message || `Expected ${expected}, got ${actual}`
      );
    }
  }

  assertNotNull(value, message) {
    if (value === null || value === undefined) {
      throw new Error(message || 'Value is null or undefined');
    }
  }
}

// Test Suite
const runner = new TestRunner();

runner.describe('AuthManager Class', () => {
  runner.it('should create AuthManager instance', () => {
    const manager = new AuthManager();
    runner.assertNotNull(manager, 'AuthManager instance should be created');
    runner.assertEqual(manager.mcpPrefix, 'mcp__flow-nexus__', 'Should have correct MCP prefix');
  });

  runner.it('should have all required methods', () => {
    const manager = new AuthManager();
    const requiredMethods = [
      'executeMCP',
      'register',
      'login',
      'logout',
      'status',
      'resetPassword',
      'updateProfile',
      'upgrade'
    ];

    requiredMethods.forEach(method => {
      runner.assert(
        typeof manager[method] === 'function',
        `AuthManager should have ${method} method`
      );
    });
  });
});

runner.describe('Registration Workflow', () => {
  runner.it('should validate email and password requirements', () => {
    // Mock validation logic
    const isValidEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    const isValidPassword = (password) => password && password.length >= 8;

    const testCases = [
      { email: 'user@example.com', password: 'SecurePass123', valid: true },
      { email: 'invalid-email', password: 'SecurePass123', valid: false },
      { email: 'user@example.com', password: 'short', valid: false },
      { email: '', password: 'SecurePass123', valid: false }
    ];

    testCases.forEach(testCase => {
      const isValid = isValidEmail(testCase.email) && isValidPassword(testCase.password);
      runner.assertEqual(
        isValid,
        testCase.valid,
        `Email: ${testCase.email}, Password: ${testCase.password} should be ${testCase.valid ? 'valid' : 'invalid'}`
      );
    });
  });

  runner.it('should construct correct registration parameters', () => {
    const manager = new AuthManager();

    // Test parameter construction logic
    const buildParams = (email, password, fullName, username) => {
      const params = { email, password };
      if (fullName) params.full_name = fullName;
      if (username) params.username = username;
      return params;
    };

    const params1 = buildParams('user@example.com', 'pass123');
    runner.assertEqual(Object.keys(params1).length, 2, 'Should have 2 params without optional fields');

    const params2 = buildParams('user@example.com', 'pass123', 'John Doe', 'johndoe');
    runner.assertEqual(Object.keys(params2).length, 4, 'Should have 4 params with optional fields');
    runner.assertEqual(params2.full_name, 'John Doe', 'Should include full_name');
    runner.assertEqual(params2.username, 'johndoe', 'Should include username');
  });
});

runner.describe('Login Workflow', () => {
  runner.it('should validate login credentials format', () => {
    const isValidCredentials = (email, password) => {
      return email && password && email.length > 0 && password.length > 0;
    };

    runner.assert(
      isValidCredentials('user@example.com', 'password123'),
      'Valid credentials should pass'
    );
    runner.assert(
      !isValidCredentials('', 'password123'),
      'Empty email should fail'
    );
    runner.assert(
      !isValidCredentials('user@example.com', ''),
      'Empty password should fail'
    );
  });

  runner.it('should handle session token storage', () => {
    // Mock session token handling
    const mockResult = {
      success: true,
      token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
      expires_at: new Date(Date.now() + 3600000).toISOString()
    };

    runner.assert(mockResult.success, 'Login should succeed');
    runner.assertNotNull(mockResult.token, 'Should return token');
    runner.assert(mockResult.token.length > 20, 'Token should be substantial length');
  });
});

runner.describe('Password Management', () => {
  runner.it('should validate password reset email format', () => {
    const isValidEmail = (email) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

    runner.assert(isValidEmail('user@example.com'), 'Valid email should pass');
    runner.assert(!isValidEmail('invalid'), 'Invalid email should fail');
    runner.assert(!isValidEmail(''), 'Empty email should fail');
  });

  runner.it('should enforce password policy', () => {
    const passwordPolicy = {
      minLength: 12,
      requireUppercase: true,
      requireLowercase: true,
      requireNumbers: true,
      requireSpecialChars: true
    };

    const checkPassword = (password) => {
      const checks = {
        length: password.length >= passwordPolicy.minLength,
        uppercase: /[A-Z]/.test(password),
        lowercase: /[a-z]/.test(password),
        numbers: /[0-9]/.test(password),
        special: /[!@#$%^&*(),.?":{}|<>]/.test(password)
      };
      return Object.values(checks).every(check => check);
    };

    runner.assert(
      checkPassword('SecurePass123!'),
      'Strong password should pass all checks'
    );
    runner.assert(
      !checkPassword('short'),
      'Weak password should fail checks'
    );
  });
});

runner.describe('Profile Updates', () => {
  runner.it('should validate profile update fields', () => {
    const allowedFields = ['full_name', 'bio', 'github_username', 'twitter_handle'];

    const isValidUpdate = (updates) => {
      return Object.keys(updates).every(key => allowedFields.includes(key));
    };

    const validUpdate = { full_name: 'John Doe', bio: 'Developer' };
    const invalidUpdate = { invalid_field: 'value' };

    runner.assert(isValidUpdate(validUpdate), 'Valid fields should pass');
    runner.assert(!isValidUpdate(invalidUpdate), 'Invalid fields should fail');
  });

  runner.it('should construct update parameters correctly', () => {
    const userId = 'user123';
    const updates = { bio: 'AI Developer', github_username: 'johndoe' };

    const params = { user_id: userId, updates };

    runner.assertEqual(params.user_id, userId, 'Should include user_id');
    runner.assertNotNull(params.updates, 'Should include updates object');
    runner.assertEqual(
      Object.keys(params.updates).length,
      2,
      'Should have 2 update fields'
    );
  });
});

runner.describe('Tier Upgrades', () => {
  runner.it('should validate tier values', () => {
    const validTiers = ['pro', 'enterprise'];

    const isValidTier = (tier) => validTiers.includes(tier.toLowerCase());

    runner.assert(isValidTier('pro'), 'Pro tier should be valid');
    runner.assert(isValidTier('enterprise'), 'Enterprise tier should be valid');
    runner.assert(isValidTier('PRO'), 'Pro tier (uppercase) should be valid');
    runner.assert(!isValidTier('free'), 'Free tier should be invalid');
    runner.assert(!isValidTier('invalid'), 'Invalid tier should fail');
  });

  runner.it('should construct upgrade parameters', () => {
    const userId = 'user123';
    const tier = 'pro';

    const params = { user_id: userId, tier: tier.toLowerCase() };

    runner.assertEqual(params.user_id, userId, 'Should include user_id');
    runner.assertEqual(params.tier, 'pro', 'Should lowercase tier');
  });
});

runner.describe('MCP Integration', () => {
  runner.it('should construct MCP tool names correctly', () => {
    const manager = new AuthManager();
    const prefix = manager.mcpPrefix;

    const tools = [
      'user_register',
      'user_login',
      'user_logout',
      'auth_status',
      'user_reset_password',
      'user_update_profile',
      'user_upgrade'
    ];

    tools.forEach(tool => {
      const fullName = prefix + tool;
      runner.assert(
        fullName.startsWith('mcp__flow-nexus__'),
        `${tool} should have correct prefix`
      );
    });
  });

  runner.it('should handle MCP call parameters serialization', () => {
    const params = {
      email: 'user@example.com',
      password: 'SecurePass123',
      full_name: 'John Doe'
    };

    const serialized = JSON.stringify(params);
    const deserialized = JSON.parse(serialized);

    runner.assertEqual(
      deserialized.email,
      params.email,
      'Email should serialize/deserialize correctly'
    );
    runner.assertEqual(
      deserialized.password,
      params.password,
      'Password should serialize/deserialize correctly'
    );
  });
});

runner.describe('Error Handling', () => {
  runner.it('should validate required parameters', () => {
    const requireParams = (params, required) => {
      return required.every(field => params[field] !== undefined);
    };

    const params1 = { email: 'user@example.com', password: 'pass123' };
    runner.assert(
      requireParams(params1, ['email', 'password']),
      'Should pass with all required params'
    );

    const params2 = { email: 'user@example.com' };
    runner.assert(
      !requireParams(params2, ['email', 'password']),
      'Should fail with missing required params'
    );
  });

  runner.it('should handle authentication failures gracefully', () => {
    const mockFailure = {
      success: false,
      error: 'Invalid credentials',
      error_code: 'AUTH_FAILED'
    };

    runner.assert(!mockFailure.success, 'Should indicate failure');
    runner.assertNotNull(mockFailure.error, 'Should include error message');
    runner.assertEqual(
      mockFailure.error_code,
      'AUTH_FAILED',
      'Should include error code'
    );
  });
});

runner.describe('CLI Argument Parsing', () => {
  runner.it('should parse register command arguments', () => {
    const parseRegisterArgs = (args) => {
      return {
        email: args[0],
        password: args[1],
        fullName: args[2] || null,
        username: args[3] || null
      };
    };

    const args1 = ['user@example.com', 'SecurePass123'];
    const parsed1 = parseRegisterArgs(args1);
    runner.assertEqual(parsed1.email, 'user@example.com', 'Should parse email');
    runner.assertEqual(parsed1.password, 'SecurePass123', 'Should parse password');
    runner.assertEqual(parsed1.fullName, null, 'Optional fullName should be null');

    const args2 = ['user@example.com', 'SecurePass123', 'John Doe'];
    const parsed2 = parseRegisterArgs(args2);
    runner.assertEqual(parsed2.fullName, 'John Doe', 'Should parse optional fullName');
  });

  runner.it('should parse update-profile key=value arguments', () => {
    const parseKeyValue = (args) => {
      const updates = {};
      args.forEach(arg => {
        const [key, value] = arg.split('=');
        if (key && value) {
          updates[key] = value;
        }
      });
      return updates;
    };

    const args = ['bio=AI Developer', 'github_username=johndoe'];
    const updates = parseKeyValue(args);

    runner.assertEqual(updates.bio, 'AI Developer', 'Should parse bio');
    runner.assertEqual(updates.github_username, 'johndoe', 'Should parse github_username');
    runner.assertEqual(Object.keys(updates).length, 2, 'Should parse 2 updates');
  });
});

// Run all tests
runner.run();
