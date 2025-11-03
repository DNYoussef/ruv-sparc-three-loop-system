# Driver-Navigator Mode Example
Comprehensive example of using driver-navigator pattern in pair programming

## Scenario: Building User Authentication Feature

This example demonstrates a complete driver-navigator session for implementing JWT-based authentication in an Express.js API.

---

## Session Setup

```bash
# Start driver-navigator session
claude-flow pair --start \
  --mode switch \
  --interval 10m \
  --agent senior-dev \
  --focus implementation \
  --verify \
  --test
```

**Configuration:**
- Mode: Switch (alternating driver/navigator every 10 minutes)
- Agent: senior-dev (experienced backend specialist)
- Focus: Implementation
- Verification: Enabled (0.95 threshold)
- Testing: Auto-run on changes

---

## Phase 1: Planning & Architecture (Driver: You, Navigator: AI)

### Minute 0-10: You as Driver

**Context:**
You're implementing user authentication with JWT tokens. The AI navigator will provide strategic guidance while you write the code structure.

**Your Actions as Driver:**

```javascript
// 1. Create directory structure
mkdir -p src/auth/{middleware,routes,controllers,services}
touch src/auth/auth.controller.js
touch src/auth/auth.service.js
touch src/auth/auth.routes.js
touch src/auth/jwt.middleware.js

// 2. Start with the service layer (business logic)
// File: src/auth/auth.service.js

const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const User = require('../models/User');

class AuthService {
  constructor() {
    this.saltRounds = 10;
    this.jwtSecret = process.env.JWT_SECRET;
    this.jwtRefreshSecret = process.env.JWT_REFRESH_SECRET;
  }

  async register(email, password, name) {
    // Check if user exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      throw new Error('User already exists');
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, this.saltRounds);

    // Create user
    const user = await User.create({
      email,
      password: hashedPassword,
      name
    });

    return this.generateTokens(user);
  }

  async login(email, password) {
    // Find user
    const user = await User.findOne({ email });
    if (!user) {
      throw new Error('Invalid credentials');
    }

    // Verify password
    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      throw new Error('Invalid credentials');
    }

    return this.generateTokens(user);
  }

  generateTokens(user) {
    const accessToken = jwt.sign(
      { userId: user._id, email: user.email },
      this.jwtSecret,
      { expiresIn: '15m' }
    );

    const refreshToken = jwt.sign(
      { userId: user._id },
      this.jwtRefreshSecret,
      { expiresIn: '7d' }
    );

    return { accessToken, refreshToken, user };
  }
}

module.exports = new AuthService();
```

**AI Navigator Feedback (Strategic Guidance):**

```
🧭 AI NAVIGATOR:

Good structure! Here are strategic considerations:

1. SECURITY CONCERNS:
   ✓ Password hashing with bcrypt (good)
   ⚠️  Consider rate limiting for login attempts
   ⚠️  Add refresh token rotation for better security
   ⚠️  Environment variables for secrets (make sure they're set)

2. ERROR HANDLING:
   ⚠️  Consider custom error classes for better error handling
   ⚠️  Don't leak information in error messages (user enumeration)

3. ARCHITECTURE:
   ✓ Service layer separation (good)
   💡 Consider adding a token blacklist for logout
   💡 Think about token refresh flow

4. TESTING:
   📝 You'll need to mock bcrypt and jwt for unit tests
   📝 Consider edge cases: empty inputs, SQL injection attempts

Suggested improvements for next phase:
- Add input validation
- Implement refresh token flow
- Add token blacklist for logout
- Create custom error classes

Continue? You have 2 minutes left as driver.
```

**Your Response:**

```
/note "AI suggested token refresh flow and blacklist - implement after basic auth works"
/explain refresh token rotation
```

**AI Explains:**

```
Refresh Token Rotation:

Instead of reusing the same refresh token, generate a new one each time
the access token is refreshed. This limits the damage if a refresh token
is compromised.

Flow:
1. Client sends refresh token
2. Server validates refresh token
3. If valid: generate NEW access token AND NEW refresh token
4. Invalidate old refresh token (add to blacklist)
5. Return both new tokens

Benefits:
- Limits window of vulnerability
- Enables token revocation
- Detects token theft (old token reuse = suspicious)

Implementation: Add a refresh endpoint and token blacklist cache (Redis).
```

---

## Phase 2: Implementation (Driver: AI, Navigator: You)

### Minute 10-20: AI as Driver

**⏰ Role Switch Warning:**
```
⚠️  Role switch in 30 seconds. Finish your current thought.
```

**Handoff Summary:**
```
╔══════════════════════════════════════════╗
║         ROLE HANDOFF SUMMARY             ║
╚══════════════════════════════════════════╝

Previous Role: DRIVER
New Role: NAVIGATOR

📊 Session Metrics:
  • Lines Changed: +85 -0
  • Files Modified: 1
  • Tests Added: 0
  • Coverage: 0.0%
  • Truth Score: 0.950

📝 Recent Activity:
  • code_change: src/auth/auth.service.js
  • note_added: decision

🧭 YOU ARE NAVIGATOR:
- Provide strategic direction
- Review code as AI writes
- Suggest improvements
- Think ahead about edge cases
- Keep big picture in mind

AI DRIVER PROVIDES:
- Implementation details
- Code generation
- Syntax handling
- Immediate bug fixes
```

**Your Instructions to AI Driver:**

```
/implement JWT middleware for protecting routes with token verification
/implement auth controller connecting service to routes
/implement auth routes with proper error handling
```

**AI Driver Generates:**

```javascript
// File: src/auth/jwt.middleware.js

const jwt = require('jsonwebtoken');

const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1]; // Bearer TOKEN

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid or expired token' });
    }

    req.user = user;
    next();
  });
};

const optionalAuth = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return next(); // No token, but continue
  }

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (!err) {
      req.user = user; // Token valid, attach user
    }
    next(); // Continue either way
  });
};

module.exports = { authenticateToken, optionalAuth };

// File: src/auth/auth.controller.js

const authService = require('./auth.service');

class AuthController {
  async register(req, res, next) {
    try {
      const { email, password, name } = req.body;

      // Input validation
      if (!email || !password || !name) {
        return res.status(400).json({
          error: 'Email, password, and name are required'
        });
      }

      if (password.length < 8) {
        return res.status(400).json({
          error: 'Password must be at least 8 characters'
        });
      }

      const result = await authService.register(email, password, name);

      res.status(201).json({
        message: 'User registered successfully',
        accessToken: result.accessToken,
        refreshToken: result.refreshToken,
        user: {
          id: result.user._id,
          email: result.user.email,
          name: result.user.name
        }
      });
    } catch (error) {
      if (error.message === 'User already exists') {
        return res.status(409).json({ error: error.message });
      }
      next(error);
    }
  }

  async login(req, res, next) {
    try {
      const { email, password } = req.body;

      if (!email || !password) {
        return res.status(400).json({
          error: 'Email and password are required'
        });
      }

      const result = await authService.login(email, password);

      res.json({
        message: 'Login successful',
        accessToken: result.accessToken,
        refreshToken: result.refreshToken,
        user: {
          id: result.user._id,
          email: result.user.email,
          name: result.user.name
        }
      });
    } catch (error) {
      if (error.message === 'Invalid credentials') {
        return res.status(401).json({ error: error.message });
      }
      next(error);
    }
  }

  async me(req, res) {
    // Protected route example
    res.json({
      user: {
        id: req.user.userId,
        email: req.user.email
      }
    });
  }
}

module.exports = new AuthController();

// File: src/auth/auth.routes.js

const express = require('express');
const router = express.Router();
const authController = require('./auth.controller');
const { authenticateToken } = require('./jwt.middleware');

// Public routes
router.post('/register', authController.register);
router.post('/login', authController.login);

// Protected routes
router.get('/me', authenticateToken, authController.me);

module.exports = router;
```

**Your Navigator Feedback:**

```
/review --security
```

**Security Review Results:**

```
🔒 SECURITY REVIEW:

✅ GOOD PRACTICES:
- Password hashing with bcrypt
- JWT for stateless authentication
- Bearer token pattern
- Input validation in controller
- Password length requirement (8+ chars)
- No password in response

⚠️  IMPROVEMENTS NEEDED:

1. RATE LIMITING (High Priority)
   - Add rate limiting on /register and /login
   - Prevent brute force attacks
   - Suggestion: express-rate-limit middleware

2. EMAIL VALIDATION (Medium Priority)
   - Add email format validation
   - Consider using validator.js or joi

3. CSRF PROTECTION (Medium Priority)
   - If using cookies, add CSRF tokens
   - Current implementation with Bearer tokens is okay

4. PASSWORD STRENGTH (Medium Priority)
   - 8 chars is minimum, but consider enforcing complexity
   - Numbers, uppercase, special characters

5. ERROR MESSAGES (Low Priority)
   - "User already exists" reveals email enumeration
   - Consider generic "Registration failed" message

Truth Score: 0.93 (Good, but security improvements recommended)

Fix high priority issues? (y/n)
```

**Your Response:**

```
/note "Security review: Add rate limiting and email validation - HIGH PRIORITY"
y
```

---

## Phase 3: Testing & Refinement (Driver: You, Navigator: AI)

### Minute 20-30: You as Driver Again

**You Write Tests:**

```javascript
// File: tests/auth.service.test.js

const authService = require('../src/auth/auth.service');
const User = require('../src/models/User');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

jest.mock('../src/models/User');
jest.mock('bcrypt');
jest.mock('jsonwebtoken');

describe('AuthService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('register', () => {
    it('should register a new user successfully', async () => {
      const mockUser = {
        _id: 'user123',
        email: 'test@example.com',
        name: 'Test User'
      };

      User.findOne.mockResolvedValue(null);
      bcrypt.hash.mockResolvedValue('hashedPassword');
      User.create.mockResolvedValue(mockUser);
      jwt.sign.mockReturnValue('mockToken');

      const result = await authService.register(
        'test@example.com',
        'password123',
        'Test User'
      );

      expect(result).toHaveProperty('accessToken');
      expect(result).toHaveProperty('refreshToken');
      expect(result.user).toEqual(mockUser);
    });

    it('should throw error if user already exists', async () => {
      User.findOne.mockResolvedValue({ email: 'existing@example.com' });

      await expect(
        authService.register('existing@example.com', 'password', 'Name')
      ).rejects.toThrow('User already exists');
    });
  });

  describe('login', () => {
    it('should login user with valid credentials', async () => {
      const mockUser = {
        _id: 'user123',
        email: 'test@example.com',
        password: 'hashedPassword'
      };

      User.findOne.mockResolvedValue(mockUser);
      bcrypt.compare.mockResolvedValue(true);
      jwt.sign.mockReturnValue('mockToken');

      const result = await authService.login('test@example.com', 'password123');

      expect(result).toHaveProperty('accessToken');
      expect(result).toHaveProperty('refreshToken');
    });

    it('should throw error for invalid password', async () => {
      const mockUser = {
        _id: 'user123',
        email: 'test@example.com',
        password: 'hashedPassword'
      };

      User.findOne.mockResolvedValue(mockUser);
      bcrypt.compare.mockResolvedValue(false);

      await expect(
        authService.login('test@example.com', 'wrongpassword')
      ).rejects.toThrow('Invalid credentials');
    });
  });
});
```

**Run Tests:**

```bash
npm test
```

**Test Results:**

```
 PASS  tests/auth.service.test.js
  AuthService
    register
      ✓ should register a new user successfully (15ms)
      ✓ should throw error if user already exists (5ms)
    login
      ✓ should login user with valid credentials (8ms)
      ✓ should throw error for invalid password (6ms)

Test Suites: 1 passed, 1 total
Tests:       4 passed, 4 total
Coverage:    87.5% lines, 85.0% branches
```

---

## Session Metrics & Summary

```bash
/metrics
```

**Final Metrics:**

```json
{
  "sessionId": "pair_a1b2c3d4e5f6",
  "duration": 1800,
  "mode": "switch",
  "switches": 3,

  "code": {
    "linesAdded": 287,
    "linesRemoved": 12,
    "filesCreated": 5,
    "filesModified": 5
  },

  "testing": {
    "testsWritten": 4,
    "testsPassing": 4,
    "testsFailing": 0,
    "coverage": 87.5
  },

  "quality": {
    "avgTruthScore": 0.947,
    "minTruthScore": 0.930,
    "maxTruthScore": 0.965,
    "commits": 1,
    "rollbacks": 0
  },

  "collaboration": {
    "roleSwitches": 3,
    "driverTime": 900,
    "navigatorTime": 900,
    "commandsExecuted": 12
  }
}
```

**Session Summary:**

```
✅ COMPLETED: User Authentication Feature

What We Built:
- JWT-based authentication service
- Register and login endpoints
- Password hashing with bcrypt
- Token-based authorization middleware
- Comprehensive test suite (87.5% coverage)

Security Measures:
✓ Password hashing
✓ JWT tokens (15m access, 7d refresh)
✓ Input validation
✓ Proper error handling
⚠️  TODO: Rate limiting
⚠️  TODO: Email validation

Next Steps:
1. Add rate limiting middleware
2. Implement refresh token rotation
3. Add token blacklist for logout
4. Improve email validation
5. Add integration tests

Truth Score: 0.947 ✅
Ready for commit: YES
```

---

## Key Takeaways from Driver-Navigator Mode

### As Driver:
- **Write code** while getting strategic guidance
- **Ask questions** when stuck or uncertain
- **Stay focused** on implementation details
- **Use navigator feedback** to improve as you code

### As Navigator:
- **Think ahead** about architecture and edge cases
- **Provide guidance** without micromanaging
- **Review continuously** as code is written
- **Suggest improvements** and best practices

### Benefits:
✅ Continuous code review
✅ Knowledge sharing in real-time
✅ Better design decisions
✅ Higher code quality
✅ Faster learning
✅ Reduced bugs

### When to Use:
- Learning new technologies
- Complex feature implementation
- Code refactoring
- Debugging difficult issues
- Knowledge transfer sessions
