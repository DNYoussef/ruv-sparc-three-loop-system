# Code Review Categories - Comprehensive Reference

This document provides a detailed breakdown of the review categories used by the Code Review Assistant's multi-agent swarm.

## 1. Security Review

### Agent: Security Reviewer
**Specialization**: Vulnerability detection, secure coding practices, OWASP compliance

### Review Areas

#### 1.1 Injection Vulnerabilities
- **SQL Injection**: Unparameterized queries, string concatenation
- **NoSQL Injection**: Unsafe MongoDB queries, object injection
- **Command Injection**: Unsafe shell command execution
- **LDAP Injection**: Unvalidated LDAP queries
- **XPath Injection**: Unsafe XML queries

**Detection Patterns**:
```javascript
// SQL Injection
db.query(`SELECT * FROM users WHERE id=${userId}`)  // ❌ Vulnerable
db.query('SELECT * FROM users WHERE id=?', [userId]) // ✅ Safe

// Command Injection
exec(`git clone ${userInput}`) // ❌ Vulnerable
execFile('git', ['clone', userInput]) // ✅ Safe
```

#### 1.2 Cross-Site Scripting (XSS)
- **Reflected XSS**: User input reflected in response
- **Stored XSS**: Persistent malicious scripts in database
- **DOM-based XSS**: Client-side script manipulation

**Detection Patterns**:
```javascript
// React XSS
<div>{userInput}</div> // ✅ Safe (React auto-escapes)
<div dangerouslySetInnerHTML={{ __html: userInput }} /> // ❌ Vulnerable

// innerHTML XSS
element.innerHTML = userInput; // ❌ Vulnerable
element.textContent = userInput; // ✅ Safe
```

#### 1.3 Authentication & Authorization
- **Broken Authentication**: Weak password policies, session management
- **Broken Access Control**: Missing authorization checks, IDOR vulnerabilities
- **JWT Issues**: Weak secrets, missing expiration, algorithm confusion

#### 1.4 Sensitive Data Exposure
- **Hardcoded Secrets**: API keys, passwords in code
- **Logging Sensitive Data**: Passwords, tokens in logs
- **Insecure Storage**: Plaintext passwords, unencrypted data

#### 1.5 Dependencies & Supply Chain
- **Outdated Dependencies**: Known CVEs in packages
- **Malicious Packages**: Typosquatting, compromised packages
- **License Compliance**: Incompatible licenses

### Severity Levels

| Level | CVSS Score | Examples | Action |
|-------|------------|----------|--------|
| CRITICAL | 9.0-10.0 | SQL injection, RCE, Authentication bypass | Block merge |
| HIGH | 7.0-8.9 | XSS, Insecure deserialization, CSRF | Request changes |
| MEDIUM | 4.0-6.9 | Information disclosure, weak crypto | Suggest fix |
| LOW | 0.1-3.9 | Security misconfiguration, verbose errors | Optional fix |

### OWASP Top 10 (2021) Mapping

1. **A01 - Broken Access Control**: Authorization checks, IDOR, path traversal
2. **A02 - Cryptographic Failures**: Encryption, hashing, key management
3. **A03 - Injection**: SQL, NoSQL, command, LDAP injection
4. **A04 - Insecure Design**: Threat modeling, secure patterns
5. **A05 - Security Misconfiguration**: Default configs, unnecessary features
6. **A06 - Vulnerable Components**: Outdated dependencies, known CVEs
7. **A07 - Authentication Failures**: Session management, password policies
8. **A08 - Software/Data Integrity**: CI/CD security, unsigned packages
9. **A09 - Security Logging Failures**: Insufficient logging, missing alerts
10. **A10 - Server-Side Request Forgery**: SSRF vulnerabilities

---

## 2. Performance Review

### Agent: Performance Analyst
**Specialization**: Bottleneck detection, algorithm optimization, scalability

### Review Areas

#### 2.1 Algorithm Complexity
- **Time Complexity**: O(n), O(n²), O(n log n) analysis
- **Space Complexity**: Memory usage patterns
- **Unnecessary Computations**: Redundant calculations

**Thresholds**:
- ✅ O(1), O(log n), O(n): Acceptable
- ⚠️ O(n log n): Review needed
- ❌ O(n²), O(2^n): Refactor required

#### 2.2 Database Performance
- **N+1 Query Problem**: Sequential queries in loops
- **Missing Indexes**: Unindexed WHERE/JOIN columns
- **Full Table Scans**: SELECT * without WHERE
- **Missing Pagination**: Loading entire tables

#### 2.3 Memory Management
- **Memory Leaks**: Unbounded caches, event listener leaks
- **Large Object Allocation**: Unnecessary large data structures
- **Streaming Opportunities**: Processing large files in memory

#### 2.4 Network Performance
- **API Over-fetching**: Requesting unnecessary fields
- **Missing Caching**: No HTTP cache headers
- **Bundle Size**: Large JavaScript bundles
- **Resource Loading**: Unoptimized images, fonts

#### 2.5 Concurrency & Parallelism
- **Sequential Processing**: Opportunities for parallelization
- **Blocking Operations**: Synchronous I/O, long computations
- **Race Conditions**: Unsafe concurrent access

### Performance Metrics

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| API Response Time | <100ms | 100-500ms | >500ms |
| Database Query | <10ms | 10-50ms | >50ms |
| Page Load (First Contentful Paint) | <1s | 1-3s | >3s |
| Bundle Size | <100KB | 100-500KB | >500KB |
| Memory Usage | <50MB | 50-200MB | >200MB |

---

## 3. Style & Maintainability Review

### Agent: Style Reviewer
**Specialization**: Clean code principles, coding standards, best practices

### Review Areas

#### 3.1 Code Style Consistency
- **Naming Conventions**: camelCase, PascalCase, UPPER_SNAKE_CASE
- **Formatting**: Indentation, spacing, line length
- **Import Organization**: Grouping, ordering
- **Quote Style**: Single vs double quotes

#### 3.2 Clean Code Principles

##### Single Responsibility Principle (SRP)
- One function = one responsibility
- One class = one reason to change

##### DRY (Don't Repeat Yourself)
- Code duplication >3 instances
- Extract to reusable functions

##### YAGNI (You Aren't Gonna Need It)
- Avoid premature abstraction
- No unused code paths

##### KISS (Keep It Simple, Stupid)
- Avoid unnecessary complexity
- Clear over clever

#### 3.3 Code Complexity Metrics

**Cyclomatic Complexity**:
- ✅ 1-10: Simple, easy to test
- ⚠️ 11-20: Complex, review needed
- ❌ >20: High risk, refactor required

**Cognitive Complexity**:
- Measures "how hard to understand"
- Nesting adds more weight than sequential code

**NASA Coding Standards**:
- Max function length: 50 lines
- Max nesting depth: 4 levels
- Max parameters: 6 parameters
- Max file length: 500 lines

#### 3.4 Code Smells

| Smell | Description | Fix |
|-------|-------------|-----|
| God Object | Class with too many responsibilities | Split into smaller classes |
| Long Method | Function >50 lines | Extract to smaller functions |
| Long Parameter List | >6 parameters | Use object parameter |
| Duplicate Code | Same code in multiple places | Extract to reusable function |
| Dead Code | Unused variables, unreachable code | Remove |
| Magic Numbers | Hardcoded numbers | Extract to named constants |
| Shotgun Surgery | Small change requires many file edits | Improve cohesion |
| Feature Envy | Method uses another class's data | Move method |

#### 3.5 Design Patterns

**Recommended Patterns**:
- **Factory**: Object creation
- **Strategy**: Algorithm selection
- **Observer**: Event handling
- **Decorator**: Dynamic behavior addition
- **Repository**: Data access abstraction
- **Dependency Injection**: Loose coupling

**Anti-Patterns to Avoid**:
- **God Object**: Too many responsibilities
- **Spaghetti Code**: Tangled control flow
- **Golden Hammer**: One solution for all problems
- **Premature Optimization**: Optimize before measuring

---

## 4. Test Coverage Review

### Agent: Test Specialist
**Specialization**: Test quality, coverage analysis, test strategy

### Review Areas

#### 4.1 Test Coverage Metrics

**Types of Coverage**:
- **Line Coverage**: Percentage of lines executed
- **Branch Coverage**: Percentage of if/else branches tested
- **Function Coverage**: Percentage of functions called
- **Statement Coverage**: Percentage of statements executed

**Coverage Thresholds**:
- ✅ ≥80%: Excellent
- ⚠️ 60-79%: Acceptable
- ❌ <60%: Insufficient

#### 4.2 Test Quality

**Test Pyramid**:
```
       /\
      /E2E\       10% - End-to-End Tests
     /______\
    /Integr.\    20% - Integration Tests
   /__________\
  /Unit Tests \  70% - Unit Tests
 /______________\
```

**Test Types**:
1. **Unit Tests**: Individual functions/methods
2. **Integration Tests**: Component interactions
3. **End-to-End Tests**: Full user workflows
4. **Performance Tests**: Load, stress, endurance
5. **Security Tests**: Vulnerability scanning

#### 4.3 Test Best Practices

**AAA Pattern** (Arrange-Act-Assert):
```javascript
test('should calculate total price', () => {
  // Arrange
  const cart = new Cart();
  cart.addItem({ price: 10, quantity: 2 });

  // Act
  const total = cart.calculateTotal();

  // Assert
  expect(total).toBe(20);
});
```

**Test Independence**:
- No shared state between tests
- Each test sets up own fixtures
- Order-independent execution

**Test Naming**:
```javascript
// ✅ Good
test('should return 404 when user not found')
test('should hash password before saving')

// ❌ Bad
test('test1')
test('user test')
```

#### 4.4 Edge Cases & Boundary Conditions

**Essential Edge Cases**:
- Empty arrays/strings
- Null/undefined values
- Maximum/minimum values
- Negative numbers
- Concurrent access
- Network failures
- Timeout scenarios

#### 4.5 Mock & Stub Quality

**When to Mock**:
- External APIs
- Database calls
- File system operations
- Time-dependent logic

**Mock Best Practices**:
```javascript
// ✅ Good - Mock external dependencies
const mockFetch = jest.fn().mockResolvedValue({ data: 'test' });

// ❌ Bad - Don't mock internal logic
const mockCalculate = jest.fn(); // Test real implementation instead
```

---

## 5. Documentation Review

### Agent: Documentation Reviewer
**Specialization**: API docs, code comments, README completeness

### Review Areas

#### 5.1 Code Comments

**When to Comment**:
- **WHY**: Explain reasoning, not WHAT
- **Algorithms**: Complex logic explanation
- **Workarounds**: Why hack exists
- **TODOs**: With ticket numbers

**When NOT to Comment**:
- **Self-Explanatory Code**: Good names over comments
- **Obsolete Comments**: Update or remove

```javascript
// ❌ Bad - Obvious comment
// Increment i
i++;

// ✅ Good - Explains reasoning
// Use binary search instead of linear for O(log n) performance
const index = binarySearch(array, target);

// ✅ Good - Documents workaround
// HACK: Safari doesn't support CSS aspect-ratio
// TODO: Remove when Safari 15+ adoption >95% (Ticket: UI-1234)
```

#### 5.2 JSDoc / TypeDoc

**Required for Public APIs**:
```javascript
/**
 * Calculates the total price including tax and discount
 * @param {number} price - Base price before tax
 * @param {number} taxRate - Tax rate (0-1, e.g., 0.08 for 8%)
 * @param {number} [discountPercent=0] - Optional discount (0-100)
 * @returns {number} Final price after tax and discount
 * @throws {Error} If price or taxRate is negative
 * @example
 * calculateTotal(100, 0.08, 10); // Returns 97.20
 */
function calculateTotal(price, taxRate, discountPercent = 0) {
  // Implementation
}
```

#### 5.3 API Documentation

**Required Sections**:
1. **Overview**: What the API does
2. **Authentication**: How to authenticate
3. **Endpoints**: List all routes
4. **Request/Response Examples**: Concrete examples
5. **Error Codes**: All possible errors
6. **Rate Limiting**: Usage limits

**OpenAPI / Swagger**:
```yaml
/users/{id}:
  get:
    summary: Get user by ID
    parameters:
      - name: id
        in: path
        required: true
        schema:
          type: integer
    responses:
      200:
        description: User found
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/User'
      404:
        description: User not found
```

#### 5.4 README Documentation

**Essential Sections**:
1. **Project Description**: One-liner + detailed description
2. **Installation**: Step-by-step setup
3. **Usage**: Basic examples
4. **Configuration**: Environment variables, config files
5. **API Reference**: Link to API docs
6. **Contributing**: Contribution guidelines
7. **License**: License type
8. **Support**: How to get help

#### 5.5 Changelog

**Keep a Changelog Format**:
```markdown
# Changelog

## [1.2.0] - 2024-03-15

### Added
- User authentication with JWT
- Password reset functionality

### Changed
- Improved error messages in API responses
- Updated dependencies to latest versions

### Fixed
- SQL injection vulnerability in login endpoint
- Memory leak in WebSocket connections

### Security
- Patched CVE-2024-1234 in dependency X
```

---

## Scoring & Merge Decision

### Overall Score Calculation

```
Overall Score = (
  Security Score * 0.30 +      // 30% weight
  Performance Score * 0.25 +   // 25% weight
  Style Score * 0.20 +         // 20% weight
  Test Score * 0.15 +          // 15% weight
  Documentation Score * 0.10   // 10% weight
)
```

### Merge Decision Logic

```
IF critical_security_issues > 0:
    DECISION = "request_changes"

ELIF all_tests_passing == false:
    DECISION = "request_changes"

ELIF overall_score >= 90:
    DECISION = "approve"

ELIF overall_score >= 80:
    DECISION = "approve_with_suggestions"

ELSE:
    DECISION = "request_changes"
```

### Quality Gates

| Gate | Requirement | Blocker? |
|------|-------------|----------|
| Security | 0 critical issues | Yes |
| Performance | No O(n²) in hot paths | No |
| Style | Cyclomatic complexity <20 | No |
| Tests | All tests passing | Yes |
| Tests | Coverage ≥60% | No |
| Documentation | Public APIs documented | No |

---

## Best Practices Summary

### DO
- ✅ Use parameterized queries
- ✅ Hash passwords with bcrypt
- ✅ Add pagination to all list endpoints
- ✅ Write tests before implementation (TDD)
- ✅ Document public APIs with JSDoc
- ✅ Use meaningful variable names
- ✅ Keep functions under 50 lines
- ✅ Review your own code before PR

### DON'T
- ❌ Hardcode secrets or URLs
- ❌ Use console.log in production
- ❌ Nest code more than 4 levels deep
- ❌ Have functions with >6 parameters
- ❌ Write duplicate code
- ❌ Commit commented-out code
- ❌ Ignore linter warnings
- ❌ Skip writing tests

---

## Related Resources

- [OWASP Top 10](https://owasp.org/Top10/)
- [Clean Code by Robert C. Martin](https://www.oreilly.com/library/view/clean-code/9780136083238/)
- [NASA C Coding Standards](https://ntrs.nasa.gov/api/citations/19950022400/downloads/19950022400.pdf)
- [Google Style Guides](https://google.github.io/styleguide/)
- [The Pragmatic Programmer](https://pragprog.com/titles/tpp20/the-pragmatic-programmer-20th-anniversary-edition/)
