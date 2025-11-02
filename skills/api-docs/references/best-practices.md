# API Documentation Best Practices

**Purpose**: Comprehensive guide to creating and maintaining high-quality API documentation that developers love

## 📋 Table of Contents

1. [Core Principles](#core-principles)
2. [Documentation Structure](#documentation-structure)
3. [Writing Guidelines](#writing-guidelines)
4. [Code Examples](#code-examples)
5. [Interactive Documentation](#interactive-documentation)
6. [Versioning & Deprecation](#versioning--deprecation)
7. [Maintenance & Updates](#maintenance--updates)
8. [Common Anti-Patterns](#common-anti-patterns)

## Core Principles

### 1. Developer-First Mindset

**Principle**: Write for developers who need to integrate quickly.

```markdown
✅ GOOD:
"Send a POST request to /api/v1/users with a JSON body containing email and name."

❌ BAD:
"The system processes incoming requests to the user creation endpoint."
```

**Key Elements**:
- Clear, actionable instructions
- Minimal jargon
- Quick-start examples
- Realistic use cases

### 2. Complete & Accurate

**Principle**: Document ALL endpoints, parameters, and responses.

```yaml
# ✅ Complete documentation
paths:
  /users:
    get:
      summary: List users
      parameters:
        - name: page
          in: query
          description: Page number for pagination
          required: false
          schema:
            type: integer
            default: 1
      responses:
        '200':
          description: Success
        '401':
          description: Unauthorized
        '500':
          description: Server error

# ❌ Incomplete documentation
paths:
  /users:
    get:
      summary: Get users
      responses:
        '200':
          description: OK
```

### 3. Stay Current

**Principle**: Documentation must match implementation.

**Best Practices**:
- Auto-generate from code annotations
- Run docs validation in CI/CD
- Version docs alongside code
- Tag docs with API version
- Archive old version docs

### 4. Show, Don't Tell

**Principle**: Provide working examples for everything.

```markdown
✅ GOOD:
```bash
curl -X POST https://api.example.com/v1/users \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","name":"John Doe"}'
```

Response:
```json
{
  "success": true,
  "data": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

❌ BAD:
"Use the POST method to create a user."
```

## Documentation Structure

### Recommended Organization

```
api-docs/
├── getting-started/
│   ├── quickstart.md           # 5-minute integration
│   ├── authentication.md       # Auth setup
│   └── first-request.md        # Hello World
│
├── guides/
│   ├── pagination.md           # How to paginate
│   ├── filtering.md            # Advanced filtering
│   ├── error-handling.md       # Error responses
│   └── rate-limiting.md        # Rate limits
│
├── reference/
│   ├── openapi.yaml            # OpenAPI spec
│   ├── users.md                # Users endpoint
│   ├── posts.md                # Posts endpoint
│   └── errors.md               # Error codes
│
├── changelog/
│   ├── v3.md                   # v3 changes
│   ├── v2.md                   # v2 changes
│   └── migration-v2-to-v3.md   # Migration guide
│
└── sdks/
    ├── javascript.md           # JavaScript SDK
    ├── python.md               # Python SDK
    └── curl.md                 # cURL examples
```

### Progressive Disclosure

**Layer 1: Quick Start** (5 minutes)
```markdown
# Quick Start

## 1. Get API Key
Sign up at https://example.com/signup

## 2. Make First Request
```bash
curl https://api.example.com/v1/users \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## 3. Next Steps
- [Authentication Guide](./authentication.md)
- [API Reference](./reference/openapi.yaml)
```

**Layer 2: Guides** (15-30 minutes)
- Conceptual explanations
- Common use cases
- Best practices
- Troubleshooting

**Layer 3: Reference** (Deep dive)
- Complete API reference
- All parameters documented
- All responses documented
- Edge cases covered

## Writing Guidelines

### Use Active Voice

```markdown
✅ GOOD:
"Send the request to /api/v1/users"
"The server returns a 201 status code"

❌ BAD:
"The request should be sent to /api/v1/users"
"A 201 status code will be returned by the server"
```

### Be Specific

```markdown
✅ GOOD:
"Rate limit: 100 requests per minute per API key"
"Response time: 95th percentile < 200ms"
"Max request size: 10MB"

❌ BAD:
"Rate limits apply"
"Fast response times"
"Large requests not supported"
```

### Document Errors Thoroughly

```yaml
responses:
  '400':
    description: Bad Request - Invalid input
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/Error'
        examples:
          invalid_email:
            summary: Invalid email format
            value:
              error: "Invalid email format"
              code: "VALIDATION_ERROR"
              field: "email"
              message: "Email must be a valid email address"

          missing_field:
            summary: Required field missing
            value:
              error: "Missing required field"
              code: "MISSING_FIELD"
              field: "name"
              message: "The 'name' field is required"
```

### Include Context

```markdown
✅ GOOD:
**Cursor-based pagination**

Use cursor-based pagination for better performance with large datasets.
Unlike offset pagination, cursors remain stable when data is added/removed.

Example:
```bash
# First page
GET /api/v1/users?limit=10

# Next page (use cursor from previous response)
GET /api/v1/users?limit=10&cursor=eyJpZCI6MTIzfQ==
```

❌ BAD:
"Use cursor parameter for pagination."
```

## Code Examples

### Multiple Languages

Provide examples in popular languages:

```markdown
# Creating a User

## cURL
```bash
curl -X POST https://api.example.com/v1/users \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","name":"John Doe"}'
```

## JavaScript (fetch)
```javascript
const response = await fetch('https://api.example.com/v1/users', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${YOUR_TOKEN}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    email: 'user@example.com',
    name: 'John Doe'
  })
});
const data = await response.json();
```

## Python (requests)
```python
import requests

response = requests.post(
    'https://api.example.com/v1/users',
    headers={'Authorization': f'Bearer {YOUR_TOKEN}'},
    json={'email': 'user@example.com', 'name': 'John Doe'}
)
data = response.json()
```
```

### Realistic Examples

```markdown
✅ GOOD: Realistic use case
```javascript
// Example: Create user, then create their first post
async function onboardNewUser(email, name) {
  // 1. Create user
  const userResponse = await fetch('/api/v1/users', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({email, name})
  });
  const user = await userResponse.json();

  // 2. Create welcome post
  const postResponse = await fetch('/api/v1/posts', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${user.token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      title: 'Welcome!',
      content: `Hello ${name}, welcome to our platform!`
    })
  });

  return await postResponse.json();
}
```

❌ BAD: Trivial example
```javascript
// Example
fetch('/api/v1/users')
```
```

## Interactive Documentation

### Swagger UI Best Practices

```yaml
# Enable try-it-out functionality
swagger:
  ui:
    tryItOutEnabled: true
    displayRequestDuration: true
    defaultModelsExpandDepth: 1
    defaultModelExpandDepth: 1
    docExpansion: list

# Pre-fill examples
components:
  schemas:
    CreateUserRequest:
      type: object
      example:
        email: "user@example.com"
        name: "John Doe"
        role: "user"
```

### GraphQL Playground Configuration

```javascript
app.get('/playground', expressPlayground({
  endpoint: '/graphql',
  settings: {
    'editor.theme': 'dark',
    'editor.reuseHeaders': true,
    'request.credentials': 'include'
  },
  tabs: [
    {
      endpoint: '/graphql',
      query: `# Example query
query GetUserWithPosts {
  user(id: "123") {
    name
    posts {
      title
    }
  }
}`
    }
  ]
}));
```

## Versioning & Deprecation

### Deprecation Notices

```yaml
paths:
  /users/{id}:
    get:
      summary: Get user (DEPRECATED)
      deprecated: true
      description: |
        ⚠️ **DEPRECATED**: This endpoint is deprecated and will be removed
        in v3.0.0 (sunset date: 2025-12-31).

        **Migration**: Use GET /v2/users/{id} instead, which returns
        UUID identifiers instead of integers.

        **Changes in v2**:
        - User IDs are now UUIDs instead of integers
        - Response includes `role` field
        - Timestamps are ISO 8601 strings

        **Migration Guide**: [v1 to v2 Migration](./migration/v1-to-v2.md)

      x-sunset-date: "2025-12-31"
      x-migration-guide: "/docs/migration/v1-to-v2"

      responses:
        '200':
          description: Success
          headers:
            X-API-Warn:
              schema:
                type: string
              description: Deprecation warning
              example: "This endpoint is deprecated. Use /v2/users/{id}"
            X-API-Sunset:
              schema:
                type: string
                format: date
              description: Sunset date
              example: "2025-12-31"
```

### Version-Specific Documentation

```markdown
# API Documentation v2

**Status**: Stable
**Released**: 2024-06-01
**Current**: ✅ Yes

## What's New in v2
- UUID-based identifiers
- ISO 8601 timestamps
- Cursor-based pagination
- Enhanced error messages

## Breaking Changes from v1
| v1 | v2 | Migration |
|----|----|----|
| Integer IDs | UUID IDs | Use UUID mapping table |
| Unix timestamps | ISO 8601 | Convert with `new Date(ts*1000).toISOString()` |
| Offset pagination | Cursor pagination | Switch to cursor-based |

## Migration Guide
See [v1 to v2 Migration Guide](./migration/v1-to-v2.md)
```

## Maintenance & Updates

### Documentation CI/CD

```yaml
# .github/workflows/docs.yml
name: Documentation

on:
  push:
    paths:
      - 'docs/**'
      - 'openapi.yaml'
      - 'src/routes/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Validate OpenAPI spec
        run: |
          npm install -g @apidevtools/swagger-cli
          swagger-cli validate openapi.yaml

      - name: Check for broken links
        run: |
          npm install -g markdown-link-check
          find docs -name "*.md" -exec markdown-link-check {} \;

      - name: Generate documentation
        run: npm run docs:generate

      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        run: npm run docs:deploy
```

### Automated Changelog

```javascript
// scripts/generate-changelog.js
const { execSync } = require('child_process');

// Get all commits since last tag
const commits = execSync('git log $(git describe --tags --abbrev=0)..HEAD --oneline')
  .toString()
  .split('\n')
  .filter(Boolean);

const changelog = {
  added: [],
  changed: [],
  deprecated: [],
  removed: [],
  fixed: [],
  security: []
};

commits.forEach(commit => {
  if (commit.includes('feat:')) changelog.added.push(commit);
  if (commit.includes('fix:')) changelog.fixed.push(commit);
  if (commit.includes('BREAKING:')) changelog.changed.push(commit);
  if (commit.includes('deprecated:')) changelog.deprecated.push(commit);
});

console.log(generateMarkdown(changelog));
```

## Common Anti-Patterns

### ❌ Anti-Pattern 1: Assuming Knowledge

```markdown
❌ BAD:
"Use JWT for authentication."

✅ GOOD:
"Authentication uses JWT (JSON Web Tokens). Obtain a token from
/auth/login, then include it in the Authorization header:

Authorization: Bearer YOUR_TOKEN

Tokens expire after 1 hour. Refresh using /auth/refresh."
```

### ❌ Anti-Pattern 2: Outdated Examples

```markdown
❌ BAD:
# Example from 2020 (outdated)
POST /api/users

✅ GOOD:
# Example (current as of API v2.0.0)
POST /api/v2/users

# Note: v1 endpoint /api/users is deprecated (sunset: 2025-12-31)
```

### ❌ Anti-Pattern 3: Missing Error Documentation

```markdown
❌ BAD:
Returns 400 if invalid.

✅ GOOD:
Returns 400 Bad Request with error details:
{
  "error": "Validation failed",
  "code": "VALIDATION_ERROR",
  "details": [
    {
      "field": "email",
      "message": "Invalid email format",
      "value": "not-an-email"
    }
  ]
}

Common validation errors:
- INVALID_EMAIL: Email format is invalid
- MISSING_FIELD: Required field not provided
- DUPLICATE_EMAIL: Email already registered
```

### ❌ Anti-Pattern 4: No Context for Rate Limits

```markdown
❌ BAD:
Rate limited.

✅ GOOD:
Rate Limiting
-------------
100 requests per minute per API key.

Headers:
- X-RateLimit-Limit: 100
- X-RateLimit-Remaining: 95
- X-RateLimit-Reset: 1640995200 (Unix timestamp)

When rate limited, you'll receive:
HTTP 429 Too Many Requests
{
  "error": "Rate limit exceeded",
  "code": "RATE_LIMIT_EXCEEDED",
  "retryAfter": 45
}
```

## Checklist

Use this checklist for every API documentation update:

- [ ] All endpoints documented
- [ ] All parameters documented with types and examples
- [ ] All responses documented (success + errors)
- [ ] Authentication requirements clear
- [ ] Rate limits specified
- [ ] Examples in multiple languages
- [ ] Error codes documented
- [ ] Migration guides for breaking changes
- [ ] Deprecation notices with sunset dates
- [ ] Interactive documentation tested
- [ ] OpenAPI spec validates
- [ ] Links checked for broken references
- [ ] Version information current
- [ ] Changelog updated

---

**Last Updated**: 2025-01-15
**Maintainer**: Claude Code (Sonnet 4.5)
**Feedback**: Create issue at repository
