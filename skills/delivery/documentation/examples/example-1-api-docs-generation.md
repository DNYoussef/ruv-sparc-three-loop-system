# Example 1: API Documentation Generation

## Overview
This example demonstrates how the documentation skill automatically generates comprehensive API documentation from existing code, including endpoint descriptions, parameter validation, response schemas, and interactive examples.

## Scenario
**Context**: You have a REST API built with Express.js and TypeScript that handles user authentication and profile management. The API has been implemented but lacks documentation.

**Goal**: Generate complete API documentation with OpenAPI 3.0 specification, interactive examples, and deployment-ready docs.

**Starting Point**:
- `src/api/auth.ts` - Authentication endpoints
- `src/api/users.ts` - User management endpoints
- `src/middleware/validation.ts` - Request validators
- No existing documentation

## Walkthrough

### Step 1: Invoke Documentation Skill

```bash
# User request
"Generate API documentation for our authentication and user management endpoints"

# Auto-trigger detection
Keywords detected: "API documentation", "generate"
Skill triggered: documentation
Agent spawned: api-documentation-specialist
```

### Step 2: Code Analysis Phase

The documentation skill analyzes your codebase:

```typescript
// Example source: src/api/auth.ts
import { Router, Request, Response } from 'express';
import { body, validationResult } from 'express-validator';
import { AuthService } from '../services/auth.service';

const router = Router();
const authService = new AuthService();

/**
 * @route POST /auth/register
 * @desc Register a new user
 * @access Public
 */
router.post(
  '/auth/register',
  [
    body('email').isEmail().normalizeEmail(),
    body('password').isLength({ min: 8 }).withMessage('Password must be at least 8 characters'),
    body('name').trim().notEmpty()
  ],
  async (req: Request, res: Response) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ errors: errors.array() });
    }

    try {
      const { email, password, name } = req.body;
      const user = await authService.register({ email, password, name });
      const token = await authService.generateToken(user.id);

      res.status(201).json({
        success: true,
        data: {
          user: { id: user.id, email: user.email, name: user.name },
          token
        }
      });
    } catch (error) {
      res.status(500).json({ success: false, error: 'Registration failed' });
    }
  }
);
```

**Skill Actions**:
1. Extracts route definitions (POST /auth/register)
2. Identifies validation rules (email format, password length)
3. Analyzes request/response schemas
4. Detects error handling patterns
5. Captures JSDoc comments

### Step 3: OpenAPI Specification Generation

```yaml
# Generated: docs/api/openapi.yaml
openapi: 3.0.3
info:
  title: User Authentication API
  description: REST API for user authentication and profile management
  version: 1.0.0
  contact:
    name: API Support
    email: support@example.com

servers:
  - url: https://api.example.com/v1
    description: Production server
  - url: https://staging-api.example.com/v1
    description: Staging server
  - url: http://localhost:3000/v1
    description: Development server

paths:
  /auth/register:
    post:
      summary: Register a new user
      description: Creates a new user account with email and password
      operationId: registerUser
      tags:
        - Authentication
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - email
                - password
                - name
              properties:
                email:
                  type: string
                  format: email
                  description: User's email address
                  example: user@example.com
                password:
                  type: string
                  format: password
                  minLength: 8
                  description: User's password (minimum 8 characters)
                  example: SecurePass123!
                name:
                  type: string
                  minLength: 1
                  description: User's full name
                  example: John Doe
            examples:
              validRegistration:
                summary: Valid registration request
                value:
                  email: john.doe@example.com
                  password: MySecurePassword123
                  name: John Doe
      responses:
        '201':
          description: User successfully registered
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                    example: true
                  data:
                    type: object
                    properties:
                      user:
                        $ref: '#/components/schemas/User'
                      token:
                        type: string
                        description: JWT authentication token
                        example: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
        '400':
          description: Validation error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ValidationError'
              examples:
                invalidEmail:
                  summary: Invalid email format
                  value:
                    errors:
                      - field: email
                        message: Must be a valid email address
        '500':
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          format: uuid
          example: 123e4567-e89b-12d3-a456-426614174000
        email:
          type: string
          format: email
          example: user@example.com
        name:
          type: string
          example: John Doe
        createdAt:
          type: string
          format: date-time
          example: 2024-01-15T10:30:00Z

    ValidationError:
      type: object
      properties:
        errors:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
              message:
                type: string

    Error:
      type: object
      properties:
        success:
          type: boolean
          example: false
        error:
          type: string
          example: Registration failed

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

### Step 4: Markdown Documentation Generation

```markdown
# Generated: docs/api/README.md
# API Documentation

## Authentication

### Register User

**Endpoint**: `POST /auth/register`

**Description**: Creates a new user account with email and password.

**Authentication**: None (public endpoint)

**Request Body**:

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| email | string | Yes | Valid email format | User's email address |
| password | string | Yes | Minimum 8 characters | User's password |
| name | string | Yes | Non-empty | User's full name |

**Example Request**:

```bash
curl -X POST https://api.example.com/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john.doe@example.com",
    "password": "MySecurePassword123",
    "name": "John Doe"
  }'
```

**Success Response (201)**:

```json
{
  "success": true,
  "data": {
    "user": {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "email": "john.doe@example.com",
      "name": "John Doe"
    },
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
}
```

**Error Responses**:

**400 Bad Request** - Validation error:
```json
{
  "errors": [
    {
      "field": "email",
      "message": "Must be a valid email address"
    }
  ]
}
```

**500 Internal Server Error**:
```json
{
  "success": false,
  "error": "Registration failed"
}
```
```

### Step 5: Interactive Documentation Setup

The skill generates Swagger UI integration:

```html
<!-- Generated: docs/api/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>API Documentation</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
  <style>
    body { margin: 0; padding: 0; }
    #swagger-ui { max-width: 1200px; margin: 0 auto; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = function() {
      SwaggerUIBundle({
        url: './openapi.yaml',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIBundle.SwaggerUIStandalonePreset
        ],
        plugins: [
          SwaggerUIBundle.plugins.DownloadUrl
        ],
        layout: "StandaloneLayout"
      });
    };
  </script>
</body>
</html>
```

## Outcomes

### Files Created
```
docs/
├── api/
│   ├── README.md              # Human-readable documentation
│   ├── openapi.yaml           # OpenAPI 3.0 specification
│   ├── index.html             # Interactive Swagger UI
│   └── postman-collection.json # Postman collection (auto-generated)
```

### Quality Metrics
- **Coverage**: 100% of API endpoints documented
- **Completeness**: All request/response schemas defined
- **Validation**: OpenAPI spec validates with no errors
- **Examples**: 3+ examples per endpoint
- **Time Saved**: ~4-6 hours manual documentation work

### Business Value
1. **Developer Experience**: New developers onboard 70% faster
2. **API Adoption**: External teams can integrate without asking questions
3. **Maintenance**: Auto-sync with code changes (via hooks)
4. **Testing**: Postman collection enables immediate API testing

## Code Generated

### Full Project Structure After Documentation

```
project/
├── src/
│   ├── api/
│   │   ├── auth.ts           # Original source
│   │   └── users.ts          # Original source
│   ├── services/
│   │   └── auth.service.ts   # Original source
│   └── middleware/
│       └── validation.ts     # Original source
├── docs/
│   ├── api/
│   │   ├── README.md         # ✨ Generated
│   │   ├── openapi.yaml      # ✨ Generated
│   │   ├── index.html        # ✨ Generated
│   │   └── postman-collection.json # ✨ Generated
│   └── deployment/
│       └── api-docs-deploy.md # ✨ Generated
├── .github/
│   └── workflows/
│       └── docs-deploy.yml   # ✨ Generated (auto-deploy to GitHub Pages)
└── package.json
```

## Tips & Best Practices

### 1. Enhance Code Comments for Better Docs
```typescript
// ❌ Minimal documentation
router.post('/auth/register', async (req, res) => {
  // implementation
});

// ✅ Rich documentation
/**
 * @route POST /auth/register
 * @desc Register a new user account with email verification
 * @access Public
 * @ratelimit 5 requests per hour per IP
 * @param {string} email - User's email address
 * @param {string} password - Password (min 8 chars, must include number and special char)
 * @param {string} name - User's full name
 * @returns {object} 201 - User object with JWT token
 * @returns {object} 400 - Validation errors
 * @returns {object} 409 - Email already exists
 * @returns {object} 500 - Server error
 * @example
 * Request:
 * {
 *   "email": "user@example.com",
 *   "password": "Secure123!",
 *   "name": "John Doe"
 * }
 */
router.post('/auth/register', async (req, res) => {
  // implementation
});
```

### 2. Use TypeScript Types for Schema Generation
```typescript
// Define types/interfaces - skill auto-generates schemas
interface RegisterRequest {
  email: string;
  password: string;
  name: string;
}

interface RegisterResponse {
  success: boolean;
  data: {
    user: User;
    token: string;
  };
}
```

### 3. Enable Auto-Sync with Git Hooks
```bash
# .git/hooks/pre-commit
#!/bin/bash
npx claude-flow sparc run documentation "Update API docs for changed files"
git add docs/
```

### 4. Version Your API Docs
```yaml
# Track API changes in docs/CHANGELOG.md
## v1.1.0 (2024-01-15)
- Added: POST /auth/reset-password
- Changed: POST /auth/register now requires email verification
- Deprecated: GET /users/all (use GET /users with pagination)
```

### 5. Integrate with CI/CD
```yaml
# .github/workflows/docs-deploy.yml
name: Deploy API Docs
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Generate docs
        run: npx claude-flow sparc run documentation "Generate API docs"
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

## Advanced Usage

### Multi-Version API Documentation
```bash
# Document v1 and v2 APIs separately
npx claude-flow sparc run documentation "Generate docs for v1 API in src/api/v1"
npx claude-flow sparc run documentation "Generate docs for v2 API in src/api/v2"
```

### Custom Templates
```bash
# Use custom OpenAPI template
npx claude-flow sparc run documentation \
  --template custom-openapi.yaml \
  "Generate API docs with custom branding"
```

### Integration with External Tools
```bash
# Generate docs + deploy to Readme.io
npx claude-flow sparc run documentation "Generate docs and sync to Readme.io"

# Generate docs + create Postman workspace
npx claude-flow sparc run documentation "Generate docs and publish to Postman"
```

## Troubleshooting

### Issue: Missing Request/Response Examples
**Solution**: Add JSDoc `@example` tags to your route handlers

### Issue: Incomplete Schema Validation
**Solution**: Use express-validator or Joi schemas explicitly

### Issue: Outdated Documentation
**Solution**: Enable git hooks for auto-regeneration on code changes

### Issue: Large API (100+ endpoints)
**Solution**: Use tags to organize documentation by domain/module

## Next Steps

1. **Explore Example 2**: Learn how to automate README generation
2. **Try the skill**: Run `npx claude-flow sparc run documentation "Generate API docs"`
3. **Customize**: Edit generated templates in `docs/templates/`
4. **Integrate**: Set up CI/CD for automatic documentation deployment
