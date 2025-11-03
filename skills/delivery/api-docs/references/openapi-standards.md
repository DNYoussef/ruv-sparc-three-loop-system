# OpenAPI 3.0 Standards & Best Practices

**Version**: OpenAPI 3.0.3
**Purpose**: Complete reference for OpenAPI specification standards, validation, and best practices

## 📋 Table of Contents

1. [OpenAPI Specification Overview](#openapi-specification-overview)
2. [Core Structure](#core-structure)
3. [Schema Definitions](#schema-definitions)
4. [Security Schemes](#security-schemes)
5. [Validation Rules](#validation-rules)
6. [Best Practices](#best-practices)
7. [Common Patterns](#common-patterns)
8. [Tooling Ecosystem](#tooling-ecosystem)

## OpenAPI Specification Overview

OpenAPI (formerly Swagger) is a language-agnostic specification for describing RESTful APIs. Version 3.0 introduced significant improvements over 2.0 (Swagger).

### Key Features
- **Machine-readable**: Parseable by tools for code generation
- **Human-readable**: YAML/JSON format easy to write and read
- **Self-documenting**: API documentation auto-generated from spec
- **Validation**: Request/response validation against schemas
- **Code generation**: Client SDKs and server stubs

### Version Differences

| Feature | OpenAPI 2.0 (Swagger) | OpenAPI 3.0 |
|---------|----------------------|-------------|
| Format | JSON only | JSON or YAML |
| Servers | Single host + basePath | Multiple servers with variables |
| Request bodies | Part of parameters | Separate requestBody object |
| Callbacks | Not supported | Native callback support |
| Links | Not supported | Native link support |
| Examples | Single example | Multiple examples |
| Components | Definitions | Reusable components |

## Core Structure

### Minimal Valid Spec

```yaml
openapi: 3.0.3
info:
  title: Minimal API
  version: 1.0.0
paths:
  /health:
    get:
      responses:
        '200':
          description: OK
```

### Complete Structure

```yaml
openapi: 3.0.3

info:
  title: Complete API
  description: Detailed API description
  version: 1.0.0
  termsOfService: https://example.com/terms
  contact:
    name: API Support
    email: support@example.com
    url: https://support.example.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: https://api.example.com/v1
    description: Production
    variables:
      environment:
        default: prod
        enum: [prod, staging]
  - url: https://staging.example.com/v1
    description: Staging

tags:
  - name: Users
    description: User management operations
  - name: Posts
    description: Blog post operations

paths:
  /users:
    get:
      summary: List users
      tags: [Users]
      parameters:
        - $ref: '#/components/parameters/PageParam'
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserList'

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          format: uuid
        email:
          type: string
          format: email
      required:
        - id
        - email

  parameters:
    PageParam:
      name: page
      in: query
      schema:
        type: integer
        default: 1

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - bearerAuth: []
```

## Schema Definitions

### Data Types

OpenAPI supports all JSON Schema data types:

```yaml
components:
  schemas:
    DataTypes:
      type: object
      properties:
        # Primitives
        stringField:
          type: string
          minLength: 1
          maxLength: 100
          pattern: '^[A-Za-z]+$'

        numberField:
          type: number
          minimum: 0
          maximum: 100
          multipleOf: 0.01

        integerField:
          type: integer
          minimum: 0
          maximum: 100
          exclusiveMinimum: false

        booleanField:
          type: boolean
          default: false

        # Arrays
        arrayField:
          type: array
          items:
            type: string
          minItems: 1
          maxItems: 10
          uniqueItems: true

        # Objects
        objectField:
          type: object
          properties:
            nestedField:
              type: string
          required:
            - nestedField

        # Enums
        enumField:
          type: string
          enum: [option1, option2, option3]

        # Nullable (OpenAPI 3.0)
        nullableField:
          type: string
          nullable: true

        # Any type
        anyField:
          oneOf:
            - type: string
            - type: number
```

### Format Specifications

```yaml
components:
  schemas:
    Formats:
      type: object
      properties:
        # Date/Time
        dateField:
          type: string
          format: date           # YYYY-MM-DD
          example: "2025-01-15"

        dateTimeField:
          type: string
          format: date-time      # RFC 3339
          example: "2025-01-15T10:30:00Z"

        # Numbers
        floatField:
          type: number
          format: float

        doubleField:
          type: number
          format: double

        int32Field:
          type: integer
          format: int32

        int64Field:
          type: integer
          format: int64

        # Strings
        passwordField:
          type: string
          format: password       # Masked in UI

        emailField:
          type: string
          format: email

        uuidField:
          type: string
          format: uuid
          example: "123e4567-e89b-12d3-a456-426614174000"

        uriField:
          type: string
          format: uri
          example: "https://example.com/resource"

        # Binary
        binaryField:
          type: string
          format: binary         # Arbitrary binary data

        byteField:
          type: string
          format: byte           # Base64 encoded
```

### Schema Composition

```yaml
components:
  schemas:
    # allOf: Merge schemas (AND)
    Employee:
      allOf:
        - $ref: '#/components/schemas/Person'
        - type: object
          properties:
            employeeId:
              type: string

    # oneOf: Exactly one schema matches (XOR)
    Pet:
      oneOf:
        - $ref: '#/components/schemas/Dog'
        - $ref: '#/components/schemas/Cat'
      discriminator:
        propertyName: petType
        mapping:
          dog: '#/components/schemas/Dog'
          cat: '#/components/schemas/Cat'

    # anyOf: One or more schemas match (OR)
    Contact:
      anyOf:
        - type: object
          properties:
            email:
              type: string
        - type: object
          properties:
            phone:
              type: string

    # not: Schema must NOT match
    NonEmptyString:
      type: string
      not:
        enum: ['', null]
```

## Security Schemes

### Bearer Authentication (JWT)

```yaml
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: |
        JWT token authentication.
        Format: `Bearer <token>`
        Obtain token from `/auth/login` endpoint.

# Apply globally
security:
  - bearerAuth: []

# Or per-operation
paths:
  /users:
    get:
      security:
        - bearerAuth: []
```

### API Key Authentication

```yaml
components:
  securitySchemes:
    apiKeyHeader:
      type: apiKey
      in: header
      name: X-API-Key
      description: API key for authentication

    apiKeyQuery:
      type: apiKey
      in: query
      name: api_key

    apiKeyCookie:
      type: apiKey
      in: cookie
      name: SESSION_TOKEN
```

### OAuth2 Flows

```yaml
components:
  securitySchemes:
    oauth2:
      type: oauth2
      description: OAuth2 authentication with multiple flows
      flows:
        # Authorization Code Flow (most secure for web apps)
        authorizationCode:
          authorizationUrl: https://auth.example.com/oauth/authorize
          tokenUrl: https://auth.example.com/oauth/token
          refreshUrl: https://auth.example.com/oauth/refresh
          scopes:
            read:users: Read user information
            write:users: Modify user information
            admin: Administrative access

        # Implicit Flow (for SPAs, less secure)
        implicit:
          authorizationUrl: https://auth.example.com/oauth/authorize
          scopes:
            read:users: Read user information

        # Client Credentials Flow (for service-to-service)
        clientCredentials:
          tokenUrl: https://auth.example.com/oauth/token
          scopes:
            service:read: Service read access

        # Password Flow (for trusted clients)
        password:
          tokenUrl: https://auth.example.com/oauth/token
          scopes:
            read:users: Read user information
```

### OpenID Connect

```yaml
components:
  securitySchemes:
    openIdConnect:
      type: openIdConnect
      openIdConnectUrl: https://auth.example.com/.well-known/openid-configuration
      description: OpenID Connect authentication
```

## Validation Rules

### Required vs Optional Fields

```yaml
components:
  schemas:
    User:
      type: object
      required:
        - id
        - email
      properties:
        id:
          type: string
        email:
          type: string
        name:
          type: string        # Optional (not in required array)
        bio:
          type: string
          nullable: true      # Explicitly allows null
```

### String Validation

```yaml
components:
  schemas:
    ValidationExamples:
      type: object
      properties:
        username:
          type: string
          minLength: 3
          maxLength: 20
          pattern: '^[a-zA-Z0-9_]+$'
          example: "john_doe"

        email:
          type: string
          format: email
          pattern: '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

        hexColor:
          type: string
          pattern: '^#[0-9A-Fa-f]{6}$'
          example: "#FF5733"
```

### Number Validation

```yaml
components:
  schemas:
    NumberValidation:
      type: object
      properties:
        percentage:
          type: number
          minimum: 0
          maximum: 100
          exclusiveMinimum: false
          exclusiveMaximum: false

        price:
          type: number
          minimum: 0
          multipleOf: 0.01      # Two decimal places
          example: 19.99

        quantity:
          type: integer
          minimum: 1
          maximum: 1000
```

### Array Validation

```yaml
components:
  schemas:
    ArrayValidation:
      type: object
      properties:
        tags:
          type: array
          items:
            type: string
            minLength: 2
            maxLength: 30
          minItems: 1
          maxItems: 10
          uniqueItems: true
          example: ["javascript", "tutorial"]
```

## Best Practices

### 1. Use Descriptive Names

```yaml
# ❌ Bad: Unclear names
paths:
  /api/v1/u:
    get:
      summary: Get u
      operationId: getU

# ✅ Good: Clear, descriptive names
paths:
  /api/v1/users:
    get:
      summary: List all users
      operationId: listUsers
      description: Retrieve a paginated list of users with optional filtering
```

### 2. Provide Examples

```yaml
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          format: uuid
          example: "123e4567-e89b-12d3-a456-426614174000"
        email:
          type: string
          format: email
          example: "user@example.com"
      # Multiple examples
      example:
        id: "123e4567-e89b-12d3-a456-426614174000"
        email: "user@example.com"
        name: "John Doe"
```

### 3. Document Error Responses

```yaml
paths:
  /users/{id}:
    get:
      responses:
        '200':
          description: User found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '400':
          description: Invalid user ID format
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
              example:
                error: "Invalid UUID format"
                code: "INVALID_ID"
        '401':
          description: Unauthorized
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '404':
          description: User not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
              example:
                error: "User not found"
                code: "USER_NOT_FOUND"
        '500':
          description: Internal server error
```

### 4. Use Components for Reusability

```yaml
components:
  # Reusable schemas
  schemas:
    Error:
      type: object
      properties:
        error:
          type: string
        code:
          type: string

  # Reusable parameters
  parameters:
    PageParam:
      name: page
      in: query
      schema:
        type: integer
        default: 1

    LimitParam:
      name: limit
      in: query
      schema:
        type: integer
        default: 10
        maximum: 100

  # Reusable responses
  responses:
    UnauthorizedError:
      description: Unauthorized
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'

    NotFoundError:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Error'
```

### 5. Version Your API

```yaml
# In server URL
servers:
  - url: https://api.example.com/v1
    description: Version 1 (Stable)
  - url: https://api.example.com/v2
    description: Version 2 (Beta)

# In info
info:
  version: 2.0.0-beta
  x-api-version: v2
```

## Common Patterns

### Pagination

```yaml
components:
  parameters:
    PageParam:
      name: page
      in: query
      description: Page number (1-indexed)
      schema:
        type: integer
        minimum: 1
        default: 1

    LimitParam:
      name: limit
      in: query
      description: Number of items per page
      schema:
        type: integer
        minimum: 1
        maximum: 100
        default: 10

  schemas:
    PaginatedResponse:
      type: object
      properties:
        data:
          type: array
          items:
            type: object
        pagination:
          type: object
          properties:
            page:
              type: integer
            limit:
              type: integer
            total:
              type: integer
            totalPages:
              type: integer
```

### Filtering

```yaml
paths:
  /users:
    get:
      parameters:
        - name: role
          in: query
          description: Filter by user role
          schema:
            type: string
            enum: [user, admin, moderator]

        - name: search
          in: query
          description: Search by name or email
          schema:
            type: string

        - name: createdAfter
          in: query
          description: Filter users created after date
          schema:
            type: string
            format: date-time
```

### Sorting

```yaml
paths:
  /users:
    get:
      parameters:
        - name: sort
          in: query
          description: Sort field
          schema:
            type: string
            enum: [createdAt, name, email]
            default: createdAt

        - name: order
          in: query
          description: Sort order
          schema:
            type: string
            enum: [asc, desc]
            default: desc
```

## Tooling Ecosystem

### Validation
- **Swagger Editor**: https://editor.swagger.io (online validator)
- **Swagger CLI**: `swagger-cli validate spec.yaml`
- **OpenAPI Validator**: Language-specific validators

### Documentation
- **Swagger UI**: Interactive API documentation
- **Redoc**: Alternative documentation renderer
- **Stoplight**: API design and documentation platform

### Code Generation
- **OpenAPI Generator**: Generate client SDKs and server stubs
- **Swagger Codegen**: Generate code from OpenAPI specs

### Testing
- **Postman**: Import OpenAPI specs for testing
- **Prism**: Mock server from OpenAPI specs
- **Dredd**: API contract testing

## References

- **OpenAPI Specification**: https://spec.openapis.org/oas/v3.0.3
- **JSON Schema**: https://json-schema.org
- **Swagger Tools**: https://swagger.io/tools/
- **OpenAPI Initiative**: https://www.openapis.org

---

**Last Updated**: 2025-01-15
**Spec Version**: OpenAPI 3.0.3
**Maintainer**: Claude Code (Sonnet 4.5)
