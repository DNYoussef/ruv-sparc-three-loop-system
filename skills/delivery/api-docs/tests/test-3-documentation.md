# Test 3: API Documentation Generation

## Test Overview
Validates the `create_docs.py` script's ability to generate comprehensive API documentation from OpenAPI specifications in multiple formats (HTML, Markdown, Swagger UI, ReDoc).

## Test Objectives
- Generate Markdown documentation from OpenAPI specs
- Create Swagger UI interactive documentation
- Generate ReDoc documentation
- Validate custom HTML templates
- Test code examples generation
- Verify documentation completeness and accuracy

## Test Environment
- Python 3.8+
- Web browser for HTML testing
- Sample OpenAPI specifications
- Required dependencies: PyYAML, Jinja2 (optional)

## Test Cases

### TC3.1: Basic Markdown Generation
**Purpose**: Generate basic Markdown documentation from OpenAPI spec

**Test Data**: `sample-api.yaml` (complete OpenAPI spec)

**Execution**:
```bash
python create_docs.py --spec sample-api.yaml --output docs/ --format markdown
```

**Expected Results**:
- ✓ File created: `docs/API.md`
- ✓ Contains Table of Contents
- ✓ All endpoints documented
- ✓ Schemas section present
- ✓ Authentication section present
- ✓ Error codes documented

**Validation**:
```bash
# Check file exists
ls docs/API.md

# Check structure
grep "## Table of Contents" docs/API.md
grep "## Authentication" docs/API.md
grep "## Endpoints" docs/API.md
grep "## Schemas" docs/API.md

# Count endpoints
grep -c "#### \`" docs/API.md
# Should match number of operations in spec
```

**Sample Output Structure**:
```markdown
# My API

**Version:** 1.0.0

## Table of Contents
- [Overview](#overview)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [Users](#users)
  - [Posts](#posts)
- [Schemas](#schemas)
- [Error Codes](#error-codes)

## Overview
...

## Endpoints

### Users

#### `GET /api/users`
Retrieve all users.

**Parameters:**
| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| page | query | integer | No | Page number |
| limit | query | integer | No | Items per page |

**Responses:**
- **200**: Success
- **401**: Unauthorized

---
```

---

### TC3.2: Swagger UI Documentation
**Purpose**: Generate interactive Swagger UI documentation

**Execution**:
```bash
python create_docs.py --spec sample-api.yaml --output docs/ --format html --swagger-ui
```

**Expected Results**:
- ✓ Files created:
  - `docs/index.html`
  - `docs/openapi.json`
- ✓ HTML includes Swagger UI CDN links
- ✓ Spec embedded or referenced correctly
- ✓ Interactive API explorer functional

**Validation**:
```bash
# Check files exist
ls docs/index.html docs/openapi.json

# Verify Swagger UI references
grep "swagger-ui-dist" docs/index.html
grep "SwaggerUIBundle" docs/index.html

# Check spec URL
grep "openapi.json" docs/index.html
```

**Manual Testing**:
1. Open `docs/index.html` in browser
2. ✓ Swagger UI loads without errors
3. ✓ All endpoints visible
4. ✓ "Try it out" functionality works
5. ✓ Schemas expandable
6. ✓ Authentication can be configured

**Expected Browser Output**:
- Clean Swagger UI interface
- Expandable endpoint sections
- Request/response examples
- Model schemas with examples
- Working "Execute" buttons

---

### TC3.3: ReDoc Documentation
**Purpose**: Generate clean ReDoc documentation

**Execution**:
```bash
python create_docs.py --spec sample-api.yaml --output docs/ --format html --redoc
```

**Expected Results**:
- ✓ Files created:
  - `docs/index.html`
  - `docs/openapi.json`
- ✓ HTML includes ReDoc CDN
- ✓ Three-panel layout renders correctly
- ✓ Navigation sidebar functional

**Validation**:
```bash
# Check ReDoc script
grep "redoc.standalone.js" docs/index.html
grep "<redoc" docs/index.html

# Verify spec reference
grep "spec-url" docs/index.html
```

**Manual Testing**:
1. Open in browser
2. ✓ Three-panel layout (nav, content, code samples)
3. ✓ Smooth scrolling
4. ✓ Search functionality works
5. ✓ Code samples in multiple languages
6. ✓ Responsive design

---

### TC3.4: Both Markdown and HTML
**Purpose**: Generate both formats simultaneously

**Execution**:
```bash
python create_docs.py --spec sample-api.yaml --output docs/ --format both --swagger-ui
```

**Expected Results**:
- ✓ `docs/API.md` created (Markdown)
- ✓ `docs/index.html` created (HTML with Swagger UI)
- ✓ `docs/openapi.json` created
- ✓ Both formats contain same information

**Validation**:
```bash
ls docs/API.md docs/index.html docs/openapi.json
```

---

### TC3.5: Custom HTML Template
**Purpose**: Use custom HTML template for documentation

**Test Data**: `custom-template.html`
```html
<!DOCTYPE html>
<html>
<head>
    <title>{{ api_title }}</title>
    <style>
        /* Custom styling */
    </style>
</head>
<body>
    <header>
        <h1>{{ api_title }}</h1>
        <p>Version: {{ api_version }}</p>
    </header>
    <main>
        {{ content }}
    </main>
</body>
</html>
```

**Execution**:
```bash
python create_docs.py --spec sample-api.yaml --output docs/ --format html --template custom-template.html
```

**Expected Results**:
- ✓ Custom template used
- ✓ Variables replaced correctly
- ✓ Custom styling applied

---

### TC3.6: Large Specification
**Purpose**: Test performance with large API specs (100+ endpoints)

**Test Data**: `large-api.yaml` (200+ operations)

**Execution**:
```bash
time python create_docs.py --spec large-api.yaml --output docs/ --format both --swagger-ui
```

**Expected Results**:
- ✓ Generation completes successfully
- ✓ Markdown file < 500 KB
- ✓ HTML loads in < 3 seconds
- ✓ All endpoints present
- ✓ No memory errors

**Performance Metrics**:
- Generation time: < 10 seconds
- Markdown size: 200-400 KB
- HTML size: 50-100 KB (without CDN resources)

---

### TC3.7: Schema Examples Generation
**Purpose**: Verify schema-to-example conversion

**Test Data**: Spec with complex schemas
```yaml
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: integer
          example: 123
        name:
          type: string
          example: John Doe
        roles:
          type: array
          items:
            type: string
          example: ["admin", "user"]
        metadata:
          type: object
          properties:
            created:
              type: string
              format: date-time
```

**Expected Markdown Output**:
```markdown
### User

\`\`\`json
{
  "id": 123,
  "name": "John Doe",
  "roles": ["admin", "user"],
  "metadata": {
    "created": "2024-01-15T10:30:00Z"
  }
}
\`\`\`
```

**Validation**:
- ✓ Nested objects expanded
- ✓ Arrays shown with example item
- ✓ Format types respected (date-time, uuid, etc.)
- ✓ Example values from spec used
- ✓ Proper JSON formatting

---

### TC3.8: Multi-Server Documentation
**Purpose**: Document APIs with multiple server environments

**Test Data**: Spec with 3 servers (dev, staging, prod)

**Expected Markdown Output**:
```markdown
### Base URLs

- **Development server**: `http://localhost:3000`
- **Staging environment**: `https://staging-api.example.com`
- **Production environment**: `https://api.example.com`
```

**Swagger UI Behavior**:
- ✓ Server dropdown shows all servers
- ✓ Can switch between servers
- ✓ Requests sent to selected server

---

### TC3.9: Security Documentation
**Purpose**: Document multiple authentication methods

**Test Data**: Spec with Bearer token, API key, OAuth2

**Expected Markdown Output**:
```markdown
## Authentication

### bearerAuth

**Type:** http

**Scheme:** bearer

### apiKey

**Type:** apiKey

**In:** header

**Name:** X-API-Key

### oauth2

**Type:** oauth2

**Flows:** authorizationCode
```

**Swagger UI Behavior**:
- ✓ "Authorize" button visible
- ✓ Can configure each auth method
- ✓ Token persists across requests
- ✓ Headers/params added automatically

---

### TC3.10: Error Handling
**Purpose**: Test error conditions

**Test Cases**:

1. **Missing spec file**
   ```bash
   python create_docs.py --spec nonexistent.yaml --output docs/
   # Expected: Error message, exit code 1
   ```

2. **Invalid spec file**
   ```bash
   python create_docs.py --spec invalid.yaml --output docs/
   # Expected: YAML parse error, exit code 1
   ```

3. **Write permission error**
   ```bash
   python create_docs.py --spec spec.yaml --output /restricted/ --format html
   # Expected: Permission error, exit code 1
   ```

---

### TC3.11: Code Examples in Markdown
**Purpose**: Verify code examples generation for different languages

**Expected Output**:
```markdown
#### `GET /api/users/{id}`

**cURL:**
\`\`\`bash
curl -X GET 'https://api.example.com/api/users/123' \
  -H 'Authorization: Bearer YOUR_TOKEN'
\`\`\`

**JavaScript:**
\`\`\`javascript
const response = await fetch('https://api.example.com/api/users/123', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
\`\`\`

**Python:**
\`\`\`python
import requests

response = requests.get(
    'https://api.example.com/api/users/123',
    headers={'Authorization': f'Bearer {token}'}
)
\`\`\`
```

---

### TC3.12: Parameter Tables
**Purpose**: Test parameter documentation formatting

**Expected Markdown Table**:
```markdown
**Parameters:**

| Name | In | Type | Required | Description |
|------|-------|------|----------|-------------|
| id | path | string | Yes | User identifier (UUID) |
| include | query | array | No | Related resources to include |
| fields | query | string | No | Sparse fieldsets |
```

**Validation**:
- ✓ Table header present
- ✓ All parameters listed
- ✓ Required/optional clearly marked
- ✓ Type information correct
- ✓ Descriptions included

---

## Documentation Quality Checklist

### Completeness
- [ ] All endpoints documented
- [ ] All schemas documented
- [ ] All parameters documented
- [ ] All responses documented
- [ ] Authentication documented
- [ ] Error codes documented

### Accuracy
- [ ] Endpoint paths match spec
- [ ] HTTP methods correct
- [ ] Parameter types correct
- [ ] Response schemas accurate
- [ ] Examples valid

### Usability
- [ ] Table of contents navigable
- [ ] Code examples provided
- [ ] Search functionality (Swagger UI/ReDoc)
- [ ] Responsive design (HTML)
- [ ] Clear error messages

### Formatting
- [ ] Consistent markdown syntax
- [ ] Proper JSON formatting
- [ ] Code blocks syntax-highlighted
- [ ] Tables properly formatted
- [ ] Links working

## File Output Matrix

| Format | Files Created | Interactive | Size (typical) |
|--------|--------------|-------------|----------------|
| Markdown | API.md | No | 50-200 KB |
| HTML (Custom) | index.html | No | 30-100 KB |
| Swagger UI | index.html, openapi.json | Yes | 10-50 KB + CDN |
| ReDoc | index.html, openapi.json | Yes | 5-30 KB + CDN |
| Both | API.md, index.html, openapi.json | Yes | Combined |

## Browser Compatibility

### Swagger UI
- ✓ Chrome 90+
- ✓ Firefox 88+
- ✓ Safari 14+
- ✓ Edge 90+

### ReDoc
- ✓ Chrome 90+
- ✓ Firefox 88+
- ✓ Safari 14+
- ✓ Edge 90+

## Success Criteria
- ✓ All 12 test cases pass
- ✓ Generated docs are accurate and complete
- ✓ Interactive docs function correctly
- ✓ Performance within acceptable limits
- ✓ No broken links or references
- ✓ Responsive design works on mobile

## Deployment Options

### Static Hosting
```bash
# Copy docs to web server
cp -r docs/ /var/www/html/api-docs/

# Or use Python simple HTTP server
cd docs/
python -m http.server 8080
# Open http://localhost:8080
```

### GitHub Pages
```bash
# Push docs to gh-pages branch
git checkout -b gh-pages
git add docs/
git commit -m "Add API documentation"
git push origin gh-pages
# Enable GitHub Pages in repo settings
```

### Docker
```dockerfile
FROM nginx:alpine
COPY docs/ /usr/share/nginx/html/
EXPOSE 80
```

## Next Steps
After successful documentation generation:
1. Review generated docs for completeness
2. Customize templates and styling
3. Add code examples in additional languages
4. Set up CI/CD for automatic regeneration
5. Deploy to hosting platform
6. Share documentation URL with team
