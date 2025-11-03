---
name: api-docs
description: Generate and maintain comprehensive API documentation using OpenAPI 3.0, Swagger UI, and GraphQL Playground. Use when documenting REST APIs, GraphQL services, or creating API reference materials. Ensures consistent, machine-readable, and developer-friendly documentation.
tier: gold
capabilities:
  - OpenAPI 3.0 specification generation from source code (Flask, FastAPI, Express.js, Django)
  - Multi-format validation (swagger-cli, openapi-generator-cli, yq, yamllint)
  - Interactive documentation (Swagger UI, ReDoc)
  - Markdown and HTML documentation generation
  - Automated schema-to-example conversion
  - Code examples in multiple languages (cURL, JavaScript, Python, Go)
resources:
  scripts:
    - generate_openapi.py (350+ lines, framework auto-detection, route extraction)
    - validate_spec.sh (400+ lines, multi-tool validation pipeline)
    - create_docs.py (400+ lines, multiple output formats)
  templates:
    - openapi-3.0.yaml (complete OpenAPI 3.0 specification template)
    - endpoint-template.json (reusable endpoint definition with examples)
    - schema-template.json (comprehensive schema patterns)
  tests:
    - test-1-openapi-generation.md (12 test cases, framework validation)
    - test-2-spec-validation.md (12 test cases, compliance checks)
    - test-3-documentation.md (12 test cases, format validation)
---

# API Documentation Generator (Gold Tier)

Generate production-quality API documentation with OpenAPI 3.0, automated validation, and interactive documentation formats.

## When to Use This Skill

Use when documenting new or existing APIs, creating REST API specifications, maintaining GraphQL schema documentation, or generating interactive API explorers with Swagger UI/GraphQL Playground. Ideal for API-first development, maintaining accurate documentation, and ensuring OpenAPI 3.0 compliance.

## API Documentation Types

### REST API Documentation
- OpenAPI 3.0 specifications
- Endpoint descriptions and parameters
- Request/response examples
- Authentication schemes
- Error codes and handling

### GraphQL Documentation
- Schema definitions
- Query and mutation documentation
- Type system reference
- Resolver documentation

## Process

1. **Analyze API structure**
   - Identify all endpoints/operations
   - Document request/response schemas
   - Define authentication requirements
   - Catalog error responses

2. **Generate OpenAPI/GraphQL spec**
   - Write structured YAML/SDL definitions
   - Include comprehensive examples
   - Document edge cases
   - Add usage guidelines

3. **Create interactive documentation**
   - Configure Swagger UI for REST APIs
   - Set up GraphQL Playground
   - Add authentication testing support
   - Enable "Try It Out" functionality

4. **Add versioning and deprecation notices**
   - Document API version compatibility
   - Mark deprecated endpoints
   - Provide migration guides
   - Include changelog

5. **Generate code examples**
   - Client SDK examples (curl, JavaScript, Python)
   - Authentication flows
   - Common use case scenarios
   - Error handling patterns

## Gold Tier Features

### Automated Specification Generation
- **Framework Support**: Flask, FastAPI, Express.js, Django REST Framework
- **Auto-Detection**: Automatically identifies web framework from source code
- **Route Extraction**: Parses decorators, docstrings, and JSDoc comments
- **Parameter Inference**: Extracts path/query/body parameters from function signatures
- **Schema Generation**: Creates reusable component schemas from type hints

### Multi-Tool Validation
- **Syntax Validation**: YAML/JSON syntax checking with yamllint/jq
- **OpenAPI Compliance**: swagger-cli and openapi-generator-cli validation
- **Strict Mode**: Treat warnings as errors for production specs
- **Validation Reports**: Detailed error/warning reports with timestamps
- **Field Checks**: Required fields, best practices, security schemes

### Interactive Documentation
- **Swagger UI**: Interactive API explorer with "Try it out" functionality
- **ReDoc**: Clean three-panel documentation with search
- **Markdown**: Human-readable documentation with tables
- **HTML**: Custom templates with styling

### Code Examples
- **Multi-Language**: cURL, JavaScript, Python, Go examples
- **Authentication**: Automatic header injection (Bearer, API Key)
- **Request/Response**: Complete examples with realistic data

## Available Resources

### Scripts (resources/scripts/)
1. **generate_openapi.py**
   - Generate OpenAPI 3.0 specs from source code
   - Framework auto-detection (Flask, FastAPI, Express, Django)
   - Route extraction with docstrings/JSDoc
   - Component schema generation
   - YAML/JSON output formats

2. **validate_spec.sh**
   - Multi-tool validation pipeline
   - YAML/JSON syntax checking
   - OpenAPI 3.0 compliance validation
   - Best practices enforcement
   - Validation report generation

3. **create_docs.py**
   - Markdown documentation generation
   - Swagger UI setup
   - ReDoc setup
   - Custom HTML templates
   - Schema-to-example conversion

### Templates (resources/templates/)
1. **openapi-3.0.yaml** - Complete OpenAPI 3.0 specification template
2. **endpoint-template.json** - Reusable endpoint definition with examples
3. **schema-template.json** - Comprehensive schema patterns

### Tests (tests/)
1. **test-1-openapi-generation.md** - 12 test cases for spec generation
2. **test-2-spec-validation.md** - 12 test cases for validation
3. **test-3-documentation.md** - 12 test cases for documentation

## Usage Examples

### Generate OpenAPI Spec from Source Code
```bash
# Auto-detect framework and generate spec
python resources/scripts/generate_openapi.py \
  --source ./src \
  --output openapi.yaml \
  --framework auto

# Explicit framework with custom info
python resources/scripts/generate_openapi.py \
  --source ./app \
  --output api-spec.json \
  --framework fastapi \
  --format json \
  --title "My API" \
  --version "2.0.0"
```

### Validate OpenAPI Specification
```bash
# Standard validation
./resources/scripts/validate_spec.sh openapi.yaml

# Strict mode (warnings as errors)
./resources/scripts/validate_spec.sh openapi.yaml --strict

# Save validation report
./resources/scripts/validate_spec.sh openapi.yaml \
  --output validation-report.txt \
  --verbose
```

### Generate Documentation
```bash
# Swagger UI (interactive)
python resources/scripts/create_docs.py \
  --spec openapi.yaml \
  --output docs/ \
  --format html \
  --swagger-ui

# ReDoc (clean layout)
python resources/scripts/create_docs.py \
  --spec openapi.yaml \
  --output docs/ \
  --format html \
  --redoc

# Markdown documentation
python resources/scripts/create_docs.py \
  --spec openapi.yaml \
  --output docs/ \
  --format markdown

# Both formats
python resources/scripts/create_docs.py \
  --spec openapi.yaml \
  --output docs/ \
  --format both \
  --swagger-ui
```

## Best Practices

- **Completeness**: Document all endpoints, parameters, responses
- **Accuracy**: Keep docs synchronized with implementation
- **Clarity**: Use clear descriptions and examples
- **Consistency**: Follow OpenAPI 3.0 or GraphQL standards
- **Usability**: Enable interactive testing
- **Automation**: Integrate generation/validation into CI/CD
- **Versioning**: Track API versions and deprecations
- **Testing**: Use validation before deployment

## Integration with CI/CD

```yaml
# .github/workflows/api-docs.yml
name: API Documentation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Generate OpenAPI Spec
        run: |
          python skills/api-docs/resources/scripts/generate_openapi.py \
            --source ./src --output openapi.yaml

      - name: Validate Specification
        run: |
          bash skills/api-docs/resources/scripts/validate_spec.sh \
            openapi.yaml --strict

      - name: Generate Documentation
        run: |
          python skills/api-docs/resources/scripts/create_docs.py \
            --spec openapi.yaml --output docs/ --format both --swagger-ui

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

## Performance Metrics

| Operation | Small API | Medium API | Large API |
|-----------|-----------|------------|-----------|
| Generation | < 1s | < 3s | < 10s |
| Validation | < 1s | < 2s | < 5s |
| Documentation | < 2s | < 5s | < 15s |

**Specifications:**
- Small: 5-20 endpoints
- Medium: 20-100 endpoints
- Large: 100+ endpoints
