# Test 1: OpenAPI 3.0 Specification Generation

## Test Overview
Validates the `generate_openapi.py` script's ability to automatically generate OpenAPI 3.0 specifications from source code.

## Test Objectives
- Verify framework auto-detection (Flask, FastAPI, Express.js, Django)
- Validate route extraction from source code
- Ensure proper OpenAPI 3.0 schema generation
- Test multiple output formats (YAML, JSON)
- Verify completeness of generated specifications

## Test Environment
- Python 3.8+
- Sample applications in multiple frameworks
- Required dependencies: PyYAML, pathlib

## Test Cases

### TC1.1: Flask Application Route Extraction
**Purpose**: Extract routes from Flask applications

**Test Data**:
```python
# app.py
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    """Retrieve all users."""
    return jsonify({"users": []})

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get user by ID."""
    return jsonify({"id": user_id})

@app.route('/api/users', methods=['POST'])
def create_user():
    """Create a new user."""
    return jsonify({"created": True}), 201
```

**Execution**:
```bash
python generate_openapi.py --source ./flask-app --output openapi-flask.yaml --framework flask
```

**Expected Results**:
- ✓ Framework detected as "flask"
- ✓ 3 routes extracted (2 paths, 3 operations)
- ✓ Path parameters correctly identified (`{user_id}`)
- ✓ HTTP methods properly assigned
- ✓ Docstrings used for descriptions
- ✓ Valid YAML output generated

**Validation**:
```bash
# Check spec validity
./validate_spec.sh openapi-flask.yaml

# Verify paths exist
yq eval '.paths | keys' openapi-flask.yaml
# Expected: ["/api/users", "/api/users/{user_id}"]
```

---

### TC1.2: FastAPI Application Route Extraction
**Purpose**: Extract routes from FastAPI applications with Pydantic models

**Test Data**:
```python
# main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    id: int
    name: str
    email: str

@app.get("/api/users")
async def list_users():
    """List all users."""
    return {"users": []}

@app.post("/api/users")
async def create_user(user: User):
    """Create a new user."""
    return user

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    """Retrieve user by ID."""
    return {"id": user_id}
```

**Execution**:
```bash
python generate_openapi.py --source ./fastapi-app --output openapi-fastapi.json --framework fastapi --format json
```

**Expected Results**:
- ✓ Framework detected as "fastapi"
- ✓ 3 routes extracted with correct methods
- ✓ Parameters extracted from function signatures
- ✓ Type hints preserved
- ✓ JSON output format
- ✓ Pydantic models reflected in request bodies

**Validation**:
```bash
# Validate JSON syntax
jq empty openapi-fastapi.json

# Check paths
jq '.paths | keys' openapi-fastapi.json
```

---

### TC1.3: Express.js Route Extraction
**Purpose**: Extract routes from Express.js applications

**Test Data**:
```javascript
// routes.js
const express = require('express');
const router = express.Router();

/**
 * List all users
 */
router.get('/api/users', (req, res) => {
  res.json({ users: [] });
});

/**
 * Get user by ID
 */
router.get('/api/users/:id', (req, res) => {
  res.json({ id: req.params.id });
});

/**
 * Create new user
 */
router.post('/api/users', (req, res) => {
  res.status(201).json({ created: true });
});

module.exports = router;
```

**Execution**:
```bash
python generate_openapi.py --source ./express-app --output openapi-express.yaml --framework express
```

**Expected Results**:
- ✓ Framework detected as "express"
- ✓ Routes with `:id` converted to `{id}`
- ✓ JSDoc comments used for descriptions
- ✓ HTTP methods correctly assigned
- ✓ 3 operations generated

---

### TC1.4: Auto-Detection Mode
**Purpose**: Test framework auto-detection without explicit --framework flag

**Test Data**: Multiple framework projects in subdirectories

**Execution**:
```bash
# No framework specified - should auto-detect
python generate_openapi.py --source ./multi-framework --output openapi-auto.yaml
```

**Expected Results**:
- ✓ Correct framework detected from file indicators
- ✓ Framework detection logged to console
- ✓ Appropriate extraction method used
- ✓ Spec generated successfully

---

### TC1.5: Custom API Information
**Purpose**: Test custom info fields in generated spec

**Execution**:
```bash
python generate_openapi.py \
  --source ./app \
  --output openapi-custom.yaml \
  --title "My Custom API" \
  --version "2.1.0" \
  --description "Custom API description"
```

**Expected Results**:
- ✓ `info.title` = "My Custom API"
- ✓ `info.version` = "2.1.0"
- ✓ `info.description` = "Custom API description"

**Validation**:
```bash
yq eval '.info.title' openapi-custom.yaml
# Expected: "My Custom API"

yq eval '.info.version' openapi-custom.yaml
# Expected: "2.1.0"
```

---

### TC1.6: Component Generation
**Purpose**: Verify common components are added

**Execution**:
```bash
python generate_openapi.py --source ./app --output openapi-components.yaml
```

**Expected Results**:
- ✓ `components.securitySchemes` includes bearerAuth, apiKey, oauth2
- ✓ `components.responses` includes BadRequest, Unauthorized, NotFound, InternalError
- ✓ Common error schemas defined
- ✓ Reusable parameters created

**Validation**:
```bash
# Check security schemes
yq eval '.components.securitySchemes | keys' openapi-components.yaml
# Expected: ["apiKey", "bearerAuth", "oauth2"]

# Check common responses
yq eval '.components.responses | keys' openapi-components.yaml
# Expected: ["BadRequest", "InternalError", "NotFound", "Unauthorized"]
```

---

### TC1.7: Tag Generation
**Purpose**: Test automatic tag extraction from paths

**Test Data**: Routes with paths like `/api/users/*`, `/api/posts/*`, `/api/comments/*`

**Expected Results**:
- ✓ Tags extracted from first path segment
- ✓ `tags` array populated with unique tags
- ✓ Each tag has name and description
- ✓ Operations tagged appropriately

**Validation**:
```bash
yq eval '.tags[].name' openapi-tags.yaml
# Expected: ["Comments", "Posts", "Users"]
```

---

### TC1.8: Error Handling
**Purpose**: Test error conditions and edge cases

**Test Cases**:
1. **Non-existent source directory**
   ```bash
   python generate_openapi.py --source ./nonexistent --output out.yaml
   # Expected: Error message, exit code 1
   ```

2. **Empty source directory**
   ```bash
   python generate_openapi.py --source ./empty --output out.yaml
   # Expected: Warning about no routes found, valid but empty spec
   ```

3. **Unknown framework**
   ```bash
   python generate_openapi.py --source ./unknown --framework auto --output out.yaml
   # Expected: Warning message, minimal spec generated
   ```

---

## Performance Metrics

| Test Case | Source Files | Routes | Generation Time | Output Size |
|-----------|--------------|--------|-----------------|-------------|
| TC1.1 (Flask) | 5 | 12 | < 1s | 8 KB |
| TC1.2 (FastAPI) | 8 | 24 | < 2s | 15 KB |
| TC1.3 (Express) | 10 | 30 | < 2s | 12 KB |
| TC1.6 (Large app) | 50+ | 150+ | < 5s | 80 KB |

## Success Criteria
- ✓ All 8 test cases pass
- ✓ Generated specs are valid OpenAPI 3.0.3
- ✓ No errors during generation
- ✓ Performance within acceptable limits
- ✓ Both YAML and JSON formats supported

## Known Limitations
- Complex decorator patterns may not be detected
- Nested routers require proper imports
- Generic function names may need manual refinement
- Custom authentication schemes need manual addition

## Next Steps
After successful generation:
1. Run Test 2: Specification Validation
2. Review generated specs for completeness
3. Add custom schemas and examples
4. Generate documentation (Test 3)
