# API Documentation Generator - Comprehensive API Documentation from Code

**Version**: 2.0.0 (Silver Tier)
**Purpose**: Generate comprehensive, production-ready API documentation from code with support for OpenAPI/Swagger, GraphQL, and versioning strategies

## 🎯 What This Is

API Documentation Generator automatically creates complete API documentation from your codebase, including:
- OpenAPI 3.0/Swagger specifications
- GraphQL schema documentation
- Interactive API explorers (Swagger UI, Redoc)
- API versioning strategies
- Request/response examples
- Authentication flows
- Error handling guides

## 🚀 Quick Start

### 1. Basic REST API Documentation
```bash
# Generate OpenAPI spec from Express.js code
npx claude-flow@alpha hooks pre-task --description "Generate REST API docs"
# API docs will be auto-generated with:
# - Endpoint discovery from routes
# - Schema extraction from controllers
# - Authentication documentation
```

### 2. GraphQL Documentation
```bash
# Generate GraphQL schema documentation
npx claude-flow@alpha hooks pre-task --description "Document GraphQL API"
# Creates:
# - Type definitions
# - Query/Mutation documentation
# - Resolver descriptions
```

### 3. Interactive Documentation
```bash
# Deploy Swagger UI for testing
npx claude-flow@alpha hooks pre-task --description "Deploy interactive API docs"
# Sets up:
# - Swagger UI at /api-docs
# - Redoc at /api-docs/redoc
# - Try-it-out functionality
```

## 📋 When to Use This Skill

Use **api-docs** when you need to:
- Generate OpenAPI specs from existing code
- Create GraphQL schema documentation
- Set up interactive API documentation
- Implement API versioning strategies
- Document authentication flows
- Maintain API changelog
- Create client SDKs from specs
- Validate API contracts

**Auto-triggers on keywords**: "API docs", "OpenAPI", "Swagger", "API documentation", "REST API docs", "GraphQL schema"

## 📁 Skill Structure

```
api-docs/
├── README.md                          # This file - overview & quick start
├── skill.md                           # Complete methodology (to be created)
│
├── examples/                          # Real-world usage scenarios
│   ├── example-1-rest-api-docs.md    # REST API with OpenAPI
│   ├── example-2-graphql-docs.md     # GraphQL schema documentation
│   └── example-3-api-versioning.md   # API versioning strategies
│
├── references/                        # Supporting documentation
│   ├── openapi-standards.md          # OpenAPI 3.0 specification guide
│   └── best-practices.md             # API documentation best practices
│
└── graphviz/                          # Process diagrams
    └── workflow.dot                   # Documentation generation pipeline
```

## 💡 Key Features

### OpenAPI/Swagger Support
- OpenAPI 3.0 specification generation
- Swagger UI integration
- Redoc alternative viewer
- Schema validation
- Example generation
- Security definitions

### GraphQL Documentation
- Schema introspection
- Type documentation
- Query/Mutation examples
- Resolver descriptions
- Subscription documentation

### Versioning Strategies
- URL versioning (v1, v2)
- Header versioning
- Query parameter versioning
- Deprecation notices
- Migration guides

### Interactive Features
- Try-it-out functionality
- Request/response examples
- Authentication testing
- Error scenario documentation
- Rate limiting info

## 🎯 Quality Tier: Silver (Production Ready)

**Completion Status**:
- ✅ README.md (overview)
- ✅ 3 examples (REST, GraphQL, versioning)
- ✅ 2 references (OpenAPI standards, best practices)
- ✅ 1 GraphViz diagram (workflow)
- Total: 7+ files

**Next Tier Goals (Gold)**:
- Add resources/scripts/ for automation
- Add resources/templates/ for boilerplate
- Add tests/ for validation
- Expand to 12+ files

## 📚 Examples Overview

### Example 1: REST API Documentation (Basic)
Complete walkthrough of generating OpenAPI specs from Express.js REST API with authentication and error handling.

### Example 2: GraphQL Documentation (Intermediate)
Step-by-step guide to documenting GraphQL APIs with type definitions, queries, mutations, and subscriptions.

### Example 3: API Versioning Strategies (Advanced)
Comprehensive guide to implementing and documenting API versioning across multiple strategies.

## 🔗 Related Skills

- **when-building-backend-api-orchestrate-api-development**: Complete API development workflow
- **sop-api-development**: Standardized API development process
- **documentation**: General documentation generation
- **github-release-management**: API release automation

## 📖 Documentation Standards

### OpenAPI 3.0 Compliance
All generated specs follow OpenAPI 3.0 standards with:
- Valid schema definitions
- Complete request/response examples
- Authentication flows (OAuth2, JWT, API keys)
- Error response documentation
- Rate limiting information

### Best Practices
- Auto-generate from code annotations
- Keep docs in sync with code
- Include realistic examples
- Document all error scenarios
- Provide authentication guides
- Version documentation properly

## 🚀 Integration

### With Claude Flow
```bash
# Pre-task hook
npx claude-flow@alpha hooks pre-task --description "Generate API docs"

# Post-edit hook
npx claude-flow@alpha hooks post-edit --file "api-spec.yaml" --memory-key "api-docs/openapi-spec"

# Post-task hook
npx claude-flow@alpha hooks post-task --task-id "api-documentation"
```

### With Memory MCP
All API documentation metadata is stored in Memory MCP with tags:
- **WHO**: api-docs agent
- **WHEN**: ISO timestamp
- **PROJECT**: [your-project-name]
- **WHY**: documentation

## 🎓 Getting Started

1. **Review Examples**: Start with `examples/example-1-rest-api-docs.md`
2. **Check References**: Read `references/openapi-standards.md` for standards
3. **Follow Workflow**: Use `graphviz/workflow.dot` for process visualization
4. **Generate Docs**: Apply patterns to your codebase
5. **Deploy Interactive**: Set up Swagger UI for testing

## 🔧 Tools & Technologies

### Supported
- **OpenAPI**: 3.0, 3.1
- **GraphQL**: Schema SDL, Introspection
- **Documentation**: Swagger UI, Redoc, GraphQL Playground
- **Languages**: JavaScript/TypeScript, Python, Java, Go, Ruby
- **Frameworks**: Express, FastAPI, Spring Boot, Gin, Rails

### Generated Artifacts
- `openapi.yaml` - OpenAPI specification
- `schema.graphql` - GraphQL schema
- `/api-docs` - Swagger UI endpoint
- `/api-docs/redoc` - Redoc endpoint
- `CHANGELOG.md` - API version history

---

**Maintained by**: Claude Code (Sonnet 4.5)
**License**: Same as ruv-sparc-three-loop-system
**Support**: Create issue in repository
