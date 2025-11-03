# {{PROJECT_NAME}}

{{PROJECT_DESCRIPTION}}

## Features

- Modern {{FRAMEWORK}} version
- Security best practices
- Minimal dependencies
- Production-ready configuration
- Clean code structure

## Prerequisites

{{#if NODE}}
- Node.js >= {{NODE_VERSION}}
- npm >= 8.0.0 (or yarn/pnpm)
{{/if}}
{{#if PYTHON}}
- Python >= {{PYTHON_VERSION}}
- Poetry (recommended) or pip
{{/if}}
{{#if GO}}
- Go >= {{GO_VERSION}}
{{/if}}

## Quick Start

### Installation

{{#if NODE}}
```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env

# Start development server
npm run dev
```
{{/if}}

{{#if PYTHON}}
```bash
# Install dependencies with Poetry
poetry install

# Or with pip
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env

# Start development server
poetry run uvicorn app.main:app --reload
# Or with pip
uvicorn app.main:app --reload
```
{{/if}}

{{#if GO}}
```bash
# Install dependencies
go mod download

# Copy environment variables
cp .env.example .env

# Run the application
go run cmd/{{PROJECT_NAME}}/main.go

# Or build and run
go build -o bin/{{PROJECT_NAME}} cmd/{{PROJECT_NAME}}/main.go
./bin/{{PROJECT_NAME}}
```
{{/if}}

### Development

{{#if NODE}}
```bash
# Run in development mode with auto-reload
npm run dev

# Run tests
npm test

# Run linter
npm run lint

# Format code
npm run format

# Build for production
npm run build
```
{{/if}}

{{#if PYTHON}}
```bash
# Run with auto-reload
poetry run uvicorn app.main:app --reload

# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=app

# Run linter
poetry run ruff check .

# Format code
poetry run black .

# Type checking
poetry run mypy .
```
{{/if}}

{{#if GO}}
```bash
# Run with auto-reload (requires air)
air

# Run tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run linter
golangci-lint run

# Format code
go fmt ./...

# Build
go build -o bin/{{PROJECT_NAME}} cmd/{{PROJECT_NAME}}/main.go
```
{{/if}}

## Project Structure

```
{{PROJECT_NAME}}/
{{#if NODE}}
├── src/                  # Source code
│   ├── routes/          # API routes
│   ├── controllers/     # Business logic
│   ├── middleware/      # Express middleware
│   ├── models/          # Data models
│   └── utils/           # Utility functions
├── tests/               # Test files
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── config/              # Configuration files
├── .eslintrc.json      # ESLint configuration
├── .prettierrc         # Prettier configuration
├── package.json        # Dependencies and scripts
└── .env.example        # Environment variables template
{{/if}}
{{#if PYTHON}}
├── app/                 # Application code
│   ├── api/            # API routes
│   ├── core/           # Core functionality
│   ├── models/         # Pydantic models
│   ├── db/             # Database models
│   └── services/       # Business logic
├── tests/               # Test files
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── config/              # Configuration files
├── pyproject.toml      # Project metadata and dependencies
└── .env.example        # Environment variables template
{{/if}}
{{#if GO}}
├── cmd/                 # Main applications
│   └── {{PROJECT_NAME}}/
│       └── main.go     # Application entry point
├── internal/            # Private application code
│   ├── handler/        # HTTP handlers
│   ├── service/        # Business logic
│   └── model/          # Data models
├── pkg/                 # Public libraries
├── test/                # Additional test files
├── go.mod              # Go module definition
└── .env.example        # Environment variables template
{{/if}}
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
{{#if NODE}}
PORT=3000
NODE_ENV=development
{{/if}}
{{#if PYTHON}}
PORT=8000
ENVIRONMENT=development
{{/if}}
{{#if GO}}
PORT=8080
ENVIRONMENT=development
{{/if}}
```

{{#if DOCKER}}
## Docker Support

### Build and run with Docker

```bash
# Build Docker image
docker build -t {{PROJECT_NAME}} .

# Run container
docker run -p {{PORT}}:{{PORT}} --env-file .env {{PROJECT_NAME}}
```

### Using Docker Compose

```bash
# Start services
docker-compose up

# Start in detached mode
docker-compose up -d

# Stop services
docker-compose down
```
{{/if}}

{{#if CI_CD}}
## CI/CD

This project includes GitHub Actions workflows for:

- Automated testing on pull requests
- Linting and code quality checks
- Security vulnerability scanning
- Automated deployments (configure as needed)

Workflow files are located in `.github/workflows/`.
{{/if}}

## Testing

{{#if NODE}}
```bash
# Run all tests
npm test

# Run tests in watch mode
npm test -- --watch

# Run tests with coverage
npm test -- --coverage
```
{{/if}}

{{#if PYTHON}}
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/unit/test_main.py
```
{{/if}}

{{#if GO}}
```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run with detailed coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```
{{/if}}

## Code Quality

### Linting

{{#if NODE}}
```bash
npm run lint
npm run lint:fix  # Auto-fix issues
```
{{/if}}

{{#if PYTHON}}
```bash
poetry run ruff check .
poetry run ruff check --fix .  # Auto-fix issues
```
{{/if}}

{{#if GO}}
```bash
golangci-lint run
golangci-lint run --fix  # Auto-fix issues
```
{{/if}}

### Formatting

{{#if NODE}}
```bash
npm run format
npm run format:check  # Check without modifying
```
{{/if}}

{{#if PYTHON}}
```bash
poetry run black .
poetry run black --check .  # Check without modifying
```
{{/if}}

{{#if GO}}
```bash
go fmt ./...
gofmt -l .  # List files that need formatting
```
{{/if}}

## Deployment

{{#if DOCKER}}
### Docker Deployment

```bash
# Build production image
docker build -t {{PROJECT_NAME}}:latest .

# Tag for registry
docker tag {{PROJECT_NAME}}:latest registry.example.com/{{PROJECT_NAME}}:latest

# Push to registry
docker push registry.example.com/{{PROJECT_NAME}}:latest
```
{{/if}}

### Production Checklist

- [ ] Update environment variables for production
- [ ] Configure CORS and security headers
- [ ] Set up SSL/TLS certificates
- [ ] Configure logging and monitoring
- [ ] Set up error tracking (e.g., Sentry)
- [ ] Configure database backups
- [ ] Set up CI/CD pipeline
- [ ] Load testing
- [ ] Security audit

## Best Practices

- **Security**: Never commit `.env` files or secrets
- **Testing**: Maintain 80%+ code coverage
- **Code Quality**: Follow linting rules and formatters
- **Dependencies**: Keep dependencies up to date
- **Documentation**: Update README when adding features
- **Git**: Write clear commit messages

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

{{LICENSE}}

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review closed issues for similar problems

---

Generated with [base-template-generator](https://github.com/ruvnet/claude-flow)
