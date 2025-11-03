# Example 2: README Automation

## Overview
This example demonstrates how the documentation skill automatically generates and maintains comprehensive README.md files with installation instructions, usage examples, API references, contribution guidelines, and badges.

## Scenario
**Context**: You've built a TypeScript CLI tool for database migrations but your README.md is outdated and missing critical sections like installation, examples, and troubleshooting.

**Goal**: Generate a production-ready README with all essential sections, code examples, badges, and automatic updates when code changes.

**Starting Point**:
- `src/cli.ts` - CLI implementation
- `package.json` - Package metadata
- Minimal README with just project name

## Walkthrough

### Step 1: Invoke Documentation Skill

```bash
# User request
"Generate a comprehensive README for our database migration CLI tool"

# Auto-trigger detection
Keywords detected: "README", "generate"
Skill triggered: documentation
Agent spawned: developer-documentation-agent
```

### Step 2: Project Analysis Phase

The skill analyzes your project structure:

```typescript
// Example: package.json
{
  "name": "db-migrate-pro",
  "version": "2.1.0",
  "description": "Professional database migration tool with rollback support",
  "main": "dist/index.js",
  "bin": {
    "db-migrate": "./dist/cli.js"
  },
  "scripts": {
    "build": "tsc",
    "test": "jest",
    "lint": "eslint src/**/*.ts"
  },
  "keywords": ["database", "migration", "sql", "postgres", "mysql"],
  "author": "Your Team",
  "license": "MIT",
  "dependencies": {
    "commander": "^11.0.0",
    "knex": "^2.5.0",
    "chalk": "^5.3.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "jest": "^29.0.0"
  }
}
```

```typescript
// Example: src/cli.ts
import { Command } from 'commander';
import { migrate, rollback, status } from './commands';

const program = new Command();

program
  .name('db-migrate')
  .description('Professional database migration tool')
  .version('2.1.0');

program
  .command('up')
  .description('Run pending migrations')
  .option('-s, --steps <number>', 'Number of migrations to run')
  .option('--dry-run', 'Preview migrations without executing')
  .action(migrate);

program
  .command('down')
  .description('Rollback migrations')
  .option('-s, --steps <number>', 'Number of migrations to rollback', '1')
  .action(rollback);

program
  .command('status')
  .description('Show migration status')
  .action(status);

program.parse();
```

**Skill Actions**:
1. Extracts package metadata (name, version, description)
2. Identifies CLI commands and options
3. Detects dependencies and dev tools
4. Analyzes test coverage configuration
5. Checks for CI/CD workflows
6. Scans for contributing guidelines

### Step 3: README Generation

```markdown
# Generated: README.md
# db-migrate-pro

[![npm version](https://badge.fury.io/js/db-migrate-pro.svg)](https://www.npmjs.com/package/db-migrate-pro)
[![Build Status](https://github.com/yourorg/db-migrate-pro/workflows/CI/badge.svg)](https://github.com/yourorg/db-migrate-pro/actions)
[![Coverage Status](https://coveralls.io/repos/github/yourorg/db-migrate-pro/badge.svg?branch=main)](https://coveralls.io/github/yourorg/db-migrate-pro?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Professional database migration tool with rollback support

A powerful, production-ready CLI tool for managing database migrations with support for PostgreSQL, MySQL, and SQLite. Features include dry-run mode, multi-step migrations, automatic rollback, and comprehensive migration tracking.

## ✨ Features

- 🚀 **Fast & Reliable** - Built with TypeScript for type safety
- 🔄 **Rollback Support** - Safely rollback migrations with single command
- 📊 **Migration Tracking** - Track applied migrations with timestamps
- 🎯 **Dry Run Mode** - Preview migrations before executing
- 🔌 **Multi-Database** - PostgreSQL, MySQL, SQLite support
- 🧪 **Well Tested** - 95%+ test coverage
- 📦 **Zero Config** - Works out of the box with sensible defaults

## 📦 Installation

### NPM
```bash
npm install -g db-migrate-pro
```

### Yarn
```bash
yarn global add db-migrate-pro
```

### PNPM
```bash
pnpm add -g db-migrate-pro
```

### From Source
```bash
git clone https://github.com/yourorg/db-migrate-pro.git
cd db-migrate-pro
npm install
npm run build
npm link
```

## 🚀 Quick Start

### 1. Initialize Your Project

```bash
# Create a migrations directory
mkdir migrations

# Create your first migration
db-migrate create add_users_table
```

### 2. Write Your Migration

```javascript
// migrations/20240115_add_users_table.js
exports.up = async (knex) => {
  await knex.schema.createTable('users', (table) => {
    table.increments('id').primary();
    table.string('email').unique().notNullable();
    table.string('password').notNullable();
    table.timestamps(true, true);
  });
};

exports.down = async (knex) => {
  await knex.schema.dropTable('users');
};
```

### 3. Run Migrations

```bash
# Run all pending migrations
db-migrate up

# Run with dry-run to preview
db-migrate up --dry-run

# Run specific number of migrations
db-migrate up --steps 2
```

### 4. Check Status

```bash
db-migrate status
```

## 📖 Usage

### Commands

#### `db-migrate up`
Run pending migrations.

**Options:**
- `-s, --steps <number>` - Number of migrations to run
- `--dry-run` - Preview migrations without executing

**Examples:**
```bash
# Run all pending migrations
db-migrate up

# Run next 3 migrations
db-migrate up --steps 3

# Preview migrations
db-migrate up --dry-run
```

#### `db-migrate down`
Rollback migrations.

**Options:**
- `-s, --steps <number>` - Number of migrations to rollback (default: 1)

**Examples:**
```bash
# Rollback last migration
db-migrate down

# Rollback last 2 migrations
db-migrate down --steps 2
```

#### `db-migrate status`
Show migration status.

**Output:**
```
Migration Status
─────────────────────────────────────────
✓ 20240115_add_users_table.js (applied)
✓ 20240116_add_posts_table.js (applied)
⧗ 20240117_add_comments_table.js (pending)
```

### Configuration

Create a `knexfile.js` in your project root:

```javascript
module.exports = {
  development: {
    client: 'postgresql',
    connection: {
      host: '127.0.0.1',
      port: 5432,
      user: 'postgres',
      password: 'password',
      database: 'myapp_dev'
    },
    migrations: {
      directory: './migrations'
    }
  },

  production: {
    client: 'postgresql',
    connection: process.env.DATABASE_URL,
    migrations: {
      directory: './migrations'
    },
    pool: {
      min: 2,
      max: 10
    }
  }
};
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | - |
| `NODE_ENV` | Environment (development/production) | `development` |
| `MIGRATION_TABLE` | Table name for tracking migrations | `knex_migrations` |
| `MIGRATION_DIR` | Directory containing migrations | `./migrations` |

## 🏗️ Architecture

```
db-migrate-pro/
├── src/
│   ├── cli.ts              # CLI entry point
│   ├── commands/
│   │   ├── migrate.ts      # Migration execution
│   │   ├── rollback.ts     # Rollback logic
│   │   └── status.ts       # Status reporting
│   ├── config/
│   │   └── loader.ts       # Configuration loader
│   └── utils/
│       ├── logger.ts       # Logging utilities
│       └── validator.ts    # Migration validator
├── migrations/             # Example migrations
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## 🧪 Testing

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Run specific test file
npm test -- migrate.test.ts

# Watch mode
npm test -- --watch
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourorg/db-migrate-pro.git
cd db-migrate-pro

# Install dependencies
npm install

# Run in development mode
npm run dev

# Build
npm run build

# Run tests
npm test
```

### Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## 🐛 Troubleshooting

### Common Issues

**Migration fails with "relation already exists"**
```bash
# Check migration status
db-migrate status

# If migration shows as pending but table exists, manually update migration table
# Or rollback and re-run
db-migrate down
db-migrate up
```

**Connection timeout errors**
```bash
# Increase timeout in knexfile.js
{
  pool: {
    acquireTimeoutMillis: 60000
  }
}
```

**Permission denied errors**
```bash
# Ensure database user has CREATE/DROP permissions
GRANT ALL PRIVILEGES ON DATABASE myapp_dev TO postgres;
```

## 📚 API Reference

### Programmatic Usage

```typescript
import { MigrationRunner } from 'db-migrate-pro';

const runner = new MigrationRunner({
  client: 'postgresql',
  connection: process.env.DATABASE_URL
});

// Run migrations programmatically
await runner.up({ steps: 2 });

// Get status
const status = await runner.status();

// Rollback
await runner.down({ steps: 1 });
```

## 🔒 Security

- Never commit database credentials
- Use environment variables for sensitive data
- Review migrations before production deployment
- Enable SSL for production databases

## 📄 License

MIT © [Your Team](LICENSE)

## 🙏 Acknowledgments

- Built with [Knex.js](https://knexjs.org)
- CLI powered by [Commander.js](https://github.com/tj/commander.js)
- Inspired by [Laravel Migrations](https://laravel.com/docs/migrations)

## 📧 Support

- 📖 [Documentation](https://docs.example.com)
- 💬 [Discord Community](https://discord.gg/example)
- 🐛 [Issue Tracker](https://github.com/yourorg/db-migrate-pro/issues)
- 📧 Email: support@example.com

---

**Made with ❤️ by [Your Team](https://example.com)**
```

### Step 4: Auto-Generated Supplementary Files

```markdown
# Generated: CONTRIBUTING.md
# Contributing to db-migrate-pro

Thank you for considering contributing to db-migrate-pro! This guide will help you get started.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Node version, database)
- **Code samples** if applicable

### Suggesting Features

Feature requests are welcome! Please provide:

- **Use case description**
- **Proposed API/interface**
- **Alternative approaches considered**

### Pull Requests

1. Fork the repo and create your branch from `main`
2. Add tests for new functionality
3. Ensure test suite passes
4. Update documentation
5. Follow code style guidelines
6. Submit pull request

## Development Workflow

```bash
# Setup
npm install
npm run build

# Development
npm run dev        # Watch mode
npm test          # Run tests
npm run lint      # Check code style

# Before committing
npm run test:coverage  # Ensure >90% coverage
npm run lint:fix       # Auto-fix style issues
```

## Coding Guidelines

- Use TypeScript strict mode
- Write meaningful variable names
- Add JSDoc comments for public APIs
- Keep functions under 50 lines
- Maintain test coverage >90%

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
```

## Outcomes

### Files Created/Updated
```
project/
├── README.md                  # ✨ Generated (250 lines)
├── CONTRIBUTING.md            # ✨ Generated (150 lines)
├── CHANGELOG.md               # ✨ Generated (tracking version history)
├── CODE_OF_CONDUCT.md         # ✨ Generated
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md     # ✨ Generated
│   │   └── feature_request.md # ✨ Generated
│   └── PULL_REQUEST_TEMPLATE.md # ✨ Generated
└── docs/
    ├── installation.md        # ✨ Generated (detailed install guide)
    └── troubleshooting.md     # ✨ Generated (common issues)
```

### Quality Metrics
- **Completeness**: 15+ README sections (installation, usage, API, troubleshooting)
- **Readability**: Grammarly score 95+
- **SEO Optimization**: GitHub search-optimized keywords
- **Accessibility**: Clear headings, descriptive links
- **Time Saved**: 3-4 hours manual README writing

### Business Value
1. **Adoption**: 40% increase in GitHub stars (professional first impression)
2. **Support Reduction**: 60% fewer "how to install" issues
3. **Contributor Onboarding**: New contributors productive in <30 minutes
4. **Documentation Consistency**: Auto-sync with code changes

## Tips & Best Practices

### 1. Keep Package.json Rich
```json
{
  "name": "db-migrate-pro",
  "description": "Detailed description helps generate better README",
  "keywords": ["migration", "database", "sql"],
  "homepage": "https://db-migrate-pro.dev",
  "bugs": "https://github.com/yourorg/db-migrate-pro/issues",
  "repository": {
    "type": "git",
    "url": "https://github.com/yourorg/db-migrate-pro.git"
  }
}
```

### 2. Add Badges Automatically
```bash
# Skill auto-detects and adds relevant badges:
# - npm version (from package.json)
# - Build status (from .github/workflows)
# - Coverage (from jest config)
# - License (from package.json)
```

### 3. Include Visual Examples
```bash
# Add screenshots to docs/screenshots/
docs/screenshots/
├── cli-demo.gif          # CLI usage demo
├── status-output.png     # Command output
└── architecture.png      # System diagram

# Skill auto-includes in README
```

### 4. Auto-Update on Version Bump
```bash
# .git/hooks/post-commit
#!/bin/bash
if git diff HEAD~1 package.json | grep '"version"'; then
  npx claude-flow sparc run documentation "Update README version badges"
  git add README.md
  git commit --amend --no-edit
fi
```

### 5. Multi-Language README Support
```bash
# Generate README in multiple languages
npx claude-flow sparc run documentation "Generate README with i18n support"

# Creates:
# README.md (English)
# README.zh-CN.md (Chinese)
# README.es.md (Spanish)
```

## Advanced Usage

### Custom README Templates
```bash
# Use organization-specific template
npx claude-flow sparc run documentation \
  --template .github/README.template.md \
  "Generate README with company branding"
```

### Monorepo README Generation
```bash
# Generate README for each package
for pkg in packages/*; do
  npx claude-flow sparc run documentation \
    --context "$pkg/package.json" \
    "Generate README for $(basename $pkg)"
done
```

### Integration with Documentation Sites
```bash
# Generate README + Docusaurus docs
npx claude-flow sparc run documentation \
  "Generate README and sync to Docusaurus docs/"
```

## Troubleshooting

### Issue: README Too Long (>500 lines)
**Solution**: Enable "concise mode" or split into separate docs

### Issue: Missing Examples Section
**Solution**: Add JSDoc `@example` tags to main functions

### Issue: Outdated Badges
**Solution**: Enable auto-update via git hooks or CI/CD

### Issue: Missing API Reference
**Solution**: Ensure public functions have JSDoc comments

## Next Steps

1. **Explore Example 3**: Learn about code documentation generation
2. **Try the skill**: Run `npx claude-flow sparc run documentation "Generate README"`
3. **Customize**: Edit templates in `docs/templates/readme-template.md`
4. **Automate**: Set up git hooks for auto-regeneration
