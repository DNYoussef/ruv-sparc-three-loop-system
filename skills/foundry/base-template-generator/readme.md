# Base Template Generator

**Version**: 1.0.0 (Bronze Tier)
**Purpose**: Generate clean, minimal boilerplate code templates for common project types

## 🎯 What This Is

Base Template Generator creates foundational project templates with essential structure and best practices baked in. It generates clean, unopinionated boilerplate that serves as a solid starting point for new projects.

## 🚀 Quick Start

1. **Specify project type**: Express the type of project you need (e.g., "Node.js API", "React app", "Python CLI")
2. **Auto-generate structure**: Receive complete directory structure, configuration files, and basic implementation
3. **Start building**: Begin implementing features on top of the solid foundation

## 📋 When to Use This Skill

Use base-template-generator when you need:
- **Starting fresh projects**: Get a clean foundation without bloat
- **Consistent structure**: Ensure all projects follow best practices
- **Quick bootstrapping**: Skip repetitive boilerplate setup
- **Learning templates**: Study well-organized project structures

**Auto-triggers on**:
- "boilerplate"
- "starter template"
- "scaffold project"
- "initialize [project type]"

## 📁 Structure Overview

```
base-template-generator/
├── README.md                   # This file - overview & navigation
├── examples/                   # Concrete usage scenarios
│   └── example-1-basic.md      # Basic template generation walkthrough
```

## 🎨 Template Categories

### Backend
- Node.js/Express API
- Python FastAPI
- Go HTTP server
- REST API with authentication

### Frontend
- React SPA
- Vue.js application
- Static HTML/CSS/JS site

### CLI Tools
- Python CLI with argparse
- Node.js CLI with commander
- Bash script utilities

### Full Stack
- MERN stack (MongoDB, Express, React, Node)
- Django + React
- Next.js application

## 📊 Quality Standards

**Quality Tier**: Bronze
- ✅ README.md (this file)
- ✅ 1 example demonstrating basic usage
- **Total**: 3 files minimum

## 🔧 Features

### Core Capabilities
- **Clean architecture**: Organized directory structure
- **Configuration files**: package.json, .gitignore, .env.example
- **Basic implementation**: Starter code with comments
- **Documentation**: README and setup instructions
- **Best practices**: ESLint, Prettier, testing setup

### What's NOT Included
- Complex features or business logic
- Database migrations or seed data
- Advanced authentication/authorization
- Production deployment configurations
- CI/CD pipelines (see `cicd-intelligent-recovery` skill)

## 💡 Design Philosophy

> "Start simple, build complex. The best template is the one that gets out of your way."

Base Template Generator follows these principles:
1. **Minimal dependencies**: Only essential packages
2. **Conventional structure**: Industry-standard layouts
3. **Clear documentation**: Self-documenting code
4. **Easy to modify**: Unopinionated and flexible
5. **Production-ready**: Linting, testing, error handling included

## 📚 Examples

See `examples/` directory for detailed walkthroughs:
- **example-1-basic.md**: Generate a Node.js Express API template

## 🔗 Related Skills

- **skill-forge**: For creating new skill templates
- **sparc-methodology**: For systematic feature development
- **cicd-intelligent-recovery**: For deployment automation

## ✨ Get Started

```bash
# Request template generation
"Generate a Node.js Express API template with TypeScript"

# Or be specific about structure
"Create Python FastAPI boilerplate with pytest and poetry"
```

The skill will:
1. Analyze your requirements
2. Generate complete directory structure
3. Create configuration files
4. Provide starter implementation
5. Include setup instructions

---

**Maintained by**: Claude Code (Sonnet 4.5)
**License**: Same as ruv-sparc-three-loop-system
**Support**: Create issue in repository
