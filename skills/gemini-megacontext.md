---
skill: gemini-megacontext
description: Analyze entire codebases with Gemini's 1 million token context window - process 30K lines at once
tags: [gemini, codebase-analysis, architecture, large-context, multi-file]
version: 1.0.0
---

# Gemini Mega-Context Skill

## Purpose
Leverage Gemini CLI's massive 1 million token context window to analyze entire codebases, architectural patterns, and multi-file dependencies in a single pass - something Claude Code's context window cannot achieve.

## Unique Capability
**What Claude Code Can't Do**: Claude Code has limited context window. Gemini 2.5 Pro can process up to 1 million tokens (~1,500 pages or 30,000 lines of code) simultaneously, enabling whole-codebase analysis without losing context.

## When to Use

### Perfect For:
✅ Analyzing entire project architecture
✅ Understanding multi-file dependencies across large codebases
✅ Refactoring that requires understanding the whole system
✅ Generating comprehensive documentation from full codebase
✅ Finding patterns and anti-patterns across all files
✅ Onboarding to unfamiliar large projects
✅ Security audits requiring full codebase awareness
✅ Migration planning (understand everything before changing)

### Don't Use When:
❌ Working with single file or small module (use Claude Code)
❌ Need complex problem-solving (Claude is better)
❌ Writing new features (Gemini gets stuck in loops per user feedback)
❌ Need iterative refinement (Gemini switches to Flash after 5 min)

## How It Works

This skill spawns a **Gemini Mega-Context Agent** that:
1. Uses `gemini --all-files` to load your entire codebase
2. Leverages 1M token context for comprehensive analysis
3. Returns architectural insights, dependency maps, or refactoring plans
4. Provides results back to Claude Code for user presentation

## Usage

### Basic Codebase Analysis
```
/gemini-megacontext
```

### With Specific Question
```
/gemini-megacontext "Explain the complete architecture and how all components interact"
```

### Detailed Analysis
```
/gemini-megacontext "Map all database queries across the entire codebase and identify N+1 patterns"
```

## Input Examples

```bash
# Architecture analysis
/gemini-megacontext "Document the full system architecture with component interactions"

# Dependency mapping
/gemini-megacontext "Create a dependency graph showing how all modules relate"

# Security audit
/gemini-megacontext "Identify all authentication and authorization patterns across the codebase"

# Migration planning
/gemini-megacontext "Analyze entire codebase for Python 2 to 3 migration requirements"

# Code patterns
/gemini-megacontext "Find all API endpoints and document their authentication methods"

# Refactoring scope
/gemini-megacontext "Identify all files that would need changes to rename User to Account"
```

## Output

The agent provides:
- **Architectural Overview**: How the system is structured
- **Component Interactions**: How pieces fit together
- **Dependency Map**: What depends on what
- **Pattern Analysis**: Common patterns and anti-patterns found
- **File References**: Specific locations with file:line citations
- **Recommendations**: Improvement suggestions based on full context

## Real-World Examples

### Example 1: Architecture Documentation
```
Task: "Document our microservices architecture"

Agent analyzes all services simultaneously and provides:
- Service dependency graph
- API contract documentation
- Database schema relationships
- Authentication flow across services
- Configuration management patterns
```

### Example 2: Refactoring Impact Analysis
```
Task: "If we change the User model, what breaks?"

Agent scans entire codebase and identifies:
- 47 files with direct User references
- 12 database migrations to update
- 8 API endpoints that return User data
- 15 frontend components displaying user info
- 3 background jobs processing users
```

### Example 3: Security Audit
```
Task: "Find all places where we handle sensitive data"

Agent reviews full codebase and reports:
- All database fields storing PII
- API endpoints exposing sensitive data
- Logging statements that might leak secrets
- File upload handlers and validation
- Authentication/authorization patterns
```

## Technical Details

### Gemini CLI Command Pattern
```bash
cd /path/to/project
gemini --all-files "Your analysis question here"
```

### Context Window Specs
- **Capacity**: 1 million tokens
- **Equivalent**: ~1,500 pages or ~30,000 lines of code
- **Best for**: Projects under 30K LOC
- **Larger projects**: Agent will prioritize most relevant files

### Free Tier Limits
- 60 requests per minute
- 1,000 requests per day
- No cost with your Google account

## Best Practices

### Provide Clear Scope
✅ "Analyze authentication flow across all services"
✅ "Map database query patterns throughout the codebase"
❌ "Tell me about the code" (too vague)

### Be Specific About Output
✅ "Create a markdown dependency diagram"
✅ "List all files affected by this change"
❌ "Give me information" (unclear deliverable)

### Use for Breadth, Not Depth
✅ "What are all the API endpoints in this project?"
✅ "How do services communicate with each other?"
❌ "Implement a new feature" (Gemini gets stuck per feedback)

## Limitations (Based on Real Developer Feedback)

⚠️ **Known Issues**:
- May generate errors in its own analysis (missing XML tags, etc.)
- Can get stuck in loops trying to fix its mistakes
- Switches to Flash model after 5 minutes (Flash is "awful at coding")
- Slower than Claude for complex reasoning
- Not great for implementation tasks

✅ **Strengths**:
- Breadth of analysis is excellent
- Can summarize entire folders effectively
- Great for onboarding and auditing
- Powerful for architectural understanding

## Integration with Workflow

### Works Well With:
- Claude Code for implementation after analysis
- Root Cause Analyzer for debugging across large codebases
- Documentation generation agents
- SPARC architecture phase

### Typical Workflow:
1. Use `/gemini-megacontext` to understand architecture
2. Use Claude Code to implement specific changes
3. Use `/gemini-megacontext` again to verify impact
4. Use Claude Code for refinement and testing

## Cost Considerations

- **Free tier**: 60 req/min, 1000/day
- **No credit card required** with Google account
- **Perfect for**: Daily architecture reviews and analysis
- **Upgrade**: Available if you need more quota

## Troubleshooting

### "Context too large" Error
→ Agent will automatically prioritize most relevant files

### Analysis Too Shallow
→ Provide more specific questions about what you need

### Takes Too Long
→ Gemini CLI might be analyzing large codebase, wait for completion

### Keeps Making Mistakes
→ Known issue per user feedback; use Claude Code for implementation instead

## Related Skills

- `gemini-search`: For finding latest documentation while analyzing
- `gemini-extensions`: For Figma/Stripe integration during design analysis
- `root-cause-analyzer`: For deep debugging after identifying areas of concern
- `multi-model`: Let orchestrator decide when to use mega-context

## Success Indicators

✅ Got comprehensive architectural understanding
✅ Identified all affected files for a change
✅ Mapped dependencies across entire system
✅ Understood how components interact
✅ Generated documentation from full codebase
✅ Found patterns across many files

---

**Remember**: Use Gemini's mega-context for BREADTH (understanding the whole system), use Claude Code for DEPTH (implementing and refining solutions).
