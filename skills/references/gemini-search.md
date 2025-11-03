---
skill: gemini-search
description: Get real-time web information using Gemini's built-in Google Search grounding
tags: [gemini, web-search, real-time, documentation, current-info]
version: 1.0.0
---

# Gemini Search Skill

## Purpose
Leverage Gemini CLI's built-in Google Search grounding to fetch real-time web information, validate current best practices, and access the latest documentation - capabilities Claude Code doesn't have natively.

## Unique Capability
**What Claude Code Can't Do**: Claude Code's knowledge has a cutoff date and cannot access real-time web information during analysis. Gemini CLI has built-in Google Search integration that grounds responses in current web content with citations.

## When to Use

### Perfect For:
✅ Checking latest API documentation
✅ Finding current library versions and changelogs
✅ Validating best practices against current standards
✅ Researching breaking changes in dependencies
✅ Comparing current technology options
✅ Finding solutions to recent issues
✅ Checking security advisories and CVEs
✅ Verifying current framework conventions

### Don't Use When:
❌ Information is in your local codebase (use Claude Code)
❌ Need deep implementation logic (use Claude Code)
❌ Question doesn't require current web information
❌ Working with proprietary/internal systems

## How It Works

This skill spawns a **Gemini Search Agent** that:
1. Uses Gemini CLI's `@search` tool or built-in Google Search grounding
2. Fetches current web content with citations
3. Grounds analysis in real-time information
4. Returns findings with source URLs to Claude Code

## Usage

### Basic Search
```
/gemini-search
```

### With Specific Query
```
/gemini-search "What are the breaking changes in React 19?"
```

### Detailed Research
```
/gemini-search "Compare authentication approaches for Next.js 15 apps with latest security best practices"
```

## Input Examples

```bash
# API Documentation
/gemini-search "Latest Stripe API authentication methods 2025"

# Breaking Changes
/gemini-search "What changed in Python 3.13 that would break my code?"

# Best Practices
/gemini-search "Current best practices for securing Node.js REST APIs"

# Version Information
/gemini-search "Is TensorFlow 2.16 stable? What are known issues?"

# Framework Conventions
/gemini-search "How should I structure a Next.js 15 app directory?"

# Security Research
/gemini-search "Recent vulnerabilities in Express.js and mitigation strategies"

# Technology Comparison
/gemini-search "Compare Prisma vs Drizzle ORM for TypeScript projects 2025"
```

## Output

The agent provides:
- **Direct Answer**: Response to your query
- **Source Citations**: URLs where information was found
- **Current Status**: What's latest/stable/recommended
- **Key Findings**: Bullet points of important info
- **Recommendations**: Based on current web consensus
- **Related Resources**: Links to docs, guides, discussions

## Real-World Examples

### Example 1: API Changes
```
Query: "What changed in OpenAI API v2?"

Agent searches and returns:
- New endpoint structure with examples
- Deprecated methods and replacements
- Migration guide links
- Breaking changes to watch for
- Source: Official OpenAI docs + dev discussions
```

### Example 2: Security Advisory
```
Query: "Are there security issues with lodash 4.17.20?"

Agent searches and returns:
- CVE-2020-8203 prototype pollution vulnerability
- Affected versions: < 4.17.21
- Severity: High
- Fix: Upgrade to 4.17.21 or higher
- Sources: npm advisory, Snyk, GitHub issues
```

### Example 3: Framework Best Practices
```
Query: "How should I handle authentication in Next.js 15?"

Agent searches and returns:
- Recommended approaches (NextAuth.js, Clerk, Auth.js)
- App router vs pages router differences
- Server components considerations
- Code examples from official docs
- Sources: Next.js docs, Vercel guides, community tutorials
```

## Technical Details

### Gemini CLI Command Pattern
```bash
# Using @search tool
gemini "@search What are the latest Rust 2024 features?"

# Natural prompt with automatic search
gemini "Search for current best practices in GraphQL API security"

# Specific URL analysis
gemini "@search https://github.com/facebook/react/releases/tag/v19.0.0"
```

### Search Grounding Features
- **Automatic invocation**: GoogleSearch tool activates when needed
- **Citation**: Results include source URLs
- **Real-time**: Fetches current web content
- **Context-aware**: Combines search results with Gemini's reasoning

### Free Tier Limits
- 60 requests per minute
- 1,000 requests per day
- No additional cost for search grounding

## Best Practices

### Be Specific About Version/Date
✅ "React 19 breaking changes"
✅ "Node.js 22 LTS features"
❌ "React changes" (too vague)

### Request Sources
✅ "Find official documentation for..."
✅ "What do security researchers say about..."
✅ "Check GitHub issues for..."

### Combine with Context
✅ "I'm using Express 4.18, what security updates are recommended?"
✅ "For a TypeScript project, what's the current best testing framework?"

### Verify Critical Information
For security or architecture decisions:
1. Use `/gemini-search` to find current information
2. Review sources provided
3. Cross-reference with official docs
4. Make decision with Claude Code

## Advantages Over Claude Code WebSearch

### Gemini Search Advantages:
✅ Grounding built into analysis (not separate step)
✅ Can combine search with codebase context
✅ Better for technical documentation
✅ Free tier more generous

### When to Use Each:
- **Gemini Search**: Technical docs, API references, framework info
- **Claude WebSearch**: General information, news, broader research

## Integration with Workflow

### Typical Workflow:
1. Use `/gemini-search` to find current best practices
2. Use Claude Code to implement based on findings
3. Use `/gemini-search` to validate approach
4. Use Claude Code for refinement and testing

### Works Well With:
- `gemini-megacontext`: Search for info, then analyze with full context
- `codex-reasoning`: Search for approaches, implement with Codex
- `multi-model`: Let orchestrator decide when search is needed

## Cost Considerations

- **Free tier**: 60 req/min, 1000/day
- **No credit card required** with Google account
- **Search included**: No extra cost for Google Search grounding
- **Perfect for**: Daily development research and validation

## Troubleshooting

### No Sources Returned
→ Gemini found answer in its training data, not web search
→ Try: "Search the web for..." to force search

### Outdated Information
→ Be specific about date: "2025 best practices for..."
→ Request: "Find the most recent information about..."

### Too Generic
→ Add specificity: version numbers, framework names, use case
→ Example: "Next.js 15 App Router authentication" not "auth in Next"

### Search Not Triggered
→ Explicitly mention: "Search for...", "Find current...", "What's latest..."

## Limitations

⚠️ **Dependent on Search Quality**:
- Results only as good as what Google finds
- May return outdated info if it ranks highly
- Personal blogs may outrank official docs

⚠️ **Can't Access Private Resources**:
- No access to internal wikis
- No access to private GitHub repos
- No access to paywalled content

✅ **Best For**:
- Public documentation
- Open source projects
- Community best practices
- Security advisories
- Official changelogs

## Related Skills

- `gemini-megacontext`: Use search findings to inform codebase analysis
- `gemini-extensions`: Search for extension documentation
- `root-cause-analyzer`: Search for similar issues and solutions
- `multi-model`: Let orchestrator decide when to search

## Success Indicators

✅ Got current, accurate information with sources
✅ Found latest API documentation
✅ Identified breaking changes or updates
✅ Validated best practices against current standards
✅ Discovered security advisories or patches
✅ Sources are authoritative and recent

---

**Remember**: Use Gemini Search for CURRENT information Claude Code doesn't have access to. Always verify critical information from multiple sources.
