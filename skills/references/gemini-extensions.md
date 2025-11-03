---
skill: gemini-extensions
description: Access Gemini's 70+ extensions ecosystem including Figma, Stripe, Postman, Shopify
tags: [gemini, extensions, figma, stripe, integrations, third-party]
version: 1.0.0
---

# Gemini Extensions Skill

## Purpose
Leverage Gemini CLI's ecosystem of 70+ extensions to integrate with Figma, Stripe, Postman, Shopify, and other third-party services that Claude Code cannot directly access.

## Unique Capability
**What Claude Code Can't Do**: Direct integration with design tools (Figma), payment APIs (Stripe), API testing (Postman), e-commerce (Shopify), and 70+ other extensions. Gemini CLI provides native integrations via its extension system.

## When to Use

### Perfect For:
✅ Extracting designs from Figma and generating code
✅ Testing Stripe payment integrations
✅ Running Postman API collections
✅ Querying Shopify store data
✅ Accessing Dynatrace, Elastic, Snyk, Harness data
✅ Any task requiring third-party tool integration

### Available Extensions:
- **Figma**: Extract frames, components, design tokens → generate code
- **Stripe**: Test APIs, query payment data, validate integrations
- **Postman**: Run collections, test endpoints, validate APIs
- **Shopify**: Query products, orders, customer data
- **Dynatrace**: Performance monitoring data
- **Elastic**: Search and analytics queries
- **Snyk**: Security vulnerability scanning
- **Harness**: CD pipeline integration

Plus 60+ community extensions on GitHub.

## Usage

```bash
# Install extension
/gemini-extensions --install figma

# Use Figma
/gemini-extensions "Extract components from Figma frame XYZ and generate React code"

# Use Stripe
/gemini-extensions "Test Stripe payment intent creation with test card"

# Use Postman
/gemini-extensions "Run the 'User API' Postman collection and report results"
```

## Real Examples

### Figma → Code
```
/gemini-extensions "Extract button component from Figma frame 'Components/Buttons' and generate React component with TypeScript and styled-components"
```

### Stripe Testing
```
/gemini-extensions "Create a test payment intent for $50 USD using Stripe test API, verify webhook firing"
```

### API Testing
```
/gemini-extensions "Run my Postman collection 'API Tests v2' and identify any failing endpoints"
```

---

**See full documentation** in `.claude/agents/gemini-extensions-agent.md`
