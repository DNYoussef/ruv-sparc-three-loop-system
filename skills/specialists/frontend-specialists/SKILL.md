---
name: frontend-specialists
description: Comprehensive frontend development system with specialist agents for
  React, Vue, UI components, CSS styling, accessibility, and performance optimization.
  Use when building modern web applications, component libraries, or optimizing frontend
  performance. Includes production-ready scripts, templates, and testing utilities.
category: Frontend Development
complexity: High
tier: Gold
triggers:
- frontend
- react
- vue
- ui components
- css
- accessibility
- performance optimization
- component library
version: 1.0.0
tags:
- specialists
- domain-expert
author: ruv
---

# Frontend Specialists

A comprehensive frontend development system combining specialized agents, production scripts, and reusable templates for modern web application development.

## Purpose

Provide complete frontend development capabilities through specialized agents and tools. Each specialist focuses on a specific domain (React, Vue, UI components, CSS, accessibility, performance) while sharing common scripts and templates for rapid development.

## System Architecture

```
frontend-specialists/
├── skill.md (this file)
├── react-specialist/      # React 18+ specialist
├── vue-specialist/        # Vue 3 specialist (future)
├── ui-component-builder/  # Component library specialist (future)
├── resources/
│   ├── scripts/           # Production automation scripts
│   │   ├── component-generator.js
│   │   ├── bundle-analyzer.js
│   │   ├── a11y-checker.js
│   │   └── performance-profiler.js
│   └── templates/         # Reusable component templates
│       ├── react-component.tsx
│       ├── vue-component.vue
│       └── component.config.json
└── tests/                 # Comprehensive test suites
    ├── component-generator.test.js
    ├── bundle-analyzer.test.js
    └── a11y-checker.test.js
```

## Available Specialists

### 1. React Specialist (`react-specialist`)
**Triggers**: "React", "React 18", "Next.js", "hooks", "server components"

**Capabilities**:
- React 18+ with concurrent rendering
- Next.js 13+ App Router with Server Components
- State management (Zustand, Redux Toolkit, Jotai)
- Performance optimization (memoization, code splitting)
- Testing with React Testing Library

**Agent**: `react-developer`

### 2. Vue Specialist (Planned)
**Triggers**: "Vue", "Vue 3", "Composition API", "Nuxt"

**Capabilities**:
- Vue 3 Composition API
- Nuxt 3 with server-side rendering
- Pinia state management
- Vue testing utilities

**Agent**: `vue-developer`

### 3. UI Component Builder (Planned)
**Triggers**: "component library", "design system", "UI components"

**Capabilities**:
- Component library architecture
- Storybook integration
- Theming and customization
- Component documentation

**Agent**: `ui-component-builder`

### 4. CSS Styling Specialist (Planned)
**Triggers**: "CSS", "Tailwind", "styled-components", "CSS-in-JS"

**Capabilities**:
- Modern CSS (Grid, Flexbox, Container Queries)
- Tailwind CSS utility-first design
- CSS-in-JS (styled-components, Emotion)
- Responsive design patterns

**Agent**: `css-styling-specialist`

### 5. Accessibility Specialist (Planned)
**Triggers**: "accessibility", "a11y", "WCAG", "ARIA"

**Capabilities**:
- WCAG 2.1 AA/AAA compliance
- ARIA patterns and semantics
- Keyboard navigation
- Screen reader testing

**Agent**: `accessibility-specialist`

### 6. Performance Optimizer (Planned)
**Triggers**: "performance", "optimization", "bundle size", "lighthouse"

**Capabilities**:
- Bundle size optimization
- Code splitting strategies
- Image optimization
- Performance monitoring

**Agent**: `frontend-performance-optimizer`

## Shared Scripts

### 1. Component Generator (`component-generator.js`)
**Purpose**: Scaffold new components with best practices

```bash
# Generate React component
node resources/scripts/component-generator.js --framework react --name Button --type functional

# Generate Vue component
node resources/scripts/component-generator.js --framework vue --name Card --type composition
```

**Features**:
- TypeScript support
- Test file generation
- Storybook story creation
- Accessibility defaults (ARIA, semantic HTML)

### 2. Bundle Analyzer (`bundle-analyzer.js`)
**Purpose**: Analyze and optimize bundle sizes

```bash
# Analyze bundle
node resources/scripts/bundle-analyzer.js --project ./my-app --threshold 200

# Output: Bundle report with recommendations
```

**Features**:
- Webpack/Vite/Rollup support
- Tree-shaking analysis
- Dependency visualization
- Size recommendations

### 3. Accessibility Checker (`a11y-checker.js`)
**Purpose**: Automated accessibility audits

```bash
# Check accessibility
node resources/scripts/a11y-checker.js --url http://localhost:3000 --level AA

# Output: WCAG violations with fixes
```

**Features**:
- axe-core integration
- WCAG 2.1 AA/AAA validation
- Color contrast checking
- Keyboard navigation testing

### 4. Performance Profiler (`performance-profiler.js`)
**Purpose**: Performance metrics and optimization suggestions

```bash
# Profile performance
node resources/scripts/performance-profiler.js --url http://localhost:3000

# Output: Lighthouse scores, Web Vitals, optimization tips
```

**Features**:
- Lighthouse integration
- Core Web Vitals (LCP, FID, CLS)
- Performance budget validation
- Optimization recommendations

## Shared Templates

### 1. React Component Template (`react-component.tsx`)
- Functional component with TypeScript
- Props interface
- Accessibility attributes
- CSS module import
- Test file structure

### 2. Vue Component Template (`vue-component.vue`)
- Composition API structure
- TypeScript support
- Scoped styles
- Accessibility setup

### 3. Component Config (`component.config.json`)
- Component metadata
- Variant definitions
- Theming variables
- Documentation links

## Quality Criteria

All frontend specialists adhere to:

- ✅ **TypeScript**: Strict mode enabled
- ✅ **Accessibility**: WCAG 2.1 AA minimum
- ✅ **Performance**: Lighthouse score ≥90
- ✅ **Testing**: Coverage ≥80%
- ✅ **Bundle Size**: <200KB initial load (gzipped)
- ✅ **Browser Support**: Last 2 versions, >0.5% market share

## Testing Strategy

### Unit Tests (Jest/Vitest)
```javascript
// Component logic, hooks, utilities
describe('Button', () => {
  it('renders with correct variant', () => {
    // Test implementation
  });
});
```

### Integration Tests (React Testing Library)
```javascript
// User interactions, form submissions
test('submits form on button click', async () => {
  // Test implementation
});
```

### E2E Tests (Playwright)
```javascript
// Full user workflows
test('complete checkout flow', async ({ page }) => {
  // Test implementation
});
```

### Visual Regression (Chromatic/Percy)
```javascript
// UI consistency checks
test('button snapshot', () => {
  // Snapshot comparison
});
```

## Best Practices

**1. Component Architecture**
- Single Responsibility Principle
- Composition over inheritance
- Props validation with TypeScript
- Controlled vs uncontrolled components

**2. State Management**
- Local state for UI (useState)
- Global state for app-wide data (Zustand/Redux)
- Server state for API data (TanStack Query)
- URL state for navigation (Next.js router)

**3. Performance Optimization**
- Code splitting with lazy loading
- Memoization (React.memo, useMemo, useCallback)
- Virtual scrolling for large lists
- Image optimization (next/image, WebP)

**4. Accessibility**
- Semantic HTML elements
- ARIA attributes when needed
- Keyboard navigation support
- Focus management
- Color contrast ratios

**5. Testing**
- Test user behavior, not implementation
- Accessibility testing in all tests
- Mock external dependencies
- Integration tests over unit tests

## Integration with Other Skills

- `typescript-specialist`: TypeScript patterns and advanced types
- `wcag-accessibility`: Deep accessibility compliance
- `testing-quality`: Advanced testing strategies
- `docker-containerization`: Containerizing frontend apps
- `cicd-intelligent-recovery`: CI/CD pipeline integration

## MCP Tools

### Sandbox Testing
```javascript
// Create isolated frontend sandbox
mcp__flow-nexus__sandbox_create({
  template: "react", // or "nextjs", "vanilla"
  env_vars: { NODE_ENV: "development" }
})
```

### Browser Automation
```javascript
// Visual testing with Playwright
mcp__playwright__browser_snapshot()
mcp__playwright__browser_take_screenshot({ fullPage: true })
```

### Memory Persistence
```javascript
// Store component patterns
mcp__memory-mcp__memory_store({
  key: "frontend/react-patterns",
  value: { pattern: "compound-components", usage: "..." }
})
```

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Component Creation | 10-15 min | Time to scaffold + test |
| Feature Development | 1-2 hours | Implementation + tests |
| Lighthouse Score | ≥90 | Performance audit |
| Test Coverage | ≥80% | Jest/Vitest reports |
| Bundle Size | <200KB | Webpack/Vite analysis |
| Accessibility | WCAG 2.1 AA | axe-core validation |

## Troubleshooting

### Issue: Hydration Mismatch (Next.js)
**Solution**: Ensure server/client HTML match. Use `suppressHydrationWarning` for time-dependent content.

### Issue: Slow Performance with Large Lists
**Solution**: Implement virtualization (react-window, react-virtual).

### Issue: Bundle Size Too Large
**Solution**: Use bundle analyzer, code splitting, tree-shaking, dynamic imports.

### Issue: Accessibility Violations
**Solution**: Run axe-core, add ARIA labels, semantic HTML, keyboard support.

## Workflow: Complete Feature Development

**Step 1: Initialize with Specialist**
```bash
# Trigger appropriate specialist
"I need to build a React dashboard with charts"
→ Activates react-specialist
```

**Step 2: Generate Component Scaffolding**
```bash
node resources/scripts/component-generator.js \
  --framework react \
  --name Dashboard \
  --type functional
```

**Step 3: Implement with Best Practices**
- Use shared templates
- Follow TypeScript patterns
- Implement accessibility defaults

**Step 4: Test Comprehensively**
- Unit tests for logic
- Integration tests for interactions
- Accessibility tests with axe-core

**Step 5: Optimize Performance**
```bash
node resources/scripts/bundle-analyzer.js --project ./my-app
node resources/scripts/performance-profiler.js --url http://localhost:3000
```

**Step 6: Validate Quality**
```bash
node resources/scripts/a11y-checker.js --url http://localhost:3000 --level AA
npm run test -- --coverage
```

## Related Documentation

- [React Specialist Skill](./react-specialist/skill.md)
- [Component Generator Script](./resources/scripts/component-generator.js)
- [Bundle Analyzer Script](./resources/scripts/bundle-analyzer.js)
- [Accessibility Checker Script](./resources/scripts/a11y-checker.js)
- [Performance Profiler Script](./resources/scripts/performance-profiler.js)

---

**Skill Version**: 2.0.0 (Gold Tier)
**Last Updated**: 2025-11-02
**Maintained By**: Frontend Specialists Team
**Agent Coordination**: `react-developer`, `vue-developer`, `ui-component-builder`, `css-styling-specialist`, `accessibility-specialist`, `frontend-performance-optimizer`