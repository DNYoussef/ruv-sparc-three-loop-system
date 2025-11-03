---
name: accessibility-audit
description: WCAG 2.1 AA/AAA compliance audit with automated testing and manual review checklist
category: audit-commands
version: 2.0.0
requires_mcp: false
---

# ♿ Accessibility Audit - WCAG 2.1 Compliance

**Command**: A11y Compliance Audit | **Category**: Quality Audits

Comprehensive accessibility audit ensuring WCAG 2.1 Level AA compliance with automated testing and manual review.

## Audit Execution

```bash
#!/bin/bash
# Accessibility Audit Workflow

echo "♿ ACCESSIBILITY AUDIT"
echo "====================="

# Phase 1: Automated axe-core Testing
npx @axe-core/cli http://localhost:3000 \
  --save accessibility/axe-results.json \
  --tags wcag2a,wcag2aa

# Phase 2: Lighthouse Accessibility Score
npx lighthouse http://localhost:3000 \
  --only-categories=accessibility \
  --output json \
  --output-path accessibility/lighthouse.json

# Phase 3: Pa11y Testing
npx pa11y http://localhost:3000 \
  --standard WCAG2AA \
  --reporter json > accessibility/pa11y-results.json

# Phase 4: Generate Report
cat > accessibility/report.md <<EOF
# Accessibility Audit Report
Generated: $(date)

## Scores
- Lighthouse: $(jq '.categories.accessibility.score * 100' accessibility/lighthouse.json)
- Axe Violations: $(jq '.violations | length' accessibility/axe-results.json)
- Pa11y Issues: $(jq 'length' accessibility/pa11y-results.json)

## Critical Issues
$(jq -r '.violations[] | select(.impact == "critical") | "- [\(.id)] \(.description)"' accessibility/axe-results.json)

## Manual Review Required
- Keyboard navigation
- Screen reader compatibility
- Color contrast (complex cases)
- Focus management
EOF

echo "✅ Audit complete: accessibility/report.md"
```

## WCAG 2.1 Compliance Checklist

### Perceivable
- [ ] **1.1.1** Text alternatives for non-text content
- [ ] **1.2.1** Audio/video alternatives
- [ ] **1.3.1** Semantic HTML structure
- [ ] **1.4.3** Color contrast ratio ≥ 4.5:1 (AA) or ≥ 7:1 (AAA)
- [ ] **1.4.11** Non-text contrast ≥ 3:1

### Operable
- [ ] **2.1.1** Keyboard accessible
- [ ] **2.1.2** No keyboard trap
- [ ] **2.4.3** Logical focus order
- [ ] **2.4.7** Visible focus indicator
- [ ] **2.5.5** Target size ≥ 44×44 pixels

### Understandable
- [ ] **3.1.1** Language specified
- [ ] **3.2.1** No unexpected context changes
- [ ] **3.3.1** Error identification
- [ ] **3.3.2** Labels or instructions provided

### Robust
- [ ] **4.1.1** Valid HTML
- [ ] **4.1.2** Name, role, value for UI components
- [ ] **4.1.3** Status messages

**Agent Status**: Production-Ready
**Version**: 2.0.0
