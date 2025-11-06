---
name: wcag-accessibility
description: WCAG 2.1 AA/AAA accessibility compliance specialist for ARIA attributes, keyboard navigation, screen reader testing, color contrast validation, semantic HTML, and automated a11y testing with axe-core/Lighthouse. Use when ensuring web accessibility, meeting legal compliance (ADA, Section 508), implementing inclusive design, or requiring WCAG best practices. Handles focus management, live regions, accessible forms, and assistive technology compatibility.
category: Compliance
complexity: Medium
triggers: ["wcag", "accessibility", "a11y", "aria", "screen reader", "keyboard navigation", "color contrast", "ada compliance", "section 508", "inclusive design"]
---

# WCAG Accessibility Specialist

Expert web accessibility implementation for WCAG 2.1 AA/AAA compliance and inclusive design.

## Purpose

Comprehensive accessibility expertise including ARIA attributes, keyboard navigation, screen reader compatibility, color contrast, semantic HTML, and automated testing. Ensures web applications are usable by people with disabilities and meet legal requirements.

## When to Use

- Implementing WCAG 2.1 AA or AAA compliance
- Fixing accessibility violations found in audits
- Building accessible components and forms
- Testing with screen readers (NVDA, JAWS, VoiceOver)
- Meeting ADA or Section 508 legal requirements
- Implementing keyboard-only navigation
- Optimizing for assistive technologies

## Prerequisites

**Required**: HTML/CSS, JavaScript basics, understanding of semantic HTML

**Agents**: `tester`, `reviewer`, `code-analyzer`, `coder`

## Core Workflows

### Workflow 1: Accessible Form Implementation

**Step 1: Semantic HTML with Labels**

```html
<!-- ✅ GOOD: Proper labels and structure -->
<form>
  <div class="form-group">
    <label for="email">Email Address *</label>
    <input
      type="email"
      id="email"
      name="email"
      required
      aria-required="true"
      aria-describedby="email-help email-error"
    />
    <small id="email-help">We'll never share your email.</small>
    <span id="email-error" role="alert" aria-live="polite"></span>
  </div>

  <fieldset>
    <legend>Choose your plan</legend>
    <div>
      <input type="radio" id="plan-free" name="plan" value="free" />
      <label for="plan-free">Free</label>
    </div>
    <div>
      <input type="radio" id="plan-pro" name="plan" value="pro" />
      <label for="plan-pro">Pro</label>
    </div>
  </fieldset>

  <button type="submit">Subscribe</button>
</form>

<!-- ❌ BAD: No labels, placeholder only -->
<form>
  <input type="email" placeholder="Email" />
  <input type="text" placeholder="Plan" />
  <button>Submit</button>
</form>
```

**Step 2: Client-Side Validation with Accessibility**

```javascript
const form = document.querySelector('form');
const emailInput = document.getElementById('email');
const emailError = document.getElementById('email-error');

emailInput.addEventListener('blur', () => {
  if (!emailInput.validity.valid) {
    emailError.textContent = 'Please enter a valid email address.';
    emailInput.setAttribute('aria-invalid', 'true');
  } else {
    emailError.textContent = '';
    emailInput.removeAttribute('aria-invalid');
  }
});
```

### Workflow 2: Keyboard Navigation

**Step 1: Focus Management**

```javascript
// Modal with focus trap
class AccessibleModal {
  constructor(modalElement) {
    this.modal = modalElement;
    this.focusableElements = this.modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    this.firstFocusable = this.focusableElements[0];
    this.lastFocusable = this.focusableElements[this.focusableElements.length - 1];
  }

  open() {
    this.previouslyFocused = document.activeElement;
    this.modal.setAttribute('aria-hidden', 'false');
    this.modal.addEventListener('keydown', this.trapFocus.bind(this));
    this.firstFocusable.focus();
  }

  close() {
    this.modal.setAttribute('aria-hidden', 'true');
    this.modal.removeEventListener('keydown', this.trapFocus);
    this.previouslyFocused.focus();
  }

  trapFocus(e) {
    if (e.key === 'Tab') {
      if (e.shiftKey) {
        if (document.activeElement === this.firstFocusable) {
          e.preventDefault();
          this.lastFocusable.focus();
        }
      } else {
        if (document.activeElement === this.lastFocusable) {
          e.preventDefault();
          this.firstFocusable.focus();
        }
      }
    }
    if (e.key === 'Escape') {
      this.close();
    }
  }
}
```

**Step 2: Skip Links**

```html
<body>
  <a href="#main-content" class="skip-link">Skip to main content</a>
  <nav><!-- navigation --></nav>
  <main id="main-content" tabindex="-1">
    <!-- main content -->
  </main>
</body>

<style>
.skip-link {
  position: absolute;
  left: -9999px;
  z-index: 999;
  padding: 1em;
  background: #000;
  color: #fff;
}

.skip-link:focus {
  left: 50%;
  transform: translateX(-50%);
  top: 0;
}
</style>
```

### Workflow 3: ARIA Patterns

**Step 1: Accordion Component**

```html
<div class="accordion">
  <h3>
    <button
      aria-expanded="false"
      aria-controls="section1"
      id="accordion1"
    >
      <span>Section 1</span>
      <span aria-hidden="true">+</span>
    </button>
  </h3>
  <div id="section1" role="region" aria-labelledby="accordion1" hidden>
    <p>Section 1 content</p>
  </div>

  <h3>
    <button
      aria-expanded="false"
      aria-controls="section2"
      id="accordion2"
    >
      <span>Section 2</span>
      <span aria-hidden="true">+</span>
    </button>
  </h3>
  <div id="section2" role="region" aria-labelledby="accordion2" hidden>
    <p>Section 2 content</p>
  </div>
</div>

<script>
document.querySelectorAll('.accordion button').forEach(button => {
  button.addEventListener('click', () => {
    const expanded = button.getAttribute('aria-expanded') === 'true';
    const content = document.getElementById(button.getAttribute('aria-controls'));

    button.setAttribute('aria-expanded', !expanded);
    content.hidden = expanded;
    button.querySelector('[aria-hidden]').textContent = expanded ? '+' : '-';
  });
});
</script>
```

**Step 2: Live Regions for Dynamic Content**

```html
<div aria-live="polite" aria-atomic="true" class="sr-only">
  <!-- Announces updates to screen readers -->
</div>

<script>
function announceToScreenReader(message) {
  const liveRegion = document.querySelector('[aria-live]');
  liveRegion.textContent = message;
  setTimeout(() => {
    liveRegion.textContent = '';
  }, 1000);
}

// Usage
announceToScreenReader('Item added to cart');
</script>
```

### Workflow 4: Color Contrast and Visual Design

**Step 1: WCAG AA Contrast Requirements**

```css
/* ✅ GOOD: 4.5:1 contrast for normal text (WCAG AA) */
.text {
  color: #595959; /* on white background */
  background: #ffffff;
}

/* ✅ GOOD: 3:1 contrast for large text (18pt+ or 14pt+ bold) */
.large-text {
  font-size: 18pt;
  color: #767676;
  background: #ffffff;
}

/* ❌ BAD: 2.5:1 contrast (fails WCAG AA) */
.low-contrast {
  color: #a8a8a8;
  background: #ffffff;
}
```

**Step 2: Focus Indicators**

```css
/* ✅ GOOD: Visible focus indicator */
button:focus,
a:focus {
  outline: 3px solid #4A90E2;
  outline-offset: 2px;
}

/* Enhanced focus for keyboard users only */
button:focus-visible {
  outline: 3px solid #4A90E2;
  outline-offset: 2px;
}

/* ❌ BAD: Removing outline without replacement */
button:focus {
  outline: none; /* Don't do this */
}
```

**Step 3: Responsive Text and Zoom**

```css
/* ✅ GOOD: Relative units for zoom support */
body {
  font-size: 16px; /* Base size */
}

h1 {
  font-size: 2rem; /* 32px, scales with zoom */
}

/* ✅ GOOD: Allow text to reflow at 320px viewport */
@media (max-width: 320px) {
  .container {
    width: 100%;
    max-width: none;
  }
}
```

### Workflow 5: Automated Testing

**Step 1: axe-core Integration**

```bash
npm install --save-dev @axe-core/playwright
```

```javascript
// tests/accessibility.test.js
import { test, expect } from '@playwright/test';
import { injectAxe, checkA11y } from '@axe-core/playwright';

test('homepage should be accessible', async ({ page }) => {
  await page.goto('http://localhost:3000');
  await injectAxe(page);

  await checkA11y(page, null, {
    detailedReport: true,
    detailedReportOptions: {
      html: true,
    },
  });
});

test('forms should be accessible', async ({ page }) => {
  await page.goto('http://localhost:3000/form');
  await injectAxe(page);

  await checkA11y(page, '#form-container', {
    axeOptions: {
      runOnly: {
        type: 'tag',
        values: ['wcag2a', 'wcag2aa', 'wcag21aa'],
      },
    },
  });
});
```

**Step 2: Lighthouse CI**

```yaml
# .github/workflows/a11y.yml
name: Accessibility Audit
on: [pull_request]

jobs:
  lighthouse:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Lighthouse
        uses: treosh/lighthouse-ci-action@v9
        with:
          urls: |
            http://localhost:3000
          configPath: ./lighthouserc.json
```

```json
// lighthouserc.json
{
  "ci": {
    "assert": {
      "assertions": {
        "categories:accessibility": ["error", { "minScore": 0.9 }]
      }
    }
  }
}
```

## Best Practices

**1. Semantic HTML First**
```html
<!-- ✅ GOOD -->
<nav><ul><li><a href="/">Home</a></li></ul></nav>
<main><article><h1>Title</h1></article></main>
<footer>Footer content</footer>

<!-- ❌ BAD -->
<div class="nav"><div class="link">Home</div></div>
<div class="content"><div class="title">Title</div></div>
```

**2. Alt Text for Images**
```html
<!-- ✅ GOOD: Descriptive alt text -->
<img src="chart.png" alt="Bar chart showing sales increased 40% in Q4" />

<!-- ✅ GOOD: Decorative image -->
<img src="decorative.png" alt="" role="presentation" />

<!-- ❌ BAD: Missing alt or generic text -->
<img src="chart.png" alt="image" />
```

**3. Headings Hierarchy**
```html
<!-- ✅ GOOD: Logical heading order -->
<h1>Page Title</h1>
<h2>Section 1</h2>
<h3>Subsection 1.1</h3>
<h2>Section 2</h2>

<!-- ❌ BAD: Skipping levels -->
<h1>Page Title</h1>
<h4>Section</h4>
```

**4. Links vs Buttons**
```html
<!-- ✅ GOOD: Link for navigation -->
<a href="/page">Go to page</a>

<!-- ✅ GOOD: Button for actions -->
<button type="button" onclick="showModal()">Open Modal</button>

<!-- ❌ BAD: Link styled as button for action -->
<a href="#" onclick="showModal()">Open Modal</a>
```

**5. Tables for Tabular Data**
```html
<table>
  <caption>Monthly Sales Report</caption>
  <thead>
    <tr>
      <th scope="col">Month</th>
      <th scope="col">Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">January</th>
      <td>$10,000</td>
    </tr>
  </tbody>
</table>
```

## Quality Criteria

- ✅ Lighthouse accessibility score ≥90
- ✅ Zero critical axe-core violations
- ✅ All interactive elements keyboard accessible
- ✅ Color contrast ratio ≥4.5:1 (normal text)
- ✅ All images have alt text
- ✅ Forms have proper labels
- ✅ Passes screen reader testing (NVDA/VoiceOver)

## Testing Checklist

**Keyboard Navigation**:
- [ ] Tab through all interactive elements
- [ ] Shift+Tab works in reverse
- [ ] Enter/Space activates buttons
- [ ] Escape closes modals
- [ ] Arrow keys work in custom widgets

**Screen Reader**:
- [ ] All content is announced
- [ ] Landmarks are correctly identified
- [ ] Form errors are announced
- [ ] Dynamic content updates are announced
- [ ] Images have descriptive alt text

**Visual**:
- [ ] Text readable at 200% zoom
- [ ] Content reflows at 320px width
- [ ] Focus indicators visible
- [ ] No information conveyed by color alone

## Troubleshooting

**Issue**: Screen reader skipping content
**Solution**: Check if content is visually hidden incorrectly (use `.sr-only` class, not `display: none`)

**Issue**: Tab order incorrect
**Solution**: Avoid positive `tabindex` values, ensure DOM order matches visual order

**Issue**: Form errors not announced
**Solution**: Use `aria-live="polite"` or `role="alert"` for error messages

## Related Skills

- `react-specialist`: Accessible React components
- `testing-quality`: Automated accessibility testing
- `style-audit`: CSS code quality and a11y

## Tools

- axe DevTools: Browser extension for a11y audits
- Lighthouse: Performance and a11y audits
- NVDA: Free screen reader (Windows)
- VoiceOver: Built-in screen reader (macOS/iOS)
- WAVE: Web accessibility evaluation tool
- Color Contrast Analyzer: WCAG contrast checker

## MCP Tools

- `mcp__playwright__browser_snapshot` for visual regression testing
- `mcp__playwright__browser_evaluate` for running axe-core
- `mcp__memory-mcp__memory_store` for a11y patterns

## Success Metrics

- Lighthouse a11y score: ≥90
- axe-core violations: 0 critical, <5 moderate
- Keyboard accessibility: 100% of interactive elements
- Screen reader testing: 0 critical issues
- WCAG 2.1 AA compliance: 100%

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-11-02
