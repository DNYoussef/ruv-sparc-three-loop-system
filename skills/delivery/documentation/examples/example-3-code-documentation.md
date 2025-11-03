# Example 3: Code Documentation Generation

## Overview
This example demonstrates how the documentation skill automatically generates comprehensive inline code documentation including JSDoc comments, TypeDoc integration, function signatures, type definitions, and usage examples directly from source code.

## Scenario
**Context**: You have a complex TypeScript library for data validation with 50+ functions but minimal inline documentation. New team members struggle to understand the API.

**Goal**: Generate complete JSDoc comments for all public APIs, create TypeDoc-powered documentation site, and add inline usage examples.

**Starting Point**:
- `src/validators/` - 8 validator modules
- `src/types/` - Type definitions
- Minimal comments (5% coverage)

## Walkthrough

### Step 1: Invoke Documentation Skill

```bash
# User request
"Generate comprehensive code documentation for our validation library"

# Auto-trigger detection
Keywords detected: "code documentation", "generate"
Skill triggered: documentation
Agent spawned: technical-writing-agent
```

### Step 2: Source Code Analysis

The skill analyzes your existing code:

```typescript
// Example: src/validators/string.validator.ts (BEFORE)
export class StringValidator {
  validate(value: unknown, options?: ValidationOptions): ValidationResult {
    if (typeof value !== 'string') {
      return { valid: false, errors: ['Must be a string'] };
    }

    if (options?.minLength && value.length < options.minLength) {
      return { valid: false, errors: [`Minimum length is ${options.minLength}`] };
    }

    if (options?.maxLength && value.length > options.maxLength) {
      return { valid: false, errors: [`Maximum length is ${options.maxLength}`] };
    }

    if (options?.pattern && !options.pattern.test(value)) {
      return { valid: false, errors: ['Does not match pattern'] };
    }

    return { valid: true };
  }

  sanitize(value: string, options?: SanitizeOptions): string {
    let result = value;
    if (options?.trim) result = result.trim();
    if (options?.lowercase) result = result.toLowerCase();
    if (options?.uppercase) result = result.toUpperCase();
    return result;
  }
}
```

**Skill Actions**:
1. Identifies public methods and their parameters
2. Infers parameter types and return types
3. Detects validation logic and constraints
4. Analyzes error conditions
5. Finds related types/interfaces

### Step 3: JSDoc Comment Generation

```typescript
// Generated: src/validators/string.validator.ts (AFTER)
/**
 * String validation and sanitization utilities.
 *
 * Provides comprehensive string validation with support for length constraints,
 * pattern matching, and common sanitization operations.
 *
 * @example
 * ```typescript
 * const validator = new StringValidator();
 *
 * // Validate with constraints
 * const result = validator.validate('hello', {
 *   minLength: 3,
 *   maxLength: 10,
 *   pattern: /^[a-z]+$/
 * });
 *
 * if (result.valid) {
 *   console.log('Valid string');
 * } else {
 *   console.error('Validation errors:', result.errors);
 * }
 * ```
 *
 * @public
 * @since 1.0.0
 */
export class StringValidator {
  /**
   * Validates a value against string constraints.
   *
   * Performs type checking and optional validation against length constraints
   * and regular expression patterns. Returns a detailed validation result
   * with all validation errors.
   *
   * @param value - The value to validate (type checked at runtime)
   * @param options - Optional validation constraints
   * @param options.minLength - Minimum allowed string length (inclusive)
   * @param options.maxLength - Maximum allowed string length (inclusive)
   * @param options.pattern - Regular expression pattern to match
   *
   * @returns Validation result with success status and error messages
   *
   * @example
   * ```typescript
   * // Basic validation
   * validator.validate('test'); // { valid: true }
   *
   * // With length constraints
   * validator.validate('hi', { minLength: 5 });
   * // { valid: false, errors: ['Minimum length is 5'] }
   *
   * // With pattern matching
   * validator.validate('test123', { pattern: /^[a-z]+$/ });
   * // { valid: false, errors: ['Does not match pattern'] }
   *
   * // Multiple constraints
   * validator.validate('hello', {
   *   minLength: 3,
   *   maxLength: 10,
   *   pattern: /^[a-z]+$/
   * }); // { valid: true }
   * ```
   *
   * @throws {TypeError} Never throws - all errors returned in result object
   *
   * @see {@link sanitize} for string sanitization
   * @see {@link ValidationOptions} for all available options
   *
   * @public
   */
  validate(value: unknown, options?: ValidationOptions): ValidationResult {
    if (typeof value !== 'string') {
      return { valid: false, errors: ['Must be a string'] };
    }

    if (options?.minLength && value.length < options.minLength) {
      return { valid: false, errors: [`Minimum length is ${options.minLength}`] };
    }

    if (options?.maxLength && value.length > options.maxLength) {
      return { valid: false, errors: [`Maximum length is ${options.maxLength}`] };
    }

    if (options?.pattern && !options.pattern.test(value)) {
      return { valid: false, errors: ['Does not match pattern'] };
    }

    return { valid: true };
  }

  /**
   * Sanitizes a string by applying common transformations.
   *
   * Applies one or more sanitization operations to clean and normalize
   * string input. Operations are applied in order: trim, lowercase, uppercase.
   *
   * @param value - The string to sanitize
   * @param options - Sanitization operations to apply
   * @param options.trim - Remove leading and trailing whitespace
   * @param options.lowercase - Convert to lowercase
   * @param options.uppercase - Convert to uppercase (takes precedence over lowercase)
   *
   * @returns Sanitized string
   *
   * @example
   * ```typescript
   * // Trim whitespace
   * validator.sanitize('  hello  ', { trim: true }); // 'hello'
   *
   * // Convert case
   * validator.sanitize('Hello World', { lowercase: true }); // 'hello world'
   * validator.sanitize('Hello World', { uppercase: true }); // 'HELLO WORLD'
   *
   * // Combine operations
   * validator.sanitize('  Hello World  ', {
   *   trim: true,
   *   lowercase: true
   * }); // 'hello world'
   * ```
   *
   * @remarks
   * If both `lowercase` and `uppercase` are true, `uppercase` takes precedence.
   *
   * @public
   */
  sanitize(value: string, options?: SanitizeOptions): string {
    let result = value;
    if (options?.trim) result = result.trim();
    if (options?.lowercase) result = result.toLowerCase();
    if (options?.uppercase) result = result.toUpperCase();
    return result;
  }
}

/**
 * Options for string validation.
 *
 * @public
 */
export interface ValidationOptions {
  /**
   * Minimum allowed string length (inclusive).
   * @example 5 - Requires at least 5 characters
   */
  minLength?: number;

  /**
   * Maximum allowed string length (inclusive).
   * @example 100 - Allows up to 100 characters
   */
  maxLength?: number;

  /**
   * Regular expression pattern to match.
   * @example /^[a-zA-Z0-9]+$/ - Alphanumeric only
   */
  pattern?: RegExp;
}

/**
 * Options for string sanitization.
 *
 * @public
 */
export interface SanitizeOptions {
  /**
   * Remove leading and trailing whitespace.
   * @defaultValue false
   */
  trim?: boolean;

  /**
   * Convert string to lowercase.
   * @defaultValue false
   */
  lowercase?: boolean;

  /**
   * Convert string to uppercase.
   * @defaultValue false
   * @remarks Takes precedence over lowercase if both are true
   */
  uppercase?: boolean;
}

/**
 * Result of a validation operation.
 *
 * @public
 */
export interface ValidationResult {
  /**
   * Whether the validation passed.
   */
  valid: boolean;

  /**
   * Array of validation error messages.
   * Empty if validation passed.
   */
  errors?: string[];
}
```

### Step 4: TypeDoc Configuration & Site Generation

```typescript
// Generated: typedoc.json
{
  "entryPoints": ["src/index.ts"],
  "out": "docs",
  "theme": "default",
  "excludePrivate": true,
  "excludeProtected": true,
  "excludeExternals": true,
  "readme": "README.md",
  "name": "Data Validation Library",
  "includeVersion": true,
  "categorizeByGroup": true,
  "categoryOrder": [
    "Validators",
    "Types",
    "Utilities",
    "*"
  ],
  "navigation": {
    "includeCategories": true,
    "includeGroups": true
  },
  "plugin": [
    "typedoc-plugin-markdown",
    "typedoc-plugin-merge-modules"
  ]
}
```

```json
// Generated: package.json (scripts added)
{
  "scripts": {
    "docs:generate": "typedoc",
    "docs:serve": "npx serve docs",
    "docs:watch": "typedoc --watch",
    "docs:deploy": "npm run docs:generate && gh-pages -d docs"
  },
  "devDependencies": {
    "typedoc": "^0.25.0",
    "typedoc-plugin-markdown": "^3.17.0",
    "gh-pages": "^6.0.0"
  }
}
```

### Step 5: Module-Level Documentation

```typescript
// Generated: src/validators/index.ts
/**
 * Comprehensive data validation library.
 *
 * @remarks
 * This library provides a complete set of validators for common data types
 * including strings, numbers, arrays, objects, and custom types. All validators
 * follow a consistent API and return detailed validation results.
 *
 * @packageDocumentation
 *
 * @example
 * ```typescript
 * import { StringValidator, NumberValidator } from '@yourorg/validators';
 *
 * // String validation
 * const strValidator = new StringValidator();
 * strValidator.validate('hello', { minLength: 3 }); // { valid: true }
 *
 * // Number validation
 * const numValidator = new NumberValidator();
 * numValidator.validate(42, { min: 0, max: 100 }); // { valid: true }
 * ```
 */

/**
 * @module Validators
 * String validation utilities
 */
export { StringValidator } from './string.validator';

/**
 * @module Validators
 * Number validation utilities
 */
export { NumberValidator } from './number.validator';

/**
 * @module Validators
 * Array validation utilities
 */
export { ArrayValidator } from './array.validator';

/**
 * @module Types
 * Validation type definitions
 */
export type {
  ValidationOptions,
  ValidationResult,
  SanitizeOptions
} from './types';
```

### Step 6: Usage Examples Generation

```typescript
// Generated: examples/string-validation.example.ts
/**
 * String Validation Examples
 *
 * Demonstrates common string validation patterns and best practices.
 */

import { StringValidator } from '../src/validators/string.validator';

// Initialize validator
const validator = new StringValidator();

// Example 1: Basic type validation
console.log('Example 1: Type Validation');
console.log(validator.validate('hello'));
// Output: { valid: true }

console.log(validator.validate(123));
// Output: { valid: false, errors: ['Must be a string'] }

// Example 2: Length constraints
console.log('\nExample 2: Length Constraints');
console.log(validator.validate('hi', { minLength: 5 }));
// Output: { valid: false, errors: ['Minimum length is 5'] }

console.log(validator.validate('hello', { minLength: 3, maxLength: 10 }));
// Output: { valid: true }

// Example 3: Pattern matching
console.log('\nExample 3: Pattern Matching');
const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
console.log(validator.validate('user@example.com', { pattern: emailPattern }));
// Output: { valid: true }

console.log(validator.validate('invalid-email', { pattern: emailPattern }));
// Output: { valid: false, errors: ['Does not match pattern'] }

// Example 4: Sanitization
console.log('\nExample 4: Sanitization');
console.log(validator.sanitize('  Hello World  ', { trim: true }));
// Output: 'Hello World'

console.log(validator.sanitize('Hello World', { lowercase: true }));
// Output: 'hello world'

// Example 5: Combining validation and sanitization
console.log('\nExample 5: Combined Operations');
function processInput(input: unknown): string | null {
  const validationResult = validator.validate(input, {
    minLength: 3,
    maxLength: 50,
    pattern: /^[a-zA-Z0-9\s]+$/
  });

  if (!validationResult.valid) {
    console.error('Validation failed:', validationResult.errors);
    return null;
  }

  return validator.sanitize(input as string, { trim: true, lowercase: true });
}

console.log(processInput('  Hello World  '));
// Output: 'hello world'

console.log(processInput('ab')); // Too short
// Output: null (with error logged)
```

### Step 7: Interactive Documentation Deployment

```yaml
# Generated: .github/workflows/docs-deploy.yml
name: Deploy Documentation

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Generate TypeDoc documentation
        run: npm run docs:generate

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
          cname: validators.yourorg.dev
```

## Outcomes

### Files Created/Updated
```
project/
├── src/
│   ├── validators/
│   │   ├── string.validator.ts     # ✨ JSDoc added (200+ lines comments)
│   │   ├── number.validator.ts     # ✨ JSDoc added
│   │   ├── array.validator.ts      # ✨ JSDoc added
│   │   └── index.ts                # ✨ Module docs added
│   └── types/
│       └── index.ts                # ✨ Type docs added
├── examples/
│   ├── string-validation.example.ts # ✨ Generated
│   ├── number-validation.example.ts # ✨ Generated
│   └── README.md                    # ✨ Generated
├── docs/                            # ✨ TypeDoc output (auto-generated)
│   ├── index.html
│   ├── modules/
│   └── classes/
├── typedoc.json                     # ✨ Generated
├── .github/
│   └── workflows/
│       └── docs-deploy.yml          # ✨ Generated
└── package.json                     # ✨ Updated (scripts added)
```

### Quality Metrics
- **Documentation Coverage**: 5% → 95% (all public APIs documented)
- **JSDoc Completeness**: 100% (params, returns, examples, remarks)
- **Example Coverage**: 3+ examples per major function
- **TypeDoc Site**: Fully generated with search, navigation
- **Time Saved**: 8-12 hours manual documentation writing

### Business Value
1. **Developer Onboarding**: 75% faster (clear API docs + examples)
2. **API Adoption**: 3x increase in library usage
3. **Support Reduction**: 50% fewer "how to use" questions
4. **Code Quality**: Better maintained (docs enforce good practices)

## Code Generated

### Full Documentation Structure

```
Documentation Assets Created:
├── Inline JSDoc (800+ lines)
│   ├── Class descriptions
│   ├── Method documentation
│   ├── Parameter details
│   ├── Return type documentation
│   ├── Usage examples
│   ├── Cross-references
│   └── Remarks and warnings
├── TypeDoc Site (auto-generated)
│   ├── API reference pages
│   ├── Search functionality
│   ├── Type definitions
│   └── Module hierarchy
├── Usage Examples (6 files)
│   ├── Basic usage
│   ├── Advanced patterns
│   ├── Error handling
│   └── Best practices
└── CI/CD Integration
    └── Auto-deploy to GitHub Pages
```

## Tips & Best Practices

### 1. Write Code-First, Document-After
```typescript
// ✅ GOOD: Write clear code, let skill generate docs
export function validateEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// Skill generates:
/**
 * Validates an email address using RFC 5322 pattern.
 * @param email - Email address to validate
 * @returns True if email is valid format
 * @example validator.validateEmail('user@example.com') // true
 */
```

### 2. Use TypeScript for Better Docs
```typescript
// TypeScript types auto-generate documentation
type ValidationLevel = 'strict' | 'loose' | 'permissive';

interface ValidatorConfig {
  level: ValidationLevel;
  throwOnError?: boolean;
}

// Skill auto-documents all type properties
```

### 3. Enable Documentation Linting
```json
// package.json
{
  "scripts": {
    "docs:lint": "typedoc --validation.invalidLink --validation.notExported"
  }
}
```

### 4. Keep Examples Runnable
```typescript
// examples/ files should be executable
// Skill auto-generates:
// - npm run examples (runs all examples)
// - npm run example:string (runs specific example)
```

### 5. Version Documentation
```bash
# Generate versioned docs
npm run docs:generate -- --out docs/v2
npm run docs:generate -- --out docs/v1

# Skill maintains version selector in docs
```

## Advanced Usage

### Custom JSDoc Tags
```typescript
// Skill recognizes custom tags
/**
 * @complexity O(n)
 * @performance Optimized for large datasets
 * @stability Stable since v1.0.0
 */
```

### Multi-Package Documentation
```bash
# Monorepo documentation
npx claude-flow sparc run documentation \
  "Generate docs for all packages in monorepo"

# Creates unified documentation site
```

### Integration with IDE
```json
// .vscode/settings.json (auto-generated)
{
  "typescript.suggest.completeFunctionCalls": true,
  "javascript.suggest.names": true,
  "editor.quickSuggestions": {
    "comments": true
  }
}
```

## Troubleshooting

### Issue: TypeDoc Build Fails
**Solution**: Check TypeScript compilation errors first
```bash
npm run build
npm run docs:generate
```

### Issue: Missing Examples in JSDoc
**Solution**: Add `@example` tags manually for complex cases

### Issue: Outdated Documentation
**Solution**: Enable git hooks to regenerate on code changes
```bash
# .git/hooks/pre-commit
npx typedoc --validation.invalidLink
```

### Issue: Large Documentation Size
**Solution**: Enable incremental TypeDoc builds
```json
{
  "incremental": true,
  "cacheDir": ".typedoc-cache"
}
```

## Next Steps

1. **Deploy Documentation**: Run `npm run docs:deploy` to publish to GitHub Pages
2. **Integrate with CI**: Enable automatic documentation updates on every commit
3. **Add Search**: Configure Algolia DocSearch for your documentation site
4. **Monitor Usage**: Track documentation views with Google Analytics
5. **Iterate**: Review developer feedback and improve examples

## Resources

- [TypeDoc Documentation](https://typedoc.org)
- [JSDoc Reference](https://jsdoc.app)
- [TSDoc Standard](https://tsdoc.org)
- [Documentation Best Practices](https://documentation.divio.com)

---

**Documentation generated with ❤️ by the Claude Flow documentation skill**
