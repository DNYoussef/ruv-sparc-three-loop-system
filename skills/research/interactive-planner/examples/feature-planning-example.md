# Interactive Planning Example: E-Commerce Checkout Feature

This example demonstrates using the interactive-planner skill to gather comprehensive requirements for adding a checkout flow to an e-commerce application.

## Scenario

**User Request**: "I want to add a complete checkout flow to my e-commerce site with payment integration, shipping options, and order confirmation."

**Complexity**: Complex (10+ files, multiple integrations)

**Timeline**: 1-2 months

## Interactive Planning Process

### Batch 1: Project Scope & Core Functionality (4 Questions)

```yaml
questions:
  - question: "What is the primary goal of this checkout implementation?"
    header: "Goal"
    multiSelect: false
    options:
      - label: "New feature"
        description: "Add checkout flow to existing e-commerce site"
      - label: "Refactoring"
        description: "Improve existing checkout experience"
      - label: "Bug fix"
        description: "Fix critical issues in current checkout"
      - label: "Performance"
        description: "Optimize checkout conversion and speed"

    user_answer: "New feature"

  - question: "What payment methods should be supported? (Select all)"
    header: "Payment"
    multiSelect: true
    options:
      - label: "Credit/debit cards"
        description: "Visa, Mastercard, Amex via Stripe/Square"
      - label: "Digital wallets"
        description: "Apple Pay, Google Pay, PayPal"
      - label: "Buy now, pay later"
        description: "Afterpay, Klarna, Affirm"
      - label: "Cryptocurrency"
        description: "Bitcoin, Ethereum via Coinbase Commerce"

    user_answer: ["Credit/debit cards", "Digital wallets", "Buy now, pay later"]

  - question: "What shipping options are needed? (Select all)"
    header: "Shipping"
    multiSelect: true
    options:
      - label: "Standard shipping"
        description: "5-7 business days, lowest cost"
      - label: "Express shipping"
        description: "2-3 business days, medium cost"
      - label: "Overnight shipping"
        description: "Next business day, highest cost"
      - label: "In-store pickup"
        description: "Customer picks up at physical location"

    user_answer: ["Standard shipping", "Express shipping", "In-store pickup"]

  - question: "What order management features are required? (Select all)"
    header: "Order Mgmt"
    multiSelect: true
    options:
      - label: "Order tracking"
        description: "Real-time tracking with carrier integration"
      - label: "Order history"
        description: "Customer can view past orders"
      - label: "Order cancellation"
        description: "Customer can cancel within window"
      - label: "Returns/refunds"
        description: "Automated returns and refund processing"

    user_answer: ["Order tracking", "Order history", "Order cancellation"]
```

**Analysis After Batch 1**:
- Core functionality: Complete checkout flow with multiple payment methods
- Key features: Payment processing, shipping calculation, order management
- Integration needs: Payment gateway, shipping carriers, order tracking

### Batch 2: Technical Architecture (4 Questions)

```yaml
questions:
  - question: "What technology stack is your e-commerce site using?"
    header: "Stack"
    multiSelect: false
    options:
      - label: "React/Next.js"
        description: "React with Next.js for SSR/SSG"
      - label: "Vue/Nuxt"
        description: "Vue with Nuxt for SSR"
      - label: "Shopify"
        description: "Shopify platform with Liquid/React"
      - label: "WooCommerce"
        description: "WordPress with WooCommerce"

    user_answer: "React/Next.js"

  - question: "What payment gateway should we integrate?"
    header: "Gateway"
    multiSelect: false
    options:
      - label: "Stripe"
        description: "Stripe Checkout + Payment Intents API"
      - label: "Square"
        description: "Square Payments API"
      - label: "PayPal"
        description: "PayPal REST API"
      - label: "Braintree"
        description: "Braintree (owned by PayPal)"

    user_answer: "Stripe"

    # Follow-up based on Stripe selection
    follow_up:
      - question: "What Stripe integration approach?"
        options:
          - "Stripe Checkout (hosted)"
          - "Payment Element (embedded)"
          - "Custom integration"
        user_answer: "Payment Element (embedded)"

  - question: "What database are you using for orders/customers?"
    header: "Database"
    multiSelect: false
    options:
      - label: "PostgreSQL"
        description: "PostgreSQL with transactional integrity"
      - label: "MongoDB"
        description: "MongoDB for flexible schema"
      - label: "Firebase"
        description: "Firebase Firestore for real-time sync"
      - label: "Supabase"
        description: "Supabase (Postgres) with real-time features"

    user_answer: "PostgreSQL"

  - question: "What shipping carrier integrations? (Select all)"
    header: "Carriers"
    multiSelect: true
    options:
      - label: "USPS"
        description: "USPS with EasyPost or direct API"
      - label: "UPS"
        description: "UPS with real-time rates"
      - label: "FedEx"
        description: "FedEx with tracking integration"
      - label: "ShipStation"
        description: "ShipStation multi-carrier platform"

    user_answer: ["USPS", "UPS", "ShipStation"]
```

**Analysis After Batch 2**:
- Tech stack: React/Next.js frontend with PostgreSQL backend
- Payment: Stripe Payment Element (embedded)
- Shipping: Multi-carrier via ShipStation + direct USPS/UPS APIs
- Architecture: Full-stack Next.js with API routes for payment/shipping

### Batch 3: User Experience & Design (4 Questions)

```yaml
questions:
  - question: "What checkout flow type should we implement?"
    header: "Flow Type"
    multiSelect: false
    options:
      - label: "Single-page"
        description: "All steps on one page (best for mobile)"
      - label: "Multi-step"
        description: "Separate pages for shipping/payment/review"
      - label: "Accordion"
        description: "Expandable sections (Stripe-style)"
      - label: "Guest + account"
        description: "Allow guest checkout or account creation"

    user_answer: "Guest + account"

  - question: "What information should we collect? (Select all)"
    header: "User Info"
    multiSelect: true
    options:
      - label: "Email"
        description: "Required for order confirmation"
      - label: "Phone number"
        description: "For shipping notifications"
      - label: "Billing address"
        description: "Required for payment verification"
      - label: "Marketing opt-in"
        description: "Newsletter/promotional emails"

    user_answer: ["Email", "Phone number", "Billing address", "Marketing opt-in"]

  - question: "What validation and security features? (Select all)"
    header: "Security"
    multiSelect: true
    options:
      - label: "Address validation"
        description: "Google Maps API for address autocomplete"
      - label: "Fraud detection"
        description: "Stripe Radar for fraud prevention"
      - label: "3D Secure"
        description: "3DS2 for additional card verification"
      - label: "SSL/HTTPS"
        description: "Encrypted checkout pages"

    user_answer: ["Address validation", "Fraud detection", "3D Secure", "SSL/HTTPS"]

  - question: "What user notifications are needed? (Select all)"
    header: "Notifications"
    multiSelect: true
    options:
      - label: "Order confirmation email"
        description: "Immediate email after order placed"
      - label: "Shipping notifications"
        description: "Email/SMS when order ships"
      - label: "Delivery notifications"
        description: "Email/SMS when order delivered"
      - label: "SMS updates"
        description: "Text message notifications"

    user_answer: ["Order confirmation email", "Shipping notifications", "Delivery notifications"]
```

**Analysis After Batch 3**:
- UX: Guest checkout with optional account creation
- Data collection: Email, phone, billing address, marketing opt-in
- Security: Full security stack (address validation, fraud detection, 3DS, SSL)
- Notifications: Email-based (confirmation, shipping, delivery)

### Batch 4: Quality & Testing (4 Questions)

```yaml
questions:
  - question: "What testing coverage is required? (Select all)"
    header: "Testing"
    multiSelect: true
    options:
      - label: "Unit tests"
        description: "Jest for component/utility testing"
      - label: "Integration tests"
        description: "API endpoint testing with Supertest"
      - label: "E2E tests"
        description: "Playwright for full checkout flows"
      - label: "Payment testing"
        description: "Stripe test mode with test cards"

    user_answer: ["Unit tests", "Integration tests", "E2E tests", "Payment testing"]

  - question: "What quality level is expected?"
    header: "Quality"
    multiSelect: false
    options:
      - label: "Quick prototype"
        description: "Fast MVP with minimal quality checks"
      - label: "Production MVP"
        description: "Solid MVP with essential quality measures"
      - label: "Enterprise grade"
        description: "High quality with comprehensive testing"
      - label: "Research/experimental"
        description: "Exploratory with focus on learning"

    user_answer: "Enterprise grade"

  - question: "What performance requirements exist?"
    header: "Performance"
    multiSelect: false
    options:
      - label: "Best effort"
        description: "No specific targets"
      - label: "Good UX"
        description: "<2s page load, <500ms API responses"
      - label: "High performance"
        description: "<1s page load, <200ms API responses"
      - label: "Ultra-fast"
        description: "<500ms page load, <100ms API responses"

    user_answer: "High performance"

  - question: "What accessibility compliance is needed?"
    header: "A11y"
    multiSelect: false
    options:
      - label: "None"
        description: "No specific accessibility requirements"
      - label: "Basic"
        description: "Keyboard navigation and screen readers"
      - label: "WCAG AA"
        description: "WCAG 2.1 Level AA compliance"
      - label: "WCAG AAA"
        description: "WCAG 2.1 Level AAA compliance"

    user_answer: "WCAG AA"
```

**Analysis After Batch 4**:
- Testing: Comprehensive (unit, integration, E2E, payment testing)
- Quality: Enterprise-grade requirements
- Performance: High (<1s page load, <200ms API)
- Accessibility: WCAG 2.1 AA compliance

### Batch 5: Constraints & Context (4 Questions)

```yaml
questions:
  - question: "What is the timeline for this implementation?"
    header: "Timeline"
    multiSelect: false
    options:
      - label: "Urgent (today)"
        description: "Critical bug fix or emergency"
      - label: "This week"
        description: "High priority feature"
      - label: "This month"
        description: "Standard feature timeline"
      - label: "Flexible"
        description: "No hard deadline"

    user_answer: "This month"

  - question: "What is the team size working on this?"
    header: "Team Size"
    multiSelect: false
    options:
      - label: "Solo developer"
        description: "Just me working on this"
      - label: "Small team (2-3)"
        description: "2-3 developers collaborating"
      - label: "Medium team (4-8)"
        description: "4-8 person development team"
      - label: "Large team (9+)"
        description: "Large team with specializations"

    user_answer: "Small team (2-3)"

  - question: "What existing systems must we integrate with? (Select all)"
    header: "Integrations"
    multiSelect: true
    options:
      - label: "Inventory system"
        description: "Real-time stock checks"
      - label: "CRM"
        description: "Customer relationship management"
      - label: "Analytics"
        description: "Google Analytics, Segment, etc."
      - label: "Marketing automation"
        description: "Klaviyo, Mailchimp, etc."

    user_answer: ["Inventory system", "Analytics", "Marketing automation"]

  - question: "What compliance requirements apply? (Select all)"
    header: "Compliance"
    multiSelect: true
    options:
      - label: "PCI DSS"
        description: "Payment card industry security standards"
      - label: "GDPR"
        description: "EU data protection regulation"
      - label: "CCPA"
        description: "California consumer privacy act"
      - label: "SOC 2"
        description: "Service organization controls"

    user_answer: ["PCI DSS", "GDPR"]
```

**Analysis After Batch 5**:
- Timeline: This month (realistic for complex checkout)
- Team: Small team (2-3 developers)
- Integrations: Inventory, analytics, marketing automation
- Compliance: PCI DSS (payment security) + GDPR (EU privacy)

## Synthesized Requirements Specification

### Project Scope

**Purpose**: Add complete checkout flow to existing e-commerce site
**Project Type**: Web application (React/Next.js)
**Complexity**: Complex (10+ files, multiple integrations)
**Timeline**: This month (4 weeks)

### Technical Decisions

**Framework**: React/Next.js with TypeScript
**Database**: PostgreSQL (for transactional integrity)
**Payment Gateway**: Stripe Payment Element (embedded)
**Shipping Integration**: ShipStation + direct USPS/UPS APIs
**Backend**: Next.js API routes
**Deployment**: Vercel (Next.js native)

### Core Features

1. **Payment Processing**
   - Credit/debit cards (Visa, Mastercard, Amex)
   - Digital wallets (Apple Pay, Google Pay, PayPal)
   - Buy now, pay later (Afterpay, Klarna)
   - Stripe Payment Element integration
   - Fraud detection via Stripe Radar
   - 3D Secure (3DS2) for card verification

2. **Shipping Management**
   - Standard shipping (5-7 days)
   - Express shipping (2-3 days)
   - In-store pickup option
   - Real-time shipping rate calculation
   - Multi-carrier support (USPS, UPS, ShipStation)

3. **Order Management**
   - Order tracking with carrier integration
   - Order history for customers
   - Order cancellation (within window)
   - Email notifications (confirmation, shipping, delivery)

4. **User Experience**
   - Guest checkout + optional account creation
   - Address validation (Google Maps API)
   - Single-page checkout flow
   - Responsive design (mobile-optimized)
   - WCAG 2.1 AA accessibility compliance

### Quality Requirements

**Testing Coverage**:
- Unit tests: Jest for components/utilities (80%+ coverage)
- Integration tests: Supertest for API endpoints
- E2E tests: Playwright for full checkout flows
- Payment testing: Stripe test mode with test cards

**Performance Targets**:
- Page load: <1 second
- API response time: <200ms
- Checkout completion: <30 seconds

**Quality Level**: Enterprise-grade
- Comprehensive error handling
- Logging and monitoring (Sentry)
- Security scanning (Snyk)
- Code review required
- Automated CI/CD pipeline

### Constraints

**Timeline**: 4 weeks (this month)
**Team Size**: 2-3 developers
**Budget**: Moderate (standard SaaS costs for Stripe, ShipStation, etc.)

**Compliance**:
- PCI DSS: Payment security (Stripe handles most)
- GDPR: EU data protection (cookie consent, data deletion)

**Integrations**:
- Inventory system (stock checks before checkout)
- Analytics (Google Analytics, event tracking)
- Marketing automation (Klaviyo for abandoned cart emails)

### Implementation Plan (4 Weeks)

**Week 1: Foundation**
- Database schema (orders, payments, shipping)
- Stripe Payment Element integration
- Basic checkout UI components

**Week 2: Core Features**
- Shipping rate calculation
- ShipStation integration
- Order management APIs
- Guest checkout flow

**Week 3: Polish & Testing**
- Address validation
- Fraud detection
- Email notifications
- E2E test suite

**Week 4: Launch Preparation**
- Performance optimization
- Security audit
- WCAG AA compliance check
- Production deployment

### Missing Information (Follow-up Required)

1. What should happen with abandoned carts? (Save for 24h? Email reminders?)
2. Do you need multi-currency support? (USD only or international?)
3. What tax calculation service? (Avalara, TaxJar, or manual?)
4. Should we support discount codes/coupons?
5. Do you need gift wrapping or order notes features?

### Confidence Level: HIGH

**Total Questions**: 20 (5 batches of 4)
**Complete Answers**: 20 (100%)
**Categories Covered**: All 5 (core, technical, UX, quality, constraints)

---

**Generated**: 2025-01-15T10:30:00Z
**Tool Used**: interactive-planner skill v1.0.0
