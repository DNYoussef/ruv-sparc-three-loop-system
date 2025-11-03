# Example 2: Template Usage for Rapid Application Deployment

## Scenario

Your startup needs to rapidly prototype and deploy multiple SaaS applications for customer demos. Instead of building each application from scratch, you'll leverage Flow Nexus templates to deploy pre-configured applications in minutes. You'll:

- Browse available templates in the app store
- Deploy a full-stack e-commerce application
- Customize the deployment with your branding and configuration
- Set up authentication and payment processing
- Monitor deployment status and costs

## Prerequisites

- Flow Nexus account with Pro tier (for template access)
- Active session with authentication
- Basic understanding of environment variables
- Payment method configured (for premium templates)

## Walkthrough

### Step 1: Authenticate and Check Tier

```bash
# Login to Flow Nexus
mcp__flow-nexus__user_login {
  email: "founder@startup.com",
  password: "secure_password"
}

# Check user profile and tier
mcp__flow-nexus__user_profile { user_id: "usr_abc123" }
```

**Expected Output:**
```json
{
  "user": {
    "id": "usr_abc123",
    "email": "founder@startup.com",
    "tier": "pro",
    "credits": 2500,
    "created_at": "2025-10-15T08:00:00Z",
    "verified": true
  },
  "subscription": {
    "plan": "pro",
    "billing_cycle": "monthly",
    "next_billing_date": "2025-12-01T00:00:00Z"
  }
}
```

### Step 2: Browse Available Templates

```bash
# List all available templates
mcp__flow-nexus__template_list {
  category: "fullstack",
  featured: true,
  limit: 20
}
```

**Expected Output:**
```json
{
  "templates": [
    {
      "template_id": "tmpl_ecommerce_001",
      "name": "E-commerce Starter Kit",
      "description": "Full-stack e-commerce with Next.js, Stripe, and PostgreSQL",
      "category": "fullstack",
      "featured": true,
      "tier": "pro",
      "cost": 50,
      "rating": 4.8,
      "deployments": 1247,
      "stack": ["nextjs", "react", "postgresql", "stripe", "tailwind"],
      "features": [
        "User authentication (NextAuth.js)",
        "Product catalog with search",
        "Shopping cart and checkout",
        "Stripe payment integration",
        "Admin dashboard",
        "Order management",
        "Responsive design"
      ]
    },
    {
      "template_id": "tmpl_saas_dashboard_002",
      "name": "SaaS Dashboard Template",
      "description": "Multi-tenant SaaS platform with billing and analytics",
      "category": "fullstack",
      "featured": true,
      "tier": "pro",
      "cost": 75,
      "rating": 4.9,
      "deployments": 892,
      "stack": ["react", "express", "mongodb", "stripe", "recharts"],
      "features": [
        "Multi-tenant architecture",
        "Subscription billing",
        "Analytics dashboard",
        "Team management",
        "API key management",
        "Usage tracking"
      ]
    },
    {
      "template_id": "tmpl_blog_cms_003",
      "name": "Headless Blog CMS",
      "description": "SEO-optimized blog with markdown editor and API",
      "category": "cms",
      "featured": true,
      "tier": "free",
      "cost": 0,
      "rating": 4.6,
      "deployments": 3421,
      "stack": ["nextjs", "mdx", "contentlayer", "vercel"],
      "features": [
        "Markdown/MDX support",
        "SEO optimization",
        "RSS feed",
        "Sitemap generation",
        "Code syntax highlighting",
        "Dark mode"
      ]
    }
  ],
  "total": 47,
  "page": 1,
  "per_page": 20
}
```

### Step 3: Get Detailed Template Information

```bash
# Get full details for e-commerce template
mcp__flow-nexus__template_get {
  template_id: "tmpl_ecommerce_001"
}
```

**Expected Output:**
```json
{
  "template": {
    "id": "tmpl_ecommerce_001",
    "name": "E-commerce Starter Kit",
    "description": "Production-ready e-commerce platform with all essential features",
    "version": "2.4.0",
    "author": "Flow Nexus Team",
    "license": "MIT",
    "documentation": "https://docs.flow-nexus.io/templates/ecommerce",
    "demo_url": "https://demo-ecommerce.flow-nexus.io",
    "source_repo": "https://github.com/flow-nexus/ecommerce-starter",
    "stack": {
      "frontend": "Next.js 14 (App Router)",
      "backend": "Next.js API Routes",
      "database": "PostgreSQL 15",
      "auth": "NextAuth.js",
      "payments": "Stripe",
      "email": "Resend",
      "storage": "AWS S3",
      "search": "Algolia",
      "styling": "Tailwind CSS + shadcn/ui"
    },
    "required_variables": [
      {
        "name": "DATABASE_URL",
        "description": "PostgreSQL connection string",
        "example": "postgresql://user:pass@host:5432/db",
        "required": true
      },
      {
        "name": "NEXTAUTH_SECRET",
        "description": "Secret for NextAuth.js session encryption",
        "example": "openssl rand -base64 32",
        "required": true
      },
      {
        "name": "STRIPE_SECRET_KEY",
        "description": "Stripe secret API key",
        "example": "sk_test_...",
        "required": true
      },
      {
        "name": "STRIPE_WEBHOOK_SECRET",
        "description": "Stripe webhook signing secret",
        "example": "whsec_...",
        "required": true
      },
      {
        "name": "NEXT_PUBLIC_APP_URL",
        "description": "Public URL of your application",
        "example": "https://mystore.com",
        "required": true
      }
    ],
    "optional_variables": [
      {
        "name": "RESEND_API_KEY",
        "description": "Resend API key for transactional emails",
        "default": null
      },
      {
        "name": "AWS_S3_BUCKET",
        "description": "S3 bucket for product images",
        "default": null
      },
      {
        "name": "ALGOLIA_APP_ID",
        "description": "Algolia application ID for search",
        "default": null
      }
    ],
    "deployment_time": "3-5 minutes",
    "cost": 50,
    "estimated_monthly_cost": "$25-100 (depending on traffic)"
  }
}
```

### Step 4: Deploy E-commerce Template

```bash
# Deploy template with custom configuration
mcp__flow-nexus__template_deploy {
  template_id: "tmpl_ecommerce_001",
  deployment_name: "acme-fashion-store",
  variables: {
    "DATABASE_URL": "postgresql://admin:pass123@db.flow-nexus.io:5432/acme_store",
    "NEXTAUTH_SECRET": "d4f7a9b2c5e8f1a3d6b9c2e5f8a1d4b7c0e3f6a9",
    "NEXTAUTH_URL": "https://acme-fashion.flow-nexus.io",
    "STRIPE_SECRET_KEY": "sk_example_ecommerce_secret_key",
    "STRIPE_PUBLISHABLE_KEY": "pk_example_ecommerce_public_key",
    "STRIPE_WEBHOOK_SECRET": "whsec_example_signing_secret",
    "NEXT_PUBLIC_APP_URL": "https://acme-fashion.flow-nexus.io",
    "NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY": "pk_example_ecommerce_public_key",
    "RESEND_API_KEY": "re_example_resend_key",
    "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID_EXAMPLE",
    "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY_EXAMPLE",
    "AWS_S3_BUCKET": "acme-fashion-products",
    "AWS_REGION": "us-east-1"
  },
  env_vars: {
    "NODE_ENV": "production",
    "APP_NAME": "ACME Fashion Store",
    "SUPPORT_EMAIL": "support@acmefashion.com",
    "DEFAULT_CURRENCY": "USD",
    "SHIPPING_COUNTRIES": "US,CA,GB,AU",
    "TAX_RATE": "0.08",
    "FREE_SHIPPING_THRESHOLD": "50"
  }
}
```

**Expected Output:**
```json
{
  "deployment": {
    "deployment_id": "dep_acme_001",
    "name": "acme-fashion-store",
    "template_id": "tmpl_ecommerce_001",
    "status": "deploying",
    "progress": 15,
    "stages": [
      {
        "name": "Infrastructure Provisioning",
        "status": "completed",
        "duration": 45
      },
      {
        "name": "Database Setup",
        "status": "in_progress",
        "progress": 60
      },
      {
        "name": "Application Build",
        "status": "pending",
        "progress": 0
      },
      {
        "name": "Dependency Installation",
        "status": "pending",
        "progress": 0
      },
      {
        "name": "Environment Configuration",
        "status": "pending",
        "progress": 0
      },
      {
        "name": "Health Check",
        "status": "pending",
        "progress": 0
      }
    ],
    "started_at": "2025-11-02T11:00:00Z",
    "estimated_completion": "2025-11-02T11:04:30Z",
    "cost_charged": 50,
    "remaining_credits": 2450
  }
}
```

### Step 5: Monitor Deployment Progress

```bash
# Check deployment status every 30 seconds
mcp__flow-nexus__workflow_status {
  workflow_id: "dep_acme_001",
  include_metrics: true
}
```

**Expected Output (After 3 minutes):**
```json
{
  "deployment": {
    "deployment_id": "dep_acme_001",
    "name": "acme-fashion-store",
    "status": "completed",
    "progress": 100,
    "stages": [
      {
        "name": "Infrastructure Provisioning",
        "status": "completed",
        "duration": 45
      },
      {
        "name": "Database Setup",
        "status": "completed",
        "duration": 62
      },
      {
        "name": "Application Build",
        "status": "completed",
        "duration": 87
      },
      {
        "name": "Dependency Installation",
        "status": "completed",
        "duration": 34
      },
      {
        "name": "Environment Configuration",
        "status": "completed",
        "duration": 12
      },
      {
        "name": "Health Check",
        "status": "completed",
        "duration": 8
      }
    ],
    "completed_at": "2025-11-02T11:04:08Z",
    "total_duration": 248,
    "url": "https://acme-fashion.flow-nexus.io",
    "admin_url": "https://acme-fashion.flow-nexus.io/admin",
    "api_url": "https://acme-fashion.flow-nexus.io/api",
    "database_url": "postgresql://admin:pass123@db.flow-nexus.io:5432/acme_store",
    "credentials": {
      "admin_email": "admin@acmefashion.com",
      "admin_password": "temp_password_change_me"
    }
  },
  "metrics": {
    "build_time": "87s",
    "bundle_size": "3.2 MB",
    "pages_generated": 47,
    "api_routes": 23,
    "database_migrations": 18
  }
}
```

### Step 6: Access Deployed Application

```bash
# Get application details
mcp__flow-nexus__app_get { app_id: "dep_acme_001" }
```

**Expected Output:**
```json
{
  "app": {
    "id": "dep_acme_001",
    "name": "ACME Fashion Store",
    "url": "https://acme-fashion.flow-nexus.io",
    "status": "running",
    "version": "1.0.0",
    "template": "E-commerce Starter Kit v2.4.0",
    "deployed_at": "2025-11-02T11:04:08Z",
    "endpoints": {
      "frontend": "https://acme-fashion.flow-nexus.io",
      "admin": "https://acme-fashion.flow-nexus.io/admin",
      "api": "https://acme-fashion.flow-nexus.io/api",
      "webhooks": "https://acme-fashion.flow-nexus.io/api/webhooks"
    },
    "health": {
      "status": "healthy",
      "uptime": "00:15:42",
      "response_time": "124ms",
      "last_checked": "2025-11-02T11:19:50Z"
    },
    "resources": {
      "cpu": "1 vCPU",
      "memory": "2 GB",
      "storage": "20 GB SSD",
      "bandwidth": "Unlimited"
    }
  }
}
```

### Step 7: Configure Stripe Webhooks

```bash
# Update Stripe webhook configuration
mcp__flow-nexus__sandbox_configure {
  sandbox_id: "dep_acme_001",
  run_commands: [
    "stripe listen --forward-to https://acme-fashion.flow-nexus.io/api/webhooks/stripe",
    "stripe trigger payment_intent.succeeded"
  ]
}
```

### Step 8: Monitor Application Analytics

```bash
# Get analytics for the deployed app
mcp__flow-nexus__app_analytics {
  app_id: "dep_acme_001",
  timeframe: "7d"
}
```

**Expected Output:**
```json
{
  "analytics": {
    "period": "Last 7 days",
    "traffic": {
      "total_visits": 3247,
      "unique_visitors": 1892,
      "page_views": 14523,
      "avg_session_duration": "4m 32s",
      "bounce_rate": "32.4%"
    },
    "ecommerce": {
      "total_orders": 87,
      "revenue": "$4,325.50",
      "avg_order_value": "$49.72",
      "conversion_rate": "2.68%",
      "cart_abandonment_rate": "64.3%"
    },
    "performance": {
      "avg_response_time": "187ms",
      "p95_response_time": "342ms",
      "error_rate": "0.12%",
      "uptime": "99.97%"
    },
    "top_pages": [
      { "path": "/", "views": 3247, "avg_time": "1m 23s" },
      { "path": "/products", "views": 2891, "avg_time": "3m 12s" },
      { "path": "/products/summer-collection", "views": 1567, "avg_time": "2m 45s" },
      { "path": "/cart", "views": 892, "avg_time": "1m 56s" },
      { "path": "/checkout", "views": 234, "avg_time": "3m 18s" }
    ]
  }
}
```

### Step 9: Update Application Configuration

```bash
# Update environment variables
mcp__flow-nexus__app_update {
  app_id: "dep_acme_001",
  updates: {
    "env_vars": {
      "FREE_SHIPPING_THRESHOLD": "75",
      "SALE_ACTIVE": "true",
      "SALE_DISCOUNT": "0.20",
      "FEATURED_COLLECTION": "summer-2025"
    },
    "settings": {
      "maintenance_mode": false,
      "rate_limit": "100/minute",
      "cache_ttl": 3600
    }
  }
}
```

### Step 10: Scale Application

```bash
# Scale resources for Black Friday sale
mcp__flow-nexus__sandbox_configure {
  sandbox_id: "dep_acme_001",
  env_vars: {
    "SCALE_MODE": "auto",
    "MIN_INSTANCES": "3",
    "MAX_INSTANCES": "10",
    "TARGET_CPU": "70"
  }
}
```

## Outcomes

### What You Achieved

1. **Rapid Deployment**: Full-stack e-commerce app deployed in <5 minutes
2. **Production-Ready**: Complete with authentication, payments, database, and admin dashboard
3. **Customization**: Configured with your branding, environment variables, and business logic
4. **Monitoring**: Set up analytics and health checks for production monitoring
5. **Scalability**: Configured auto-scaling for traffic spikes
6. **Cost Efficiency**: Used pre-built template (50 credits) vs building from scratch (500+ hours)

### Metrics

- **Deployment Time**: 4 minutes 8 seconds
- **Template Cost**: 50 rUv credits ($5)
- **Time Saved**: ~80 hours of development
- **Features Included**: 47 pages, 23 API routes, 18 database tables
- **Production-Ready**: Immediately accessible at custom URL

## Tips and Best Practices

### 1. Browse Templates by Use Case

```bash
# E-commerce templates
mcp__flow-nexus__template_list { category: "ecommerce" }

# SaaS templates
mcp__flow-nexus__template_list { category: "saas" }

# Blog/CMS templates
mcp__flow-nexus__template_list { category: "cms" }

# Admin dashboards
mcp__flow-nexus__template_list { category: "admin" }
```

### 2. Use Template Search

```bash
# Search for specific technologies
mcp__flow-nexus__app_search {
  search: "nextjs stripe postgresql",
  category: "fullstack",
  limit: 10
}
```

### 3. Check Template Ratings and Reviews

```bash
# Rate a template after deployment
mcp__flow-nexus__neural_rate_template {
  template_id: "tmpl_ecommerce_001",
  rating: 5,
  review: "Excellent template! Deployed in minutes. Great documentation.",
  user_id: "usr_abc123"
}
```

### 4. Clone and Customize Templates

```bash
# Deploy template to sandbox for customization
mcp__flow-nexus__sandbox_create {
  template: "nextjs",
  name: "custom-ecommerce"
}

# Upload customized files
mcp__flow-nexus__sandbox_upload {
  sandbox_id: "sb_custom_001",
  file_path: "/app/components/CustomCheckout.tsx",
  content: "... your custom component ..."
}

# Publish as your own template
mcp__flow-nexus__neural_publish_template {
  model_id: "sb_custom_001",
  name: "ACME Custom E-commerce",
  description: "Customized e-commerce with ACME branding",
  category: "ecommerce",
  price: 100,
  user_id: "usr_abc123"
}
```

### 5. Use Environment-Specific Configurations

```bash
# Development deployment
mcp__flow-nexus__template_deploy {
  template_id: "tmpl_ecommerce_001",
  deployment_name: "acme-dev",
  variables: {
    "DATABASE_URL": "postgresql://localhost:5432/dev",
    "STRIPE_SECRET_KEY": "sk_test_...",
    "NODE_ENV": "development"
  }
}

# Production deployment
mcp__flow-nexus__template_deploy {
  template_id: "tmpl_ecommerce_001",
  deployment_name: "acme-prod",
  variables: {
    "DATABASE_URL": "postgresql://prod.db:5432/production",
    "STRIPE_SECRET_KEY": "sk_live_...",
    "NODE_ENV": "production"
  }
}
```

### 6. Monitor Deployment Costs

```bash
# Check balance before deployment
mcp__flow-nexus__check_balance {}

# Get payment history
mcp__flow-nexus__get_payment_history { limit: 10 }

# Set up billing alerts
mcp__flow-nexus__configure_auto_refill {
  enabled: true,
  threshold: 100,
  amount: 500
}
```

### 7. Use Swarm Templates for Complex Workflows

```bash
# List swarm templates
mcp__flow-nexus__swarm_templates_list {
  category: "enterprise",
  includeStore: true
}

# Create swarm from template
mcp__flow-nexus__swarm_create_from_template {
  template_name: "microservices-orchestration",
  overrides: {
    "maxAgents": 12,
    "strategy": "adaptive"
  }
}
```

### 8. Implement Blue-Green Deployments

```bash
# Deploy new version to "green" environment
mcp__flow-nexus__template_deploy {
  template_id: "tmpl_ecommerce_001",
  deployment_name: "acme-green",
  variables: { ... }
}

# Test green environment
# ... run tests ...

# Switch traffic from blue to green
mcp__flow-nexus__app_update {
  app_id: "acme-prod",
  updates: {
    "routing": {
      "target": "acme-green",
      "strategy": "instant"
    }
  }
}
```

### 9. Use Real-Time Deployment Monitoring

```bash
# Subscribe to deployment events
mcp__flow-nexus__realtime_subscribe {
  table: "deployments",
  event: "UPDATE",
  filter: "deployment_id = 'dep_acme_001'"
}

# Get live updates
mcp__flow-nexus__realtime_list {}
```

### 10. Leverage Template Marketplace

```bash
# Browse featured templates
mcp__flow-nexus__template_list {
  featured: true,
  limit: 50
}

# Get template by name
mcp__flow-nexus__template_get {
  template_name: "E-commerce Starter Kit"
}

# Deploy quickly by name
mcp__flow-nexus__template_deploy {
  template_name: "E-commerce Starter Kit",
  deployment_name: "quick-store",
  variables: { ... }
}
```

## Troubleshooting

### Deployment Fails During Build

**Problem**: Template deployment fails at "Application Build" stage
**Solution**: Check build logs and environment variables

```bash
mcp__flow-nexus__sandbox_logs {
  sandbox_id: "dep_acme_001",
  lines: 200
}
```

### Missing Required Variables

**Problem**: Error "Missing required variable: DATABASE_URL"
**Solution**: Get template requirements first

```bash
mcp__flow-nexus__template_get { template_id: "tmpl_ecommerce_001" }
# Check "required_variables" in response
```

### Template Not Found

**Problem**: Error "Template 'xyz' not found"
**Solution**: Search for correct template name or ID

```bash
mcp__flow-nexus__app_search {
  search: "ecommerce",
  category: "fullstack"
}
```

## Next Steps

1. **Customize Your App**: Modify code, add features, update branding
2. **Set Up CI/CD**: Automate deployments with GitHub Actions
3. **Configure Custom Domain**: Point your domain to the deployed app
4. **Add Monitoring**: Integrate Sentry, LogRocket, or similar tools
5. **Optimize Performance**: Use CDN, caching, and image optimization
6. **Publish Your Template**: Share your customized template in the marketplace
