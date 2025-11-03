# Example 1: Sandbox Deployment for Multi-Agent Code Execution

## Scenario

Your team needs to build and test a distributed microservices architecture with multiple agents working in parallel. Each agent needs an isolated execution environment to develop, test, and deploy their services without interfering with other agents' work. You'll use Flow Nexus sandboxes to create isolated environments for:

- Backend API development (Node.js)
- Frontend development (React)
- Database migrations (PostgreSQL)
- Integration testing
- Performance benchmarking

## Prerequisites

- Flow Nexus account (register via `npx flow-nexus@latest register`)
- Active session (login via `npx flow-nexus@latest login`)
- Basic understanding of E2B sandboxes
- MCP server configured for Flow Nexus

## Walkthrough

### Step 1: Initialize Authentication

```bash
# Check authentication status
mcp__flow-nexus__auth_status { detailed: true }

# If not authenticated, initialize
mcp__flow-nexus__auth_init { mode: "user" }

# Login with credentials
mcp__flow-nexus__user_login {
  email: "developer@company.com",
  password: "secure_password_here"
}
```

**Expected Output:**
```json
{
  "success": true,
  "user": {
    "id": "usr_abc123",
    "email": "developer@company.com",
    "tier": "pro",
    "credits": 1000
  },
  "session": {
    "token": "sess_xyz789",
    "expires": "2025-11-03T12:00:00Z"
  }
}
```

### Step 2: Create Backend API Sandbox

```bash
# Create Node.js sandbox for backend development
mcp__flow-nexus__sandbox_create {
  template: "node",
  name: "backend-api-dev",
  env_vars: {
    "NODE_ENV": "development",
    "PORT": "3000",
    "DATABASE_URL": "postgresql://localhost:5432/devdb",
    "JWT_SECRET": "dev_secret_key_change_in_prod",
    "API_VERSION": "v1"
  },
  install_packages: [
    "express",
    "jsonwebtoken",
    "bcrypt",
    "pg",
    "dotenv",
    "cors",
    "helmet"
  ],
  timeout: 7200,
  metadata: {
    "project": "microservices-demo",
    "agent": "backend-developer",
    "purpose": "API development and testing"
  }
}
```

**Expected Output:**
```json
{
  "sandbox_id": "sb_backend_001",
  "status": "running",
  "template": "node",
  "created_at": "2025-11-02T10:00:00Z",
  "packages_installed": 7,
  "env_vars_set": 5,
  "endpoint": "https://sb-backend-001.flow-nexus.io"
}
```

### Step 3: Deploy Backend Code to Sandbox

```bash
# Upload API server code
mcp__flow-nexus__sandbox_upload {
  sandbox_id: "sb_backend_001",
  file_path: "/app/server.js",
  content: `
const express = require('express');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const { Pool } = require('pg');
const helmet = require('helmet');
const cors = require('cors');

const app = express();
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Authentication middleware
const authenticate = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Unauthorized' });

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    res.status(403).json({ error: 'Invalid token' });
  }
};

// Routes
app.post('/api/v1/auth/register', async (req, res) => {
  try {
    const { email, password } = req.body;
    const hashedPassword = await bcrypt.hash(password, 10);

    const result = await pool.query(
      'INSERT INTO users (email, password) VALUES ($1, $2) RETURNING id, email',
      [email, hashedPassword]
    );

    const token = jwt.sign(
      { userId: result.rows[0].id, email: result.rows[0].email },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    res.status(201).json({ user: result.rows[0], token });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/v1/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    const result = await pool.query('SELECT * FROM users WHERE email = $1', [email]);

    if (result.rows.length === 0) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const user = result.rows[0];
    const validPassword = await bcrypt.compare(password, user.password);

    if (!validPassword) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const token = jwt.sign(
      { userId: user.id, email: user.email },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    res.json({ user: { id: user.id, email: user.email }, token });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/v1/users/me', authenticate, async (req, res) => {
  try {
    const result = await pool.query(
      'SELECT id, email, created_at FROM users WHERE id = $1',
      [req.user.userId]
    );
    res.json({ user: result.rows[0] });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(\`API server running on port \${PORT}\`);
});
`
}
```

### Step 4: Execute Backend Server in Sandbox

```bash
# Start the backend server
mcp__flow-nexus__sandbox_execute {
  sandbox_id: "sb_backend_001",
  code: "node /app/server.js",
  timeout: 300,
  capture_output: true,
  working_dir: "/app"
}
```

**Expected Output:**
```json
{
  "execution_id": "exec_001",
  "status": "running",
  "stdout": "API server running on port 3000\n",
  "stderr": "",
  "exit_code": null,
  "started_at": "2025-11-02T10:05:00Z"
}
```

### Step 5: Create Frontend Sandbox

```bash
# Create React sandbox for frontend development
mcp__flow-nexus__sandbox_create {
  template: "react",
  name: "frontend-app-dev",
  env_vars: {
    "REACT_APP_API_URL": "https://sb-backend-001.flow-nexus.io/api/v1",
    "REACT_APP_ENV": "development"
  },
  install_packages: [
    "axios",
    "react-router-dom",
    "zustand",
    "@tanstack/react-query"
  ],
  timeout: 7200,
  metadata: {
    "project": "microservices-demo",
    "agent": "frontend-developer",
    "purpose": "React UI development"
  }
}
```

### Step 6: Monitor Sandbox Status

```bash
# Check backend sandbox status
mcp__flow-nexus__sandbox_status { sandbox_id: "sb_backend_001" }

# Check frontend sandbox status
mcp__flow-nexus__sandbox_status { sandbox_id: "sb_frontend_001" }

# Get execution logs
mcp__flow-nexus__sandbox_logs {
  sandbox_id: "sb_backend_001",
  lines: 100
}
```

### Step 7: List All Active Sandboxes

```bash
# List all running sandboxes
mcp__flow-nexus__sandbox_list { status: "running" }
```

**Expected Output:**
```json
{
  "sandboxes": [
    {
      "sandbox_id": "sb_backend_001",
      "name": "backend-api-dev",
      "template": "node",
      "status": "running",
      "uptime": "00:15:32",
      "created_at": "2025-11-02T10:00:00Z"
    },
    {
      "sandbox_id": "sb_frontend_001",
      "name": "frontend-app-dev",
      "template": "react",
      "status": "running",
      "uptime": "00:08:12",
      "created_at": "2025-11-02T10:07:20Z"
    }
  ],
  "total": 2
}
```

### Step 8: Cleanup Sandboxes

```bash
# Stop backend sandbox
mcp__flow-nexus__sandbox_stop { sandbox_id: "sb_backend_001" }

# Delete frontend sandbox
mcp__flow-nexus__sandbox_delete { sandbox_id: "sb_frontend_001" }
```

## Outcomes

### What You Achieved

1. **Isolated Development Environments**: Created separate sandboxes for backend and frontend development with no interference
2. **Environment Configuration**: Set up environment variables and installed dependencies automatically
3. **Code Deployment**: Uploaded and executed application code in cloud sandboxes
4. **Monitoring**: Tracked sandbox status, logs, and execution metrics
5. **Resource Management**: Learned to list, stop, and delete sandboxes efficiently

### Metrics

- **Setup Time**: ~5 minutes for two sandboxes
- **Deployment Speed**: Code uploaded and executed in <30 seconds
- **Isolation**: Complete separation between environments
- **Cost**: ~10 rUv credits per hour per sandbox
- **Scalability**: Can create up to 100 sandboxes in parallel

## Tips and Best Practices

### 1. Environment Variable Management

```bash
# Use secure secrets management
mcp__flow-nexus__sandbox_create {
  template: "node",
  env_vars: {
    "DATABASE_URL": "${VAULT_SECRET:db_url}",  # Reference from vault
    "API_KEY": "${ENCRYPTED:api_key}"          # Encrypted storage
  }
}
```

### 2. Pre-Install Common Packages

```bash
# Create a startup script for common setup
mcp__flow-nexus__sandbox_create {
  template: "node",
  startup_script: `
    npm install -g pnpm
    pnpm install --frozen-lockfile
    pnpm run build
    pnpm run test
  `
}
```

### 3. Use Templates for Consistency

```bash
# Deploy from pre-built template
mcp__flow-nexus__template_deploy {
  template_name: "express-api-boilerplate",
  deployment_name: "my-api-instance",
  variables: {
    "port": "3000",
    "db_connection": "postgresql://..."
  }
}
```

### 4. Monitor Resource Usage

```bash
# Check credit balance before creating sandboxes
mcp__flow-nexus__check_balance {}

# Enable auto-refill to avoid interruptions
mcp__flow-nexus__configure_auto_refill {
  enabled: true,
  threshold: 100,  # Refill when credits < 100
  amount: 500      # Add $50 worth of credits
}
```

### 5. Implement Health Checks

```bash
# Add health check endpoint to your applications
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    timestamp: new Date().toISOString()
  });
});
```

### 6. Use Execution Streams for Real-Time Monitoring

```bash
# Subscribe to execution stream
mcp__flow-nexus__execution_stream_subscribe {
  sandbox_id: "sb_backend_001",
  stream_type: "claude-code"
}

# Check stream status
mcp__flow-nexus__execution_stream_status {
  sandbox_id: "sb_backend_001"
}
```

### 7. Organize Sandboxes with Metadata

```bash
# Tag sandboxes for easy filtering
mcp__flow-nexus__sandbox_create {
  template: "node",
  metadata: {
    "team": "backend",
    "project": "ecommerce",
    "environment": "staging",
    "owner": "john.doe@company.com",
    "cost_center": "engineering"
  }
}
```

### 8. Implement Graceful Shutdown

```javascript
// In your application code
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully...');
  await server.close();
  await pool.end();
  process.exit(0);
});
```

### 9. Use Storage for Persistent Data

```bash
# Upload files to persistent storage
mcp__flow-nexus__storage_upload {
  bucket: "app-data",
  path: "backups/db-snapshot-2025-11-02.sql",
  content: "... base64 encoded data ..."
}

# List stored files
mcp__flow-nexus__storage_list {
  bucket: "app-data",
  path: "backups/"
}
```

### 10. Batch Sandbox Operations

```bash
# Create multiple sandboxes in one operation
[Single Message - Parallel Sandbox Creation]:
  mcp__flow-nexus__sandbox_create { template: "node", name: "api-1" }
  mcp__flow-nexus__sandbox_create { template: "node", name: "api-2" }
  mcp__flow-nexus__sandbox_create { template: "react", name: "frontend-1" }
  mcp__flow-nexus__sandbox_create { template: "python", name: "ml-worker-1" }
```

## Troubleshooting

### Sandbox Creation Fails

**Problem**: Sandbox creation returns error "Insufficient credits"
**Solution**: Check balance and refill credits

```bash
mcp__flow-nexus__check_balance {}
mcp__flow-nexus__create_payment_link { amount: 50 }
```

### Package Installation Timeout

**Problem**: npm install takes too long and times out
**Solution**: Use pnpm or yarn with frozen lockfile

```bash
mcp__flow-nexus__sandbox_configure {
  sandbox_id: "sb_001",
  run_commands: [
    "npm install -g pnpm",
    "pnpm install --frozen-lockfile --prefer-offline"
  ]
}
```

### Environment Variables Not Loading

**Problem**: Application can't access environment variables
**Solution**: Verify variables are set in sandbox configuration

```bash
mcp__flow-nexus__sandbox_status { sandbox_id: "sb_001" }
# Check "env_vars" in response
```

## Next Steps

1. **Explore Templates**: Browse available templates with `mcp__flow-nexus__template_list`
2. **Deploy Complex Apps**: Use `mcp__flow-nexus__template_deploy` for full-stack applications
3. **Set Up CI/CD**: Integrate sandbox deployment into GitHub Actions workflows
4. **Monitor Costs**: Track usage with `mcp__flow-nexus__get_payment_history`
5. **Scale Horizontally**: Create multiple sandboxes for load testing and performance benchmarking
