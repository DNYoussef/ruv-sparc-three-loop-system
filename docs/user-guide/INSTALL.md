# Installation Guide

Welcome to the rUv SPARC UI Dashboard! This guide will walk you through the installation process step-by-step.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [First-Time Setup](#first-time-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## Prerequisites

Before installing the rUv SPARC UI Dashboard, ensure you have the following software installed:

### Required Software

#### 1. Docker Desktop
- **Version**: 20.10 or higher
- **Download**: https://www.docker.com/products/docker-desktop
- **Purpose**: Runs containerized services (PostgreSQL, Redis)

**Installation Steps**:
```bash
# Windows: Download and run Docker Desktop installer
# macOS: Download Docker.dmg and drag to Applications
# Linux: Follow distribution-specific instructions

# Verify installation
docker --version
docker-compose --version
```

**Expected Output**:
```
Docker version 24.0.0, build abc1234
Docker Compose version v2.20.0
```

#### 2. Node.js
- **Version**: 18.x or higher (LTS recommended)
- **Download**: https://nodejs.org/
- **Purpose**: Runs the frontend and backend services

**Installation Steps**:
```bash
# Download and install from nodejs.org
# Or use a version manager (recommended)

# Using nvm (Node Version Manager)
nvm install 18
nvm use 18

# Verify installation
node --version
npm --version
```

**Expected Output**:
```
v18.17.0
9.8.0
```

#### 3. Python
- **Version**: 3.9 or higher
- **Download**: https://www.python.org/downloads/
- **Purpose**: Runs MCP servers and backend services

**Installation Steps**:
```bash
# Download and install from python.org
# Ensure "Add Python to PATH" is checked during installation

# Verify installation
python --version
pip --version
```

**Expected Output**:
```
Python 3.11.5
pip 23.2.1
```

#### 4. Git
- **Version**: 2.x or higher
- **Download**: https://git-scm.com/downloads
- **Purpose**: Clone the repository

**Verification**:
```bash
git --version
```

### Optional but Recommended

- **Visual Studio Code**: Code editor with integrated terminal
- **Postman**: API testing tool
- **OBS Studio**: For recording video tutorials (if contributing documentation)

---

## Installation Steps

### Step 1: Clone the Repository

```bash
# Navigate to your projects directory
cd ~/projects

# Clone the repository
git clone https://github.com/yourusername/ruv-sparc-ui-dashboard.git

# Navigate into the project
cd ruv-sparc-ui-dashboard
```

**What you should see**:
```
Cloning into 'ruv-sparc-ui-dashboard'...
remote: Enumerating objects: 1523, done.
remote: Counting objects: 100% (1523/1523), done.
Receiving objects: 100% (1523/1523), 2.45 MiB | 3.21 MiB/s, done.
```

### Step 2: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your preferred editor
# Windows: notepad .env
# macOS/Linux: nano .env
# VS Code: code .env
```

**Required Configuration**:

```env
# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ruv_sparc
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ruv_sparc

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Backend Configuration
BACKEND_PORT=3001
NODE_ENV=development

# Frontend Configuration
FRONTEND_PORT=3000
REACT_APP_API_URL=http://localhost:3001

# WebSocket Configuration
WS_PORT=3002

# Memory MCP Configuration
MEMORY_MCP_ENABLED=true
MEMORY_MCP_URL=http://localhost:8765

# Authentication (change in production!)
JWT_SECRET=your-secret-key-change-in-production
SESSION_SECRET=your-session-secret-change-in-production

# Optional: Analytics
ENABLE_ANALYTICS=false
```

**⚠️ Security Note**: Change `JWT_SECRET` and `SESSION_SECRET` in production!

### Step 3: Install Dependencies

#### Backend Dependencies
```bash
# Navigate to backend directory
cd backend

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt

# Return to project root
cd ..
```

**Expected Output**:
```
added 342 packages in 12.5s
Successfully installed fastapi-0.104.1 uvicorn-0.24.0 ...
```

#### Frontend Dependencies
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Return to project root
cd ..
```

**Expected Output**:
```
added 1243 packages in 28.3s
```

### Step 4: Start Docker Services

```bash
# Start PostgreSQL and Redis containers
docker-compose up -d

# Verify containers are running
docker ps
```

**Expected Output**:
```
CONTAINER ID   IMAGE              STATUS          PORTS
abc123def456   postgres:15        Up 10 seconds   0.0.0.0:5432->5432/tcp
def456ghi789   redis:7-alpine     Up 10 seconds   0.0.0.0:6379->6379/tcp
```

**What's happening**:
- PostgreSQL database starts on port 5432
- Redis cache starts on port 6379
- Data is persisted in Docker volumes

### Step 5: Initialize Database

```bash
# Run database migrations
cd backend
npm run migrate

# Seed initial data (optional)
npm run seed
```

**Expected Output**:
```
Running migrations...
✓ Created tables: users, projects, tasks, agents, workflows
✓ Created indexes
✓ Seeded default admin user

Database initialized successfully!
```

### Step 6: Start the Application

**Option A: Development Mode (Recommended for first-time setup)**

```bash
# Terminal 1: Start backend
cd backend
npm run dev

# Terminal 2: Start frontend
cd frontend
npm start

# Terminal 3: Start WebSocket server
cd backend
npm run ws

# Terminal 4: Start Memory MCP server (if using Memory MCP)
npx memory-mcp start
```

**Option B: Production Mode**

```bash
# Build frontend
cd frontend
npm run build

# Start backend with built frontend
cd ../backend
npm start
```

**What you should see**:

**Backend** (Terminal 1):
```
[2025-01-08 10:30:00] Server started on http://localhost:3001
[2025-01-08 10:30:00] Database connected
[2025-01-08 10:30:00] Redis connected
```

**Frontend** (Terminal 2):
```
Compiled successfully!

You can now view ruv-sparc-ui in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.1.100:3000
```

**WebSocket** (Terminal 3):
```
WebSocket server running on ws://localhost:3002
```

---

## First-Time Setup

### 1. Create Admin User

**Option A: Using the Web Interface**

1. Open your browser and navigate to http://localhost:3000
2. Click "Create Account" on the login page
3. Fill in the registration form:
   - **Username**: admin
   - **Email**: admin@example.com
   - **Password**: (choose a strong password)
4. Click "Sign Up"
5. You'll be redirected to the dashboard

**Option B: Using the CLI**

```bash
cd backend
npm run create-admin

# Follow the prompts
# Enter username: admin
# Enter email: admin@example.com
# Enter password: ********
```

**Expected Output**:
```
Admin user created successfully!
Username: admin
Email: admin@example.com
Role: admin
```

### 2. First Login

1. Navigate to http://localhost:3000
2. Enter your credentials
3. Click "Log In"

**You should see**:
- Dashboard with empty state (no projects or tasks)
- Welcome message
- Quick start guide

### 3. Configure Preferences

1. Click the **Settings** icon (⚙️) in the top-right corner
2. Configure your preferences:
   - **Theme**: Light / Dark / Auto
   - **Notifications**: Email, Push, In-app
   - **Time Zone**: Your local timezone
   - **Date Format**: MM/DD/YYYY or DD/MM/YYYY
3. Click "Save Settings"

### 4. Create Your First Project

1. Click the **"+ New Project"** button
2. Fill in project details:
   - **Name**: My First Project
   - **Description**: Learning the rUv SPARC UI Dashboard
   - **Tags**: tutorial, learning
3. Click "Create Project"

### 5. Create Your First Task

1. Navigate to the **Calendar** view
2. Click on today's date
3. Fill in task details:
   - **Title**: Complete installation
   - **Description**: Set up the dashboard successfully
   - **Priority**: High
   - **Due Date**: Today
4. Click "Create Task"

**Congratulations!** 🎉 You've successfully set up the rUv SPARC UI Dashboard!

---

## Verification

### Health Checks

**Backend Health Check**:
```bash
curl http://localhost:3001/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  "memory_mcp": "available",
  "uptime": 123.45
}
```

**Frontend Accessibility**:
- Open http://localhost:3000
- You should see the login page
- No console errors in browser DevTools (F12)

**WebSocket Connection**:
```bash
# Using wscat (install with: npm install -g wscat)
wscat -c ws://localhost:3002

# You should see: Connected (press ^C to quit)
```

### Database Verification

```bash
# Connect to PostgreSQL
docker exec -it ruv-sparc-postgres psql -U postgres -d ruv_sparc

# List tables
\dt

# Expected output:
#  Schema |     Name      | Type  |  Owner
# --------+---------------+-------+----------
#  public | users         | table | postgres
#  public | projects      | table | postgres
#  public | tasks         | table | postgres
#  public | agents        | table | postgres
#  public | workflows     | table | postgres

# Exit
\q
```

---

## Troubleshooting

### Common Issues

#### 1. Docker Not Running

**Error**:
```
Cannot connect to the Docker daemon. Is the docker daemon running?
```

**Solution**:
- **Windows/macOS**: Open Docker Desktop application
- **Linux**: `sudo systemctl start docker`
- Verify: `docker ps` should not error

#### 2. Port Already in Use

**Error**:
```
Error: listen EADDRINUSE: address already in use :::3000
```

**Solution**:

**Option A: Kill the process using the port**
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:3000 | xargs kill -9
```

**Option B: Change the port**
```bash
# Edit .env file
FRONTEND_PORT=3001  # Change to an available port

# Restart the application
```

#### 3. Database Connection Failed

**Error**:
```
Error: connect ECONNREFUSED 127.0.0.1:5432
```

**Solution**:
1. Verify Docker containers are running:
   ```bash
   docker ps
   ```
2. If PostgreSQL container is not running:
   ```bash
   docker-compose up -d postgres
   ```
3. Check database logs:
   ```bash
   docker logs ruv-sparc-postgres
   ```
4. Verify connection string in `.env`:
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ruv_sparc
   ```

#### 4. Memory MCP Unavailable

**Error**:
```
Warning: Memory MCP service unavailable
```

**Solution**:
1. Check if Memory MCP is installed:
   ```bash
   npm list -g memory-mcp
   ```
2. Install if missing:
   ```bash
   npm install -g memory-mcp
   ```
3. Start the service:
   ```bash
   npx memory-mcp start
   ```
4. Verify it's running:
   ```bash
   curl http://localhost:8765/health
   ```

#### 5. npm install Fails

**Error**:
```
npm ERR! code EACCES
npm ERR! syscall access
```

**Solution**:
- **Windows**: Run terminal as Administrator
- **macOS/Linux**: Don't use `sudo` with npm
- Fix permissions:
  ```bash
  # macOS/Linux
  sudo chown -R $USER:$USER ~/.npm
  sudo chown -R $USER:$USER ~/projects/ruv-sparc-ui-dashboard
  ```

#### 6. Frontend Build Fails

**Error**:
```
Module not found: Error: Can't resolve 'react'
```

**Solution**:
```bash
# Clear node_modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

#### 7. WebSocket Connection Refused

**Error** (in browser console):
```
WebSocket connection to 'ws://localhost:3002/' failed
```

**Solution**:
1. Check if WebSocket server is running:
   ```bash
   lsof -i :3002  # macOS/Linux
   netstat -ano | findstr :3002  # Windows
   ```
2. Start WebSocket server:
   ```bash
   cd backend
   npm run ws
   ```
3. Check firewall settings (allow port 3002)

---

## Next Steps

Now that you have the rUv SPARC UI Dashboard installed, here's what to do next:

### 1. Read the User Guide
- Learn how to use the Calendar UI
- Explore the Project Dashboard
- Understand the Agent Monitor
- Configure advanced settings

**Link**: [USER_GUIDE.md](USER_GUIDE.md)

### 2. Explore the API
- Learn how to authenticate
- Make API requests
- Integrate with external tools

**Link**: [API_GUIDE.md](API_GUIDE.md)

### 3. Join the Community
- GitHub Discussions: https://github.com/yourusername/ruv-sparc-ui-dashboard/discussions
- Discord: https://discord.gg/ruv-sparc
- Twitter: @ruvSPARC

### 4. Contribute
- Report bugs: https://github.com/yourusername/ruv-sparc-ui-dashboard/issues
- Submit pull requests
- Improve documentation

---

## Support

If you encounter issues not covered in this guide:

1. **Check the FAQ**: [FAQ.md](FAQ.md)
2. **Search existing issues**: https://github.com/yourusername/ruv-sparc-ui-dashboard/issues
3. **Ask on Discord**: https://discord.gg/ruv-sparc
4. **Create a new issue**: Include:
   - Operating system and version
   - Node.js, Python, Docker versions
   - Full error message
   - Steps to reproduce

---

**Happy building with rUv SPARC UI Dashboard!** 🚀
