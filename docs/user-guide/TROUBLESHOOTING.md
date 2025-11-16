# Troubleshooting Guide

This guide helps you resolve common issues with the rUv SPARC UI Dashboard.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Connection Issues](#connection-issues)
3. [Authentication Issues](#authentication-issues)
4. [Performance Issues](#performance-issues)
5. [UI/UX Issues](#uiux-issues)
6. [Data Issues](#data-issues)
7. [Agent Issues](#agent-issues)
8. [Integration Issues](#integration-issues)
9. [Getting Help](#getting-help)

---

## Installation Issues

### 1. Docker Desktop Not Running

**Symptoms**:
- Error: `Cannot connect to the Docker daemon`
- `docker-compose up` fails

**Diagnosis**:
```bash
docker ps
# Error: Cannot connect to the Docker daemon...
```

**Solutions**:

**Windows/macOS**:
1. Open Docker Desktop application
2. Wait for it to fully start (whale icon in tray should be stable)
3. Verify with `docker ps`

**Linux**:
```bash
# Start Docker service
sudo systemctl start docker

# Enable on boot
sudo systemctl enable docker

# Verify
docker ps
```

**If Docker Desktop won't start**:
- Check system requirements (virtualization enabled in BIOS)
- Restart computer
- Reinstall Docker Desktop
- Check for conflicting VPN/antivirus software

---

### 2. Port Already in Use

**Symptoms**:
- Error: `EADDRINUSE: address already in use :::3000`
- Application won't start

**Diagnosis**:
```bash
# Windows
netstat -ano | findstr :3000

# macOS/Linux
lsof -i :3000
```

**Solutions**:

**Option A: Kill the process**

**Windows**:
```bash
# Find PID from netstat output
taskkill /PID <PID> /F
```

**macOS/Linux**:
```bash
# Kill process using port 3000
lsof -ti:3000 | xargs kill -9
```

**Option B: Change the port**

Edit `.env`:
```env
FRONTEND_PORT=3001  # Change to available port
BACKEND_PORT=3002
WS_PORT=3003
```

Then restart the application.

---

### 3. npm install Fails

**Symptoms**:
- `npm ERR! code EACCES`
- Permission errors during installation

**Solutions**:

**Windows**:
- Run terminal as Administrator
- Or use WSL2 for better permission handling

**macOS/Linux**:
```bash
# Fix npm permissions
sudo chown -R $USER:$USER ~/.npm
sudo chown -R $USER:$USER ~/projects/ruv-sparc-ui-dashboard

# Never use sudo with npm!
# If you've used sudo before, fix ownership:
sudo chown -R $USER:$USER node_modules
```

**If installation is slow or hangs**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

---

### 4. Database Migration Fails

**Symptoms**:
- `npm run migrate` fails
- Tables not created

**Diagnosis**:
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check database logs
docker logs ruv-sparc-postgres
```

**Solutions**:

**1. Ensure PostgreSQL is running**:
```bash
docker-compose up -d postgres
```

**2. Check connection string** in `.env`:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ruv_sparc
```

**3. Reset database** (⚠️ deletes all data):
```bash
# Stop containers
docker-compose down

# Remove volumes
docker volume rm ruv-sparc-ui-dashboard_postgres_data

# Restart and migrate
docker-compose up -d
cd backend
npm run migrate
```

**4. Manual migration**:
```bash
# Connect to database
docker exec -it ruv-sparc-postgres psql -U postgres -d ruv_sparc

# Run migrations manually
\i /path/to/migrations/001_initial.sql

# Exit
\q
```

---

## Connection Issues

### 1. WebSocket Connection Refused

**Symptoms**:
- Browser console: `WebSocket connection to 'ws://localhost:3002/' failed`
- Real-time updates not working

**Diagnosis**:
```bash
# Check if WebSocket server is running
# Windows
netstat -ano | findstr :3002

# macOS/Linux
lsof -i :3002
```

**Solutions**:

**1. Start WebSocket server**:
```bash
cd backend
npm run ws
```

**2. Check firewall** (Windows):
```
Settings → Privacy & Security → Windows Firewall
→ Allow an app through firewall
→ Add Node.js (if not listed)
```

**3. Verify WebSocket URL** in frontend config:
```javascript
// frontend/src/config.js
export const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:3002';
```

**4. Browser extensions**:
- Disable ad blockers temporarily
- Disable privacy extensions (they may block WebSockets)

---

### 2. Memory MCP Unavailable

**Symptoms**:
- Warning: `Memory MCP service unavailable`
- Agent coordination not working

**Diagnosis**:
```bash
# Check if Memory MCP is running
curl http://localhost:8765/health
```

**Solutions**:

**1. Install Memory MCP** (if not installed):
```bash
npm install -g memory-mcp
```

**2. Start Memory MCP**:
```bash
npx memory-mcp start

# Or run in background
npx memory-mcp start --daemon
```

**3. Verify configuration** in `.env`:
```env
MEMORY_MCP_ENABLED=true
MEMORY_MCP_URL=http://localhost:8765
```

**4. Check logs**:
```bash
# Memory MCP logs location
cat ~/.memory-mcp/logs/latest.log
```

**5. Reset Memory MCP**:
```bash
# Stop service
npx memory-mcp stop

# Clear data (⚠️ deletes stored memories)
rm -rf ~/.memory-mcp/data

# Restart
npx memory-mcp start
```

---

### 3. Database Connection Failed

**Symptoms**:
- Error: `connect ECONNREFUSED 127.0.0.1:5432`
- Backend won't start

**Diagnosis**:
```bash
# Check if PostgreSQL container is running
docker ps | grep postgres

# If not running, check why
docker logs ruv-sparc-postgres
```

**Solutions**:

**1. Start PostgreSQL**:
```bash
docker-compose up -d postgres
```

**2. Verify connection string** in `.env`:
```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ruv_sparc
```

**3. Test connection manually**:
```bash
# Using psql
psql postgresql://postgres:postgres@localhost:5432/ruv_sparc

# Using Docker exec
docker exec -it ruv-sparc-postgres psql -U postgres -d ruv_sparc
```

**4. Check for port conflicts**:
```bash
# Make sure nothing else is using port 5432
# Windows
netstat -ano | findstr :5432

# macOS/Linux
lsof -i :5432
```

**5. Reset PostgreSQL container**:
```bash
docker-compose down
docker volume rm ruv-sparc-ui-dashboard_postgres_data
docker-compose up -d postgres
```

---

## Authentication Issues

### 1. Login Fails with Correct Credentials

**Symptoms**:
- Correct username/password rejected
- Error: "Invalid credentials"

**Diagnosis**:
```bash
# Check if user exists in database
docker exec -it ruv-sparc-postgres psql -U postgres -d ruv_sparc
SELECT id, email, username FROM users WHERE email = 'your-email@example.com';
```

**Solutions**:

**1. Reset password**:
```bash
cd backend
npm run reset-password -- --email admin@example.com
```

**2. Verify password hashing**:
- Ensure `bcrypt` is installed: `npm list bcrypt`
- Check backend logs for password comparison errors

**3. Clear browser cookies**:
- Chrome: Settings → Privacy → Clear browsing data → Cookies
- Firefox: Settings → Privacy → Clear Data → Cookies

**4. Check JWT secret** in `.env`:
```env
JWT_SECRET=your-secret-key-change-in-production
```

If changed after login, all tokens are invalid. Users must re-login.

---

### 2. Token Expired Error

**Symptoms**:
- Logged out unexpectedly
- Error: "Token expired"

**Solutions**:

**1. Refresh token automatically** (frontend):
```javascript
// Check if token is expiring soon
const tokenExpiresIn = getTokenExpirationTime(token);
if (tokenExpiresIn < 300) { // Less than 5 minutes
  await refreshToken();
}
```

**2. Increase token lifetime** (backend `.env`):
```env
JWT_EXPIRATION=3600  # 1 hour (default)
# Increase to 7200 for 2 hours, etc.
```

**3. Use refresh tokens**:
- Frontend should store and use refresh tokens
- Automatically request new access token when expired

---

### 3. API Key Not Working

**Symptoms**:
- 401 Unauthorized with API key
- Error: "Invalid API key"

**Solutions**:

**1. Verify API key header**:
```bash
curl http://localhost:3001/api/v1/tasks \
  -H "X-API-Key: your-api-key-here"
```

**2. Check if API key is active**:
- Log in to web UI
- Settings → API Access
- Verify key is not revoked

**3. Regenerate API key**:
- Revoke old key
- Generate new key
- Update applications using the key

---

## Performance Issues

### 1. Slow Page Load

**Symptoms**:
- Initial page load takes >5 seconds
- Sluggish navigation

**Diagnosis**:
```bash
# Check backend response time
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:3001/api/v1/tasks

# curl-format.txt:
# time_total: %{time_total}s
```

**Solutions**:

**1. Enable production build** (frontend):
```bash
cd frontend
npm run build

# Serve built files
cd ../backend
npm start
```

**2. Check database query performance**:
```sql
-- Connect to database
docker exec -it ruv-sparc-postgres psql -U postgres -d ruv_sparc

-- Analyze slow queries
EXPLAIN ANALYZE SELECT * FROM tasks WHERE status = 'pending';

-- Add missing indexes
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_due_date ON tasks(due_date);
```

**3. Reduce API request frequency**:
- Implement request debouncing
- Cache responses client-side
- Use pagination

**4. Optimize images/assets**:
```bash
# Compress images
npm install -g imagemin-cli
imagemin frontend/public/images/* --out-dir=frontend/public/images/optimized
```

---

### 2. High Memory Usage

**Symptoms**:
- System runs out of RAM
- Docker containers using excessive memory

**Diagnosis**:
```bash
# Check Docker memory usage
docker stats

# Check Node.js memory usage
# Add to backend startup
node --max-old-space-size=2048 server.js
```

**Solutions**:

**1. Limit Docker memory**:

Edit `docker-compose.yml`:
```yaml
services:
  postgres:
    mem_limit: 512m
  redis:
    mem_limit: 256m
```

**2. Increase Node.js heap size**:
```bash
# In package.json
"scripts": {
  "start": "node --max-old-space-size=4096 server.js"
}
```

**3. Clear Memory MCP cache**:
```bash
npx memory-mcp clear-cache
```

**4. Restart services**:
```bash
docker-compose restart
```

---

### 3. Agent Workflows Slow

**Symptoms**:
- Workflows take much longer than estimated
- Agents appear stuck

**Diagnosis**:
- Check Agent Monitor for bottlenecks
- Review agent logs for errors

**Solutions**:

**1. Check agent resource allocation**:
```javascript
// backend/config/agents.js
export const agentConfig = {
  coder: {
    maxConcurrent: 2, // Increase for parallel tasks
    timeout: 300000   // Increase timeout if needed
  }
};
```

**2. Review workflow design**:
- Identify sequential vs. parallel tasks
- Optimize task dependencies
- Remove unnecessary steps

**3. Restart agents**:
- Agent Monitor → Select agent → Restart

**4. Clear agent memory**:
```bash
npx memory-mcp clear --filter "agent/*"
```

---

## UI/UX Issues

### 1. Calendar Not Displaying Tasks

**Symptoms**:
- Calendar loads but tasks don't appear
- No errors in console

**Diagnosis**:
```bash
# Check API response
curl http://localhost:3001/api/v1/tasks \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Solutions**:

**1. Check date filters**:
- Ensure calendar view includes task due dates
- Try switching to different view (month/week/day)

**2. Clear browser cache**:
```
Chrome: Ctrl+Shift+Delete → Clear cache
Firefox: Ctrl+Shift+Delete → Clear cache
```

**3. Check Redux state** (if using Redux):
```javascript
// In browser console
console.log(store.getState().tasks);
```

**4. Verify API permissions**:
- Ensure user has permission to view tasks
- Check API logs for 403 Forbidden errors

---

### 2. Drag-and-Drop Not Working

**Symptoms**:
- Cannot drag tasks to reschedule
- Drag events not firing

**Solutions**:

**1. Browser compatibility**:
- Update to latest browser version
- Try different browser (Chrome, Firefox, Edge)

**2. Check for conflicting libraries**:
- Disable browser extensions
- Check for JavaScript errors in console

**3. Re-enable drag-and-drop** (if disabled):
```javascript
// frontend/src/components/Calendar.js
<Task draggable={true} onDragStart={handleDragStart} />
```

**4. Clear component state**:
```javascript
// Force re-render
localStorage.removeItem('calendar-state');
window.location.reload();
```

---

### 3. Dark Mode Not Persisting

**Symptoms**:
- Dark mode resets to light mode on refresh
- Theme preference not saved

**Solutions**:

**1. Check localStorage**:
```javascript
// In browser console
localStorage.setItem('theme', 'dark');
window.location.reload();
```

**2. Verify theme persistence** in code:
```javascript
// frontend/src/hooks/useTheme.js
useEffect(() => {
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    setTheme(savedTheme);
  }
}, []);
```

**3. Check browser privacy settings**:
- Ensure cookies/localStorage are allowed
- Disable "Clear on exit" settings

---

## Data Issues

### 1. Tasks Not Saving

**Symptoms**:
- Create/update task appears to work, but changes don't persist
- No error message

**Diagnosis**:
```bash
# Check backend logs
# In backend directory
npm run logs

# Or check Docker logs
docker logs ruv-sparc-backend
```

**Solutions**:

**1. Check database connection**:
```bash
# Verify database is running
docker ps | grep postgres

# Check for database errors
docker logs ruv-sparc-postgres
```

**2. Verify API request**:
```bash
# Test API directly
curl -X POST http://localhost:3001/api/v1/tasks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test task", "priority": "medium"}'
```

**3. Check transaction rollbacks**:
- Review backend logs for transaction errors
- Ensure database constraints are satisfied

**4. Test with minimal data**:
```javascript
// Create task with only required fields
{
  "title": "Minimal test"
}
```

---

### 2. Data Sync Issues Between Calendar and Projects

**Symptoms**:
- Task appears in calendar but not in project
- Or vice versa

**Solutions**:

**1. Force refresh**:
```javascript
// In browser console
window.location.reload();
```

**2. Check WebSocket connection**:
- Ensure real-time sync is working
- See [WebSocket Connection Refused](#1-websocket-connection-refused)

**3. Manually trigger sync**:
```bash
# Backend endpoint to force sync
curl -X POST http://localhost:3001/api/v1/sync/tasks \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**4. Check task-project linking**:
```sql
-- In database
SELECT t.id, t.title, p.name as project
FROM tasks t
LEFT JOIN projects p ON t.project_id = p.id
WHERE t.id = 'task_abc123';
```

---

### 3. Lost Data After System Crash

**Symptoms**:
- Recent changes missing after crash/restart
- Data reverted to earlier state

**Solutions**:

**1. Check database backup**:
```bash
# List backups
ls -lh ~/ruv-sparc-backups/

# Restore from backup
docker exec -i ruv-sparc-postgres pg_restore -U postgres -d ruv_sparc < backup.sql
```

**2. Enable auto-save** (prevent future data loss):
```javascript
// frontend/src/hooks/useAutoSave.js
useEffect(() => {
  const timer = setInterval(() => {
    saveToBackend();
  }, 30000); // Auto-save every 30 seconds

  return () => clearInterval(timer);
}, []);
```

**3. Check transaction logs**:
```sql
-- PostgreSQL transaction log
SELECT * FROM pg_stat_activity;
```

---

## Agent Issues

### 1. Agent Stuck in "Working" State

**Symptoms**:
- Agent shows "Working" for extended time
- No progress updates

**Diagnosis**:
- Check Agent Monitor logs for errors
- Review agent task timeout settings

**Solutions**:

**1. Restart agent**:
- Agent Monitor → Select agent → Restart

**2. Increase timeout** (backend):
```javascript
// backend/config/agents.js
export const agentConfig = {
  coder: {
    timeout: 600000 // 10 minutes (increase if needed)
  }
};
```

**3. Cancel stuck task**:
```bash
curl -X DELETE http://localhost:3001/api/v1/agents/coder/cancel-task \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**4. Review agent code** for infinite loops:
- Check agent implementation
- Add timeout guards

---

### 2. Agent Fails with "Out of Memory"

**Symptoms**:
- Agent crashes
- Error: "JavaScript heap out of memory"

**Solutions**:

**1. Increase Node.js memory**:
```bash
# In agent startup script
node --max-old-space-size=4096 agent.js
```

**2. Optimize agent tasks**:
- Break large tasks into smaller chunks
- Process data incrementally
- Clear variables after use

**3. Restart agent service**:
```bash
# Backend
npm run restart-agents
```

---

### 3. Workflow Not Triggering

**Symptoms**:
- Manual trigger button doesn't work
- No workflow starts

**Diagnosis**:
```bash
# Check workflow service
curl http://localhost:3001/api/v1/workflows/status \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Solutions**:

**1. Check workflow configuration**:
```javascript
// Ensure workflow template exists
const templates = await api.get('/workflows/templates');
console.log(templates);
```

**2. Verify agent availability**:
- Ensure required agents are running
- Check Agent Monitor for agent status

**3. Review workflow logs**:
```bash
# Backend logs
npm run logs | grep workflow
```

**4. Test with simple workflow**:
- Try triggering a minimal workflow
- Gradually add complexity

---

## Integration Issues

### 1. GitHub Integration Not Working

**Symptoms**:
- Cannot link tasks to GitHub issues
- GitHub sync fails

**Solutions**:

**1. Verify GitHub token**:
- Settings → Integrations → GitHub
- Regenerate token if expired

**2. Check permissions**:
- Token needs `repo` scope
- Verify organization access

**3. Test API connection**:
```bash
curl https://api.github.com/user \
  -H "Authorization: token YOUR_GITHUB_TOKEN"
```

**4. Reconnect integration**:
- Settings → Integrations → Disconnect
- Reconnect and re-authorize

---

### 2. Email Notifications Not Sending

**Symptoms**:
- Notification preferences enabled, but no emails received
- No errors shown

**Diagnosis**:
```bash
# Check email service logs
npm run logs | grep email
```

**Solutions**:

**1. Verify SMTP configuration** (`.env`):
```env
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

**2. Test SMTP connection**:
```bash
# Use test script
cd backend
npm run test-email -- --to your-email@example.com
```

**3. Check spam folder**:
- Emails may be filtered as spam
- Add sender to contacts

**4. Use app-specific password** (Gmail):
- Don't use regular password
- Generate app password: Google Account → Security → App passwords

---

## Getting Help

If you've tried the solutions above and still have issues:

### 1. Check the FAQ

**Link**: [FAQ.md](FAQ.md)

Common questions about features, usage, and troubleshooting.

### 2. Search GitHub Issues

**Link**: https://github.com/yourusername/ruv-sparc-ui-dashboard/issues

Someone may have reported the same issue with a solution.

### 3. Ask the Community

**Discord**: https://discord.gg/ruv-sparc
- #help channel for troubleshooting
- #general for usage questions

### 4. Create a GitHub Issue

If you've found a bug, create a detailed issue:

**Include**:
1. **Environment**:
   - OS (Windows 10, macOS 14, Ubuntu 22.04)
   - Node.js version (`node --version`)
   - Docker version (`docker --version`)
2. **Steps to Reproduce**:
   - Exact steps to trigger the issue
   - Expected vs. actual behavior
3. **Error Messages**:
   - Full error text
   - Screenshots if applicable
4. **Logs**:
   - Backend logs
   - Browser console logs
   - Docker logs (if relevant)

**Example**:
```markdown
## Bug Report

**Environment**:
- Windows 11
- Node.js v18.17.0
- Docker 24.0.5

**Steps to Reproduce**:
1. Create a task with title "Test"
2. Set due date to tomorrow
3. Click "Save"

**Expected**: Task should be saved and appear in calendar
**Actual**: Error "Failed to save task"

**Error Message**:
```
Error: ECONNREFUSED 127.0.0.1:5432
    at TCPConnectWrap.afterConnect [as oncomplete]
```

**Logs**: (attached)
```

### 5. Email Support

For private issues or enterprise support:

**Email**: support@ruv-sparc.io

**Include**:
- Brief description of issue
- Environment details
- Steps to reproduce
- Any sensitive information (if applicable)

---

**Remember**: Most issues can be resolved by:
1. Checking logs
2. Restarting services
3. Clearing cache
4. Verifying configuration

Happy troubleshooting! 🔧
