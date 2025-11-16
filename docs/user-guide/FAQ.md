# Frequently Asked Questions (FAQ)

Common questions and answers about the rUv SPARC UI Dashboard.

## Table of Contents

1. [General Questions](#general-questions)
2. [Getting Started](#getting-started)
3. [Features & Functionality](#features--functionality)
4. [Technical Questions](#technical-questions)
5. [Integrations](#integrations)
6. [Pricing & Licensing](#pricing--licensing)
7. [Security & Privacy](#security--privacy)
8. [Support & Community](#support--community)

---

## General Questions

### What is rUv SPARC UI Dashboard?

The rUv SPARC UI Dashboard is a comprehensive task management and workflow automation platform that combines:
- **Task Management**: Calendar-based task scheduling with drag-and-drop
- **Project Management**: Kanban boards, timelines, and analytics
- **AI Agent Workflows**: Multi-agent coordination for automated development tasks
- **Real-time Collaboration**: WebSocket-based live updates
- **Memory & Context**: Persistent memory system for intelligent decision-making

**Use Cases**:
- Software development teams tracking sprints
- Personal productivity and time management
- Research projects coordinating multiple agents
- Any workflow requiring task automation and AI assistance

---

### Who should use this dashboard?

**Ideal for**:
- **Developers**: Track coding tasks, integrate with GitHub, automate workflows
- **Project Managers**: Visualize project progress, manage teams, generate reports
- **Researchers**: Coordinate AI agents for research tasks, document processes
- **Individuals**: Personal task management with intelligent scheduling
- **Teams**: Collaborative task boards with real-time updates

**Not ideal for**:
- Enterprise-scale teams (>100 users) without modification
- Non-technical users who don't need AI agent features (simpler tools may suffice)

---

### How is this different from other task managers?

**Unique Features**:
1. **AI Agent Integration**: Built-in multi-agent system for automation (unique)
2. **Memory System**: Persistent context across sessions (rare)
3. **Real-time Workflow Visualization**: See agent activity in real-time (unique)
4. **Developer-First**: API-first design, extensive customization (rare)
5. **Open Source**: Self-hosted, full control over data (uncommon in modern tools)

**vs. Trello/Asana**:
- More technical, developer-focused
- AI agent workflows (they have manual automation)
- Self-hosted option

**vs. Jira**:
- Simpler, faster setup
- AI-native features
- Better for small-to-medium teams

**vs. Notion**:
- More structured for tasks/projects
- Real-time agent monitoring
- Less flexible for general note-taking

---

### Is this open source?

**Yes!** The rUv SPARC UI Dashboard is open source under the MIT License.

**What this means**:
- ✅ Free to use, modify, distribute
- ✅ Commercial use allowed
- ✅ Self-host on your own infrastructure
- ✅ Contribute to development
- ✅ Fork and customize

**Repository**: https://github.com/yourusername/ruv-sparc-ui-dashboard

**Contributing**: We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Getting Started

### Do I need to know how to code?

**Basic usage**: No coding required
- Web UI for task/project management
- Point-and-click interface
- Pre-built workflows

**Advanced usage**: Some technical knowledge helpful
- API integration requires JavaScript/Python
- Custom agent workflows require configuration
- Self-hosting requires Docker knowledge

**Learning Path**:
1. Start with web UI (no code)
2. Explore API examples (basic scripting)
3. Customize agents (intermediate)
4. Build custom integrations (advanced)

---

### Can I use this without Docker?

**Short answer**: Not recommended, but possible.

**Docker provides**:
- Easy setup (one command)
- Consistent environment
- Isolated services
- Simple updates

**Without Docker** (manual setup):
1. Install PostgreSQL 15+ manually
2. Install Redis 7+ manually
3. Configure connection strings
4. Manage service startup

**We recommend Docker** for 99% of users. Only skip if you have specific infrastructure requirements.

---

### How do I upgrade to a new version?

**Using Docker** (recommended):

```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker-compose down
docker-compose build
docker-compose up -d

# Run migrations
cd backend
npm run migrate
```

**Manual setup**:

```bash
# Pull latest changes
git pull origin main

# Update backend dependencies
cd backend
npm install
npm run migrate

# Update frontend dependencies
cd ../frontend
npm install
npm run build
```

**Backup first!**
```bash
# Backup database
docker exec ruv-sparc-postgres pg_dump -U postgres ruv_sparc > backup.sql

# Or use built-in backup script
npm run backup
```

---

### Can I import my existing tasks from other tools?

**Yes!** We support import from:

**Trello**:
```bash
cd backend
npm run import -- --source trello --file ~/Downloads/trello-export.json
```

**Asana** (via CSV):
```bash
npm run import -- --source csv --file tasks.csv --mapping asana
```

**Todoist**:
```bash
npm run import -- --source todoist --api-key YOUR_API_KEY
```

**Generic CSV**:
```bash
# Format: title, description, due_date, priority, tags
npm run import -- --source csv --file tasks.csv
```

**Custom JSON**:
```bash
# See docs/IMPORT_FORMAT.md for schema
npm run import -- --source json --file tasks.json
```

**Export to other tools**:
```bash
# Export as CSV, JSON, or ICS (calendar)
npm run export -- --format csv --output tasks.csv
```

---

## Features & Functionality

### How do I create recurring tasks?

**Via Web UI**:
1. Create or edit a task
2. Enable "Recurring" toggle
3. Configure pattern:
   - **Daily**: Every X days (e.g., every 2 days)
   - **Weekly**: Specific days (e.g., Mon, Wed, Fri)
   - **Monthly**: Day of month or position (e.g., last Friday)
   - **Yearly**: Specific date each year
4. Set end date (or "Never")

**Via API**:
```bash
curl -X POST http://localhost:3001/api/v1/tasks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Weekly team meeting",
    "recurrence": {
      "frequency": "weekly",
      "interval": 1,
      "daysOfWeek": [1, 3, 5],
      "endDate": "2025-12-31T23:59:59Z"
    }
  }'
```

**Examples**:
- Daily standup (weekdays only)
- Monthly review (last Friday)
- Quarterly planning (every 3 months)
- Annual performance review

---

### Can I assign tasks to other people?

**Multi-user mode** (requires setup):

**1. Enable multi-user mode** (`.env`):
```env
MULTI_USER_ENABLED=true
```

**2. Invite users**:
- Settings → Users → Invite User
- Enter email address
- User receives invitation link

**3. Assign tasks**:
- Create/edit task
- Select "Assignee" from dropdown
- Or use `@mention` in task title

**Via API**:
```bash
curl -X PATCH http://localhost:3001/api/v1/tasks/task_abc123 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "assignee_id": "user_456"
  }'
```

**Single-user mode** (default):
- Assignee field not shown
- All tasks belong to logged-in user
- Simpler for personal use

---

### What are agents and how do they work?

**Agents** are autonomous AI workers that execute tasks based on workflows.

**Types of Agents**:
- 🧠 **Planner**: Breaks projects into tasks
- 💻 **Coder**: Generates code implementations
- ✅ **Tester**: Creates and runs tests
- 🔍 **Reviewer**: Reviews code quality
- 🏗️ **Architect**: Designs system architecture
- 📊 **Analyst**: Analyzes data and metrics
- 📝 **Technical Writer**: Creates documentation

**How it works**:
1. You trigger a workflow (e.g., "Feature Development")
2. System spawns required agents (Planner → Coder → Tester → Reviewer)
3. Agents communicate via Memory MCP (shared context)
4. Each agent completes its task and passes to next
5. You monitor progress in Agent Monitor

**Example Workflow**:
```
User: "Build user authentication feature"
  ↓
Planner: Creates task breakdown
  ↓
Coder: Implements auth API
  ↓
Tester: Writes and runs tests
  ↓
Reviewer: Reviews code quality
  ↓
Complete: Feature ready for deployment
```

**Customization**:
- Configure agent behavior in `backend/config/agents.js`
- Add custom agents (requires development)
- Adjust timeouts and resource limits

---

### Can I use this offline?

**Partial offline support**:

**What works offline**:
- View previously loaded tasks/projects
- Edit cached data (syncs when online)
- Use PWA (Progressive Web App) on mobile

**What doesn't work offline**:
- Create new tasks (requires API)
- Real-time updates
- Agent workflows (require backend)
- Image/attachment uploads

**Enabling offline mode**:
1. Open app in browser (Chrome/Edge/Firefox)
2. Browser prompts to "Add to Home Screen" (PWA)
3. Click "Add"
4. Use app icon to launch

**Sync behavior**:
- Changes queued while offline
- Automatically sync when reconnected
- Conflicts resolved (last-write-wins or manual)

---

### How do I export my data?

**Via Web UI**:
1. Settings → Data & Privacy → Export Data
2. Select what to export:
   - [ ] Tasks
   - [ ] Projects
   - [ ] Agents & Workflows
   - [ ] User settings
3. Choose format: JSON, CSV, or ZIP
4. Click "Download Export"

**Via API**:
```bash
# Export all tasks as JSON
curl http://localhost:3001/api/v1/export/tasks?format=json \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o tasks.json

# Export all projects as CSV
curl http://localhost:3001/api/v1/export/projects?format=csv \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o projects.csv
```

**Via CLI**:
```bash
cd backend
npm run export -- --format json --output ~/exports/data.json
```

**Scheduled exports** (backup automation):
```bash
# Add to cron (Linux/macOS)
0 2 * * * cd /path/to/backend && npm run backup

# Or use Windows Task Scheduler
```

---

## Technical Questions

### What technologies is this built with?

**Frontend**:
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Redux Toolkit**: State management
- **React Router**: Navigation
- **FullCalendar**: Calendar component
- **React Beautiful DnD**: Drag-and-drop
- **Recharts**: Analytics charts
- **Tailwind CSS**: Styling

**Backend**:
- **Node.js 18+**: Runtime
- **Express**: Web framework
- **PostgreSQL 15**: Database
- **Redis 7**: Caching
- **Socket.io**: WebSockets
- **JWT**: Authentication
- **Prisma**: Database ORM

**AI/Agents**:
- **Claude Flow**: Agent orchestration
- **Memory MCP**: Persistent memory
- **Anthropic Claude**: AI models

**Infrastructure**:
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **GitHub Actions**: CI/CD

---

### What are the system requirements?

**Minimum** (for testing):
- **RAM**: 4 GB
- **CPU**: Dual-core processor
- **Storage**: 2 GB free space
- **OS**: Windows 10, macOS 12, Ubuntu 20.04

**Recommended** (for development):
- **RAM**: 8 GB
- **CPU**: Quad-core processor
- **Storage**: 10 GB free space (for Docker images)
- **OS**: Latest Windows 11, macOS 14, Ubuntu 22.04

**Production** (for teams):
- **RAM**: 16 GB
- **CPU**: 8+ cores
- **Storage**: 50 GB+ (depending on data)
- **OS**: Ubuntu 22.04 LTS (recommended)

**Browser Support**:
- Chrome 100+
- Firefox 100+
- Edge 100+
- Safari 15+ (limited testing)

---

### How scalable is this?

**Current Scale**:
- **Users**: 1-50 users (tested)
- **Tasks**: 10,000+ tasks
- **Projects**: 100+ projects
- **Agents**: 10 concurrent agents

**Performance**:
- Sub-100ms API response times (typical)
- Real-time updates (<1s latency)
- Handles 100 req/sec (single server)

**Scaling Up**:

**Vertical (single server)**:
- Add RAM/CPU to existing server
- Optimize database queries
- Enable Redis caching
- Good for up to ~100 users

**Horizontal (multiple servers)**:
- Load balancer (Nginx/HAProxy)
- Multiple backend instances
- PostgreSQL replication
- Redis cluster
- Shared storage (S3/NFS)
- Good for 100+ users

**Database Optimization**:
```sql
-- Add indexes for performance
CREATE INDEX idx_tasks_user_due ON tasks(user_id, due_date);
CREATE INDEX idx_tasks_project_status ON tasks(project_id, status);

-- Partition large tables
CREATE TABLE tasks_2025 PARTITION OF tasks
  FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

---

### Can I self-host on my own server?

**Yes!** Self-hosting is fully supported.

**Deployment Options**:

**1. Docker (easiest)**:
```bash
# On your server
git clone https://github.com/yourusername/ruv-sparc-ui-dashboard.git
cd ruv-sparc-ui-dashboard
cp .env.example .env
# Edit .env with production values
docker-compose -f docker-compose.prod.yml up -d
```

**2. Cloud Platforms**:
- **AWS**: ECS/Fargate, RDS, ElastiCache
- **Google Cloud**: Cloud Run, Cloud SQL, Memorystore
- **Azure**: Container Instances, PostgreSQL, Redis Cache
- **DigitalOcean**: App Platform, Managed PostgreSQL

**3. VPS (Linux)**:
```bash
# Install dependencies
sudo apt update
sudo apt install -y docker docker-compose

# Deploy application
git clone ...
cd ruv-sparc-ui-dashboard
docker-compose up -d
```

**Production Checklist**:
- [ ] Change default passwords/secrets
- [ ] Enable HTTPS (Let's Encrypt)
- [ ] Configure firewall (UFW/iptables)
- [ ] Set up backups (automated)
- [ ] Configure monitoring (Prometheus/Grafana)
- [ ] Enable rate limiting
- [ ] Review security settings

**Cost Estimates** (monthly):
- **Small** (1-10 users): $10-30 (VPS)
- **Medium** (10-50 users): $50-100 (cloud platform)
- **Large** (50+ users): $200+ (dedicated infrastructure)

---

### How do I contribute to development?

**We welcome contributions!**

**Ways to Contribute**:
1. **Report Bugs**: GitHub Issues
2. **Suggest Features**: GitHub Discussions
3. **Fix Bugs**: Submit pull requests
4. **Add Features**: Submit pull requests
5. **Improve Docs**: Update documentation
6. **Answer Questions**: Help others on Discord

**Development Setup**:
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ruv-sparc-ui-dashboard.git
cd ruv-sparc-ui-dashboard

# Install dependencies
npm install

# Create feature branch
git checkout -b feature/my-new-feature

# Make changes and test
npm test

# Commit and push
git commit -m "Add new feature"
git push origin feature/my-new-feature

# Create pull request on GitHub
```

**Contribution Guidelines**:
- Follow code style (ESLint config)
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive
- Discuss major changes in issues first

**See**: [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## Integrations

### Does this integrate with GitHub?

**Yes!** Full GitHub integration included.

**Features**:
- Link tasks to GitHub issues
- Sync task status with issue status
- Create tasks from issues
- Link pull requests to tasks
- View PR status in dashboard

**Setup**:
1. Settings → Integrations → GitHub
2. Click "Connect GitHub"
3. Authorize application (requires `repo` scope)
4. Select repositories to sync

**Usage**:
```
Task title: "Fix login bug (#123)"
→ Automatically links to github.com/owner/repo/issues/123

When issue closes on GitHub:
→ Task marked complete in dashboard
```

**API Integration**:
```bash
# Link task to GitHub issue
curl -X PATCH http://localhost:3001/api/v1/tasks/task_abc123 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "github_issue": "https://github.com/owner/repo/issues/123"
  }'
```

---

### Can I sync with Google Calendar?

**Yes!** Google Calendar sync is available.

**Setup**:
1. Settings → Integrations → Google Calendar
2. Click "Connect Google Account"
3. Authorize application
4. Select calendar to sync with

**Sync Options**:
- **One-way** (Dashboard → Google): Tasks appear in Google Calendar
- **Two-way** (bidirectional): Changes sync both ways

**What syncs**:
- Task title → Event title
- Due date → Event start time
- Description → Event description
- Priority → Event color

**Limitations**:
- Google Calendar doesn't support task status (only events)
- Recurring tasks sync as individual events
- Sync delay: ~5 minutes

**Disable sync**:
- Settings → Integrations → Google Calendar → Disconnect

---

### What other integrations are supported?

**Current Integrations**:
- ✅ **GitHub**: Issues, PRs, repositories
- ✅ **Google Calendar**: Task/event sync
- ✅ **Slack**: Notifications and commands
- ✅ **Zapier**: Custom automations (1000+ apps)

**Coming Soon**:
- 🚧 **GitLab**: Issue tracking
- 🚧 **Jira**: Import/export
- 🚧 **Microsoft Teams**: Notifications
- 🚧 **Notion**: Database sync

**Build Your Own**:
Use the API to integrate with any tool:

```javascript
// Example: Slack notification on task complete
app.post('/webhooks/task-completed', async (req, res) => {
  const { task } = req.body;

  await fetch('https://hooks.slack.com/services/YOUR/WEBHOOK/URL', {
    method: 'POST',
    body: JSON.stringify({
      text: `✅ Task completed: ${task.title}`
    })
  });

  res.sendStatus(200);
});
```

---

## Pricing & Licensing

### Is this really free?

**Yes, 100% free and open source!**

**No hidden costs**:
- ✅ Free to download
- ✅ Free to self-host
- ✅ Free for commercial use
- ✅ No user limits
- ✅ No feature restrictions

**Optional Costs**:
- Server hosting (if self-hosting): ~$10-30/month
- Cloud services (optional): Varies
- Third-party integrations (optional): Varies (e.g., Zapier plans)

**Support Options**:
- **Community Support**: Free (Discord, GitHub)
- **Priority Support**: Paid (contact sales@ruv-sparc.io)
- **Custom Development**: Paid (contact sales@ruv-sparc.io)

---

### Can I use this for my business?

**Yes!** The MIT License allows commercial use.

**Permitted**:
- ✅ Use in your company
- ✅ Sell access to hosted version
- ✅ Integrate into commercial products
- ✅ Modify and distribute
- ✅ Close-source your modifications (if desired)

**Requirements**:
- Include MIT License in distributions
- No warranty implied

**Example Commercial Uses**:
- SaaS offering (host for clients)
- Internal tool for your team
- White-label solution for customers
- Part of larger commercial product

**No Attribution Required** (beyond license inclusion).

---

## Security & Privacy

### How is my data stored?

**Local Storage** (self-hosted):
- **Database**: PostgreSQL 15 (encrypted at rest optional)
- **Cache**: Redis (in-memory, ephemeral)
- **Files**: Local filesystem or S3-compatible storage

**Data Locations**:
- Tasks, projects: PostgreSQL database
- Real-time state: Redis cache
- Agent memories: Memory MCP (local or remote)
- Uploaded files: `/uploads` directory or S3

**Database Backups**:
- Automated daily backups (configurable)
- Stored in `~/ruv-sparc-backups/`
- Retention: 30 days (configurable)

**Deletion**:
- Soft delete (recoverable for 30 days)
- Hard delete (permanent) via admin panel

---

### Is my data secure?

**Security Measures**:

**1. Authentication**:
- JWT tokens (HMAC SHA256)
- Bcrypt password hashing
- Optional 2FA (TOTP)
- API key authentication

**2. Authorization**:
- Role-based access control (RBAC)
- Per-resource permissions
- API rate limiting (1000 req/hour)

**3. Data Protection**:
- HTTPS required in production (TLS 1.3)
- SQL injection prevention (parameterized queries)
- XSS protection (React escaping)
- CSRF protection (tokens)

**4. Infrastructure**:
- Docker container isolation
- Network segmentation
- Firewall rules (UFW/iptables)

**5. Monitoring**:
- Failed login attempts logged
- Suspicious activity alerts
- Audit logs (who did what, when)

**Best Practices**:
```env
# Strong secrets (never commit to git!)
JWT_SECRET=$(openssl rand -base64 32)
SESSION_SECRET=$(openssl rand -base64 32)

# Secure database password
POSTGRES_PASSWORD=$(openssl rand -base64 16)
```

**Security Audits**:
- Regular dependency updates (Dependabot)
- OWASP Top 10 compliance
- Penetration testing (community-driven)

---

### Do you collect any analytics?

**Default**: No analytics collected

**Optional Analytics**:
```env
# Opt-in analytics (disabled by default)
ENABLE_ANALYTICS=false
```

**If enabled, we collect**:
- Usage statistics (page views, feature usage)
- Performance metrics (load times, errors)
- No personal information (anonymized)

**You control your data**:
- Self-hosted = full control
- No phone-home unless explicitly enabled
- No third-party trackers (Google Analytics, etc.)

**Transparency**:
- All analytics code is open source
- Review in `backend/analytics/` directory

---

## Support & Community

### How do I get help?

**1. Documentation** (start here):
- **Installation Guide**: [INSTALL.md](INSTALL.md)
- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md)
- **API Guide**: [API_GUIDE.md](API_GUIDE.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **FAQ**: This document

**2. Search GitHub Issues**:
- https://github.com/yourusername/ruv-sparc-ui-dashboard/issues
- Someone may have solved your problem

**3. Ask the Community**:
- **Discord**: https://discord.gg/ruv-sparc (fastest response)
- **GitHub Discussions**: For feature requests, design discussions

**4. Create a GitHub Issue**:
- If you found a bug or need a feature
- Include detailed information (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md#getting-help))

**5. Email Support** (for private issues):
- support@ruv-sparc.io
- Response time: 1-3 business days

---

### Is there a community?

**Yes! Join us:**

**Discord** (most active):
- **#general**: General discussion
- **#help**: Troubleshooting and questions
- **#showcase**: Share your setups
- **#development**: Development discussion
- **#announcements**: Updates and releases

**GitHub**:
- **Issues**: Bug reports, feature requests
- **Discussions**: Design discussions, Q&A
- **Pull Requests**: Code contributions

**Social Media**:
- **Twitter**: @ruvSPARC (updates, tips)
- **Reddit**: r/ruvSPARC (community discussions)

**Events**:
- Monthly community calls (last Friday of month)
- Quarterly roadmap reviews
- Annual conference (virtual)

---

### Can I request features?

**Absolutely!**

**How to Request**:
1. **Search first**: Check if already requested
   - GitHub Issues (open and closed)
   - GitHub Discussions
2. **Create GitHub Discussion**:
   - https://github.com/yourusername/ruv-sparc-ui-dashboard/discussions/new
   - Category: "Feature Requests"
3. **Describe the feature**:
   - **Problem**: What are you trying to solve?
   - **Solution**: How would the feature work?
   - **Alternatives**: What workarounds exist?
   - **Context**: Screenshots, examples

**Feature Request Template**:
```markdown
## Feature Request

**Is your feature request related to a problem?**
I'm always frustrated when [...]

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Any alternative solutions or features you've considered.

**Additional context**
Screenshots, mockups, examples from other tools.
```

**What Happens Next**:
1. Community discussion and feedback
2. Maintainers review feasibility
3. Prioritization (based on votes, impact, effort)
4. Implementation (by maintainers or community)

**Popular requests get priority!** Upvote (👍) features you want.

---

### Where can I see the roadmap?

**Public Roadmap**:
- https://github.com/yourusername/ruv-sparc-ui-dashboard/projects/1

**Current Focus** (Q1 2025):
- 🚧 Mobile app (React Native)
- 🚧 Offline mode improvements
- 🚧 GitLab integration
- 🚧 Advanced analytics

**Future Plans** (Q2-Q4 2025):
- AI-powered task suggestions
- Natural language task creation
- Team collaboration features
- Enterprise SSO support

**How You Can Influence**:
- Vote on feature requests (👍 reactions)
- Contribute code for priority features
- Sponsor development (GitHub Sponsors)

---

**Still have questions?**

- Join our [Discord](https://discord.gg/ruv-sparc)
- Create a [GitHub Discussion](https://github.com/yourusername/ruv-sparc-ui-dashboard/discussions/new)
- Email us: support@ruv-sparc.io

**Happy tasking!** 📋✨
