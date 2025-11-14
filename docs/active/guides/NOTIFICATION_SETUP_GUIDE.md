# Notification System - Quick Setup Guide

Complete setup guide for deploying the multi-channel notification system.

---

## 📋 Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Node.js 18+ (for frontend)
- SMTP server credentials
- SSL/TLS certificate (for push notifications)

---

## 🚀 Backend Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Generate VAPID Keys for Push Notifications

```bash
# Using Python
python -c "from pywebpush import webpush; keys = webpush.vapid_gen(); print(f'Public: {keys[\"public_key\"]}\nPrivate: {keys[\"private_key\"]}')"

# OR using Node.js web-push
npm install -g web-push
web-push generate-vapid-keys
```

**Save these keys** - you'll need them in `.env`

### 3. Configure Environment Variables

```bash
# Copy example file
cp .env.example .env

# Edit .env with your credentials
nano .env
```

**Required Configuration**:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ai_agent_scheduler

# Email (Gmail example)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-specific-password  # Not your Gmail password!
FROM_EMAIL=your-email@gmail.com
FROM_NAME=AI Agent Scheduler

# Push Notifications
VAPID_PUBLIC_KEY=<your-generated-public-key>
VAPID_PRIVATE_KEY=<your-generated-private-key>
ADMIN_EMAIL=admin@example.com

# Frontend
FRONTEND_URL=http://localhost:3000
```

**Gmail App Password Setup**:
1. Go to https://myaccount.google.com/security
2. Enable 2-Step Verification
3. Go to App Passwords (https://myaccount.google.com/apppasswords)
4. Generate password for "Mail"
5. Use this 16-character password in `SMTP_PASSWORD`

**Other SMTP Providers**:

```bash
# SendGrid
SMTP_HOST=smtp.sendgrid.net
SMTP_PORT=587
SMTP_USER=apikey
SMTP_PASSWORD=<your-sendgrid-api-key>

# Mailgun
SMTP_HOST=smtp.mailgun.org
SMTP_PORT=587
SMTP_USER=<your-mailgun-smtp-username>
SMTP_PASSWORD=<your-mailgun-smtp-password>

# AWS SES
SMTP_HOST=email-smtp.us-east-1.amazonaws.com
SMTP_PORT=587
SMTP_USER=<your-aws-smtp-username>
SMTP_PASSWORD=<your-aws-smtp-password>
```

### 4. Run Database Migrations

```bash
# Connect to PostgreSQL
psql -U postgres -d ai_agent_scheduler

# Run migration
\i backend/app/database/migrations/006_notification_tables.sql
```

**Verify Tables Created**:
```sql
\dt

-- Should show:
-- user_preferences
-- push_subscriptions
```

### 5. Start Backend Server

```bash
cd backend
python -m app.main

# OR with uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Verify Running**:
```bash
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "services": {
    "api": "running",
    "database": "connected",
    "scheduler": "active"
  }
}
```

---

## 🎨 Frontend Setup

### 1. Install Dependencies

```bash
cd frontend
npm install react-hot-toast
```

### 2. Register Service Worker

Add to `frontend/src/index.tsx`:

```typescript
// Register service worker for push notifications
if ('serviceWorker' in navigator) {
  navigator.serviceWorker
    .register('/service-worker.js')
    .then(registration => {
      console.log('Service Worker registered:', registration);
    })
    .catch(error => {
      console.error('Service Worker registration failed:', error);
    });
}
```

### 3. Add Toast Provider

Update `frontend/src/App.tsx`:

```tsx
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <div className="App">
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 5000,
          style: {
            background: '#363636',
            color: '#fff',
          },
          success: {
            duration: 5000,
            iconTheme: {
              primary: '#4CAF50',
              secondary: '#fff',
            },
          },
          error: {
            duration: 8000,
            iconTheme: {
              primary: '#f44336',
              secondary: '#fff',
            },
          },
        }}
      />

      {/* Your app content */}
    </div>
  );
}
```

### 4. Start Frontend

```bash
npm start
```

---

## ✅ Verification Steps

### 1. Test Email Notifications

```bash
curl -X POST http://localhost:8000/api/notifications/test \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"channel": "email"}'
```

**Expected**: Email sent to your configured address

### 2. Test Push Notifications

1. Navigate to http://localhost:3000/settings
2. Click "Enable" under Browser Push Notifications
3. Accept browser permission prompt
4. Click "Send Test Notification"

**Expected**: Browser notification appears

### 3. Test WebSocket Toasts

```bash
# Send via API
curl -X POST http://localhost:8000/api/notifications/test \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"channel": "websocket"}'
```

**Expected**: Toast notification appears in browser

### 4. Test All Channels

```bash
curl -X POST http://localhost:8000/api/notifications/test \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"channel": "all"}'
```

**Expected**: Email sent, push notification shown, and toast appears

---

## 🔧 Troubleshooting

### Email Not Sending

**Check SMTP credentials**:
```bash
# Test SMTP connection
python -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('your-email@gmail.com', 'your-app-password')
print('✅ SMTP connection successful')
server.quit()
"
```

**Common Issues**:
- ❌ Using Gmail password instead of App Password
- ❌ 2-Step Verification not enabled
- ❌ Firewall blocking port 587
- ❌ Wrong SMTP host/port

**Solution**: Generate new App Password and update `.env`

---

### Push Notifications Not Working

**Check HTTPS requirement**:
- Push notifications require HTTPS (except localhost)
- Use `ngrok` for testing: `ngrok http 3000`

**Check service worker registration**:
```javascript
// In browser console
navigator.serviceWorker.getRegistrations()
  .then(regs => console.log('Registrations:', regs));
```

**Check VAPID keys**:
```bash
# Verify keys are set
echo $VAPID_PUBLIC_KEY
echo $VAPID_PRIVATE_KEY

# Regenerate if needed
python -c "from pywebpush import webpush; print(webpush.vapid_gen())"
```

**Common Issues**:
- ❌ Service worker not registered
- ❌ VAPID keys not set or invalid
- ❌ HTTP instead of HTTPS
- ❌ Browser doesn't support Push API (Safari)

---

### WebSocket Not Connecting

**Check WebSocket endpoint**:
```javascript
// In browser console
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onopen = () => console.log('✅ Connected');
ws.onerror = (e) => console.error('❌ Error:', e);
```

**Common Issues**:
- ❌ Wrong WebSocket URL
- ❌ CORS not configured
- ❌ Authentication token missing
- ❌ Server not running

---

### Scheduler Not Running

**Check logs**:
```bash
# Should see:
# 🚀 Starting notification scheduler...
# ✅ Notification scheduler started
```

**Manually test scheduler**:
```bash
cd backend
python -m app.schedulers.notification_scheduler
```

**Common Issues**:
- ❌ `ENABLE_EMAIL_NOTIFICATIONS=false` in `.env`
- ❌ APScheduler not installed
- ❌ Database connection failed

---

## 📊 Monitoring

### Check Notification Logs

```bash
# Backend logs
tail -f backend/logs/app.log | grep notification

# Should see:
# ✅ Email sent successfully to user@example.com
# 📢 Notification sent to user 1: [success] Task Completed
# ✅ Push notification sent to 2 device(s)
```

### Database Queries

```sql
-- Check user preferences
SELECT * FROM user_preferences;

-- Check push subscriptions
SELECT user_id, COUNT(*) as device_count
FROM push_subscriptions
GROUP BY user_id;

-- Check tasks with reminders
SELECT id, name, next_run_at, reminder_sent
FROM tasks
WHERE status = 'active'
ORDER BY next_run_at;
```

---

## 🔐 Production Deployment

### Security Checklist

- [ ] Use environment-specific SMTP credentials
- [ ] Enable HTTPS for frontend (required for push)
- [ ] Rotate VAPID keys annually
- [ ] Set strong database passwords
- [ ] Configure CORS origins properly
- [ ] Enable rate limiting on API endpoints
- [ ] Use Redis for session management
- [ ] Set up monitoring (Sentry, Prometheus)

### Environment Variables for Production

```bash
# Production settings
DEBUG=false
RELOAD=false
WORKERS=4

# Security
SECRET_KEY=<random-64-char-string>
CORS_ORIGINS=https://yourdomain.com

# Monitoring
SENTRY_DSN=<your-sentry-dsn>
LOG_LEVEL=WARNING
```

### SSL/TLS for Push Notifications

```bash
# Using Let's Encrypt
certbot certonly --standalone -d yourdomain.com

# Update nginx config
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:3000;
    }

    location /api {
        proxy_pass http://localhost:8000;
    }

    location /ws {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 📈 Performance Optimization

### Email Sending

```python
# Use thread pool for concurrent sending
email_service = EmailService()
email_service.executor = ThreadPoolExecutor(max_workers=10)
```

### WebSocket Scaling

```bash
# Use Redis for multi-server WebSocket
pip install redis aioredis

# In app/websocket/connection_manager.py
from aioredis import Redis

redis = Redis(host='localhost', port=6379, decode_responses=True)
```

### Database Indexing

```sql
-- Already included in migration
CREATE INDEX idx_tasks_next_run_at ON tasks(next_run_at) WHERE status = 'active';
CREATE INDEX idx_push_subscriptions_user_id ON push_subscriptions(user_id);
```

---

## 🎉 Success Criteria

✅ Email notifications sent successfully
✅ Browser push notifications working
✅ WebSocket toasts appearing in real-time
✅ User preferences persisted
✅ Scheduler running automated tasks
✅ All tests passing

**Next**: See `NOTIFICATION_EXAMPLES.md` for usage examples

---

**Setup Time**: ~30 minutes
**Difficulty**: Medium
**Support**: See troubleshooting section above
