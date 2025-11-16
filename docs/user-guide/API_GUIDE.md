# API Guide

This guide explains how to use the rUv SPARC UI Dashboard REST API for programmatic access to tasks, projects, agents, and workflows.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Format](#requestresponse-format)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Code Examples](#code-examples)
8. [WebSocket API](#websocket-api)
9. [Best Practices](#best-practices)

---

## Getting Started

### Base URL

```
http://localhost:3001/api/v1
```

**Production**: Replace with your deployed URL (e.g., `https://api.ruv-sparc.io/api/v1`)

### API Versioning

The API is versioned via the URL path:
- **Current version**: `v1`
- **Deprecated versions**: Not yet applicable (first release)

### Content Type

All requests and responses use JSON:

```
Content-Type: application/json
```

---

## Authentication

The API uses **JWT (JSON Web Tokens)** for authentication.

### Obtaining a Token

#### Method 1: Login Endpoint

**Request**:
```bash
curl -X POST http://localhost:3001/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "your-password"
  }'
```

**Response**:
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": "user_123",
      "email": "admin@example.com",
      "name": "Admin User",
      "role": "admin"
    },
    "expiresIn": 3600
  }
}
```

**Token expires in 1 hour** (3600 seconds). Use the refresh token to obtain a new access token.

#### Method 2: API Key (Settings)

1. Log in to the web UI
2. Go to Settings → API Access
3. Click "Generate New Key"
4. Copy the API key (shown only once!)

**Using API Key**:
```bash
curl http://localhost:3001/api/v1/tasks \
  -H "X-API-Key: your-api-key-here"
```

### Using the Token

Include the JWT token in the `Authorization` header:

```bash
curl http://localhost:3001/api/v1/tasks \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Refreshing Tokens

When your access token expires, use the refresh token:

**Request**:
```bash
curl -X POST http://localhost:3001/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'
```

**Response**:
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expiresIn": 3600
  }
}
```

---

## API Endpoints

### Tasks

#### List Tasks

**Endpoint**: `GET /api/v1/tasks`

**Query Parameters**:
- `page` (integer): Page number (default: 1)
- `limit` (integer): Results per page (default: 20, max: 100)
- `priority` (string): Filter by priority (low, medium, high, critical)
- `status` (string): Filter by status (pending, in_progress, completed)
- `project_id` (string): Filter by project
- `tag` (string): Filter by tag
- `due_before` (ISO date): Tasks due before this date
- `due_after` (ISO date): Tasks due after this date
- `sort` (string): Sort field (created_at, due_date, priority)
- `order` (string): Sort order (asc, desc)

**Example**:
```bash
curl "http://localhost:3001/api/v1/tasks?priority=high&status=pending&limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "tasks": [
      {
        "id": "task_abc123",
        "title": "Review API documentation",
        "description": "Ensure all endpoints are documented",
        "priority": "high",
        "status": "pending",
        "due_date": "2025-01-15T14:00:00Z",
        "project_id": "project_xyz789",
        "tags": ["documentation", "review"],
        "created_at": "2025-01-08T10:00:00Z",
        "updated_at": "2025-01-08T10:00:00Z",
        "created_by": "user_123"
      }
    ],
    "pagination": {
      "current_page": 1,
      "total_pages": 5,
      "total_items": 87,
      "per_page": 20
    }
  }
}
```

#### Get Task by ID

**Endpoint**: `GET /api/v1/tasks/:id`

**Example**:
```bash
curl http://localhost:3001/api/v1/tasks/task_abc123 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "task_abc123",
    "title": "Review API documentation",
    "description": "Ensure all endpoints are documented",
    "priority": "high",
    "status": "pending",
    "due_date": "2025-01-15T14:00:00Z",
    "project": {
      "id": "project_xyz789",
      "name": "Documentation Sprint"
    },
    "tags": ["documentation", "review"],
    "comments": [
      {
        "id": "comment_1",
        "text": "Started reviewing endpoints",
        "author": "user_456",
        "created_at": "2025-01-08T11:00:00Z"
      }
    ],
    "created_at": "2025-01-08T10:00:00Z",
    "updated_at": "2025-01-08T11:00:00Z"
  }
}
```

#### Create Task

**Endpoint**: `POST /api/v1/tasks`

**Request Body**:
```json
{
  "title": "Implement user authentication",
  "description": "Add JWT-based authentication to the API",
  "priority": "high",
  "status": "pending",
  "due_date": "2025-01-20T17:00:00Z",
  "project_id": "project_xyz789",
  "tags": ["backend", "security"],
  "estimated_duration": 7200
}
```

**Example**:
```bash
curl -X POST http://localhost:3001/api/v1/tasks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Implement user authentication",
    "description": "Add JWT-based authentication to the API",
    "priority": "high",
    "due_date": "2025-01-20T17:00:00Z"
  }'
```

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "task_new456",
    "title": "Implement user authentication",
    "description": "Add JWT-based authentication to the API",
    "priority": "high",
    "status": "pending",
    "due_date": "2025-01-20T17:00:00Z",
    "created_at": "2025-01-08T12:00:00Z",
    "updated_at": "2025-01-08T12:00:00Z"
  }
}
```

#### Update Task

**Endpoint**: `PATCH /api/v1/tasks/:id`

**Request Body** (partial update):
```json
{
  "status": "in_progress",
  "priority": "critical"
}
```

**Example**:
```bash
curl -X PATCH http://localhost:3001/api/v1/tasks/task_abc123 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "in_progress"
  }'
```

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "task_abc123",
    "title": "Review API documentation",
    "status": "in_progress",
    "updated_at": "2025-01-08T13:00:00Z"
  }
}
```

#### Delete Task

**Endpoint**: `DELETE /api/v1/tasks/:id`

**Example**:
```bash
curl -X DELETE http://localhost:3001/api/v1/tasks/task_abc123 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:
```json
{
  "success": true,
  "message": "Task deleted successfully"
}
```

### Projects

#### List Projects

**Endpoint**: `GET /api/v1/projects`

**Query Parameters**:
- `page` (integer): Page number
- `limit` (integer): Results per page
- `status` (string): active, archived, completed
- `sort` (string): name, created_at, updated_at
- `order` (string): asc, desc

**Example**:
```bash
curl http://localhost:3001/api/v1/projects \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "projects": [
      {
        "id": "project_xyz789",
        "name": "Documentation Sprint",
        "description": "Complete all API documentation",
        "status": "active",
        "color": "#3498db",
        "icon": "📚",
        "task_count": 15,
        "completed_tasks": 8,
        "progress": 53.3,
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-08T12:00:00Z"
      }
    ],
    "pagination": {
      "current_page": 1,
      "total_pages": 2,
      "total_items": 12
    }
  }
}
```

#### Get Project by ID

**Endpoint**: `GET /api/v1/projects/:id`

**Include tasks**:
```bash
curl "http://localhost:3001/api/v1/projects/project_xyz789?include=tasks" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "project_xyz789",
    "name": "Documentation Sprint",
    "description": "Complete all API documentation",
    "status": "active",
    "tasks": [
      {
        "id": "task_abc123",
        "title": "Review API documentation",
        "status": "in_progress"
      }
    ],
    "analytics": {
      "total_tasks": 15,
      "completed": 8,
      "in_progress": 5,
      "pending": 2,
      "progress_percentage": 53.3
    }
  }
}
```

#### Create Project

**Endpoint**: `POST /api/v1/projects`

**Request Body**:
```json
{
  "name": "Mobile App Redesign",
  "description": "Redesign mobile app UI/UX",
  "color": "#e74c3c",
  "icon": "📱",
  "tags": ["mobile", "design"]
}
```

**Example**:
```bash
curl -X POST http://localhost:3001/api/v1/projects \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Mobile App Redesign",
    "description": "Redesign mobile app UI/UX"
  }'
```

#### Update Project

**Endpoint**: `PATCH /api/v1/projects/:id`

**Example**:
```bash
curl -X PATCH http://localhost:3001/api/v1/projects/project_xyz789 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "completed"
  }'
```

#### Delete Project

**Endpoint**: `DELETE /api/v1/projects/:id`

**Query Parameters**:
- `cascade` (boolean): Delete associated tasks (default: false)

**Example**:
```bash
curl -X DELETE "http://localhost:3001/api/v1/projects/project_xyz789?cascade=true" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Agents

#### List Agents

**Endpoint**: `GET /api/v1/agents`

**Example**:
```bash
curl http://localhost:3001/api/v1/agents \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "agents": [
      {
        "id": "agent_coder_1",
        "name": "Coder Agent",
        "type": "coder",
        "status": "working",
        "current_task": {
          "id": "task_abc123",
          "title": "Implement authentication"
        },
        "uptime_seconds": 15732,
        "tasks_completed_today": 8,
        "success_rate": 0.95
      }
    ]
  }
}
```

#### Get Agent by ID

**Endpoint**: `GET /api/v1/agents/:id`

**Include logs**:
```bash
curl "http://localhost:3001/api/v1/agents/agent_coder_1?include=logs&log_limit=50" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "agent_coder_1",
    "name": "Coder Agent",
    "type": "coder",
    "status": "working",
    "logs": [
      {
        "timestamp": "2025-01-08T14:30:00Z",
        "level": "INFO",
        "message": "Started task: Implement authentication"
      }
    ],
    "metrics": {
      "tasks_total": 142,
      "tasks_successful": 135,
      "tasks_failed": 7,
      "average_duration_seconds": 324
    }
  }
}
```

### Workflows

#### List Workflows

**Endpoint**: `GET /api/v1/workflows`

**Example**:
```bash
curl http://localhost:3001/api/v1/workflows \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "workflows": [
      {
        "id": "workflow_feature_dev",
        "name": "Feature Development",
        "description": "Complete feature development workflow",
        "status": "running",
        "progress": 45,
        "started_at": "2025-01-08T10:00:00Z",
        "estimated_completion": "2025-01-08T16:00:00Z"
      }
    ]
  }
}
```

#### Trigger Workflow

**Endpoint**: `POST /api/v1/workflows/trigger`

**Request Body**:
```json
{
  "template": "feature_development",
  "parameters": {
    "project_id": "project_xyz789",
    "feature_name": "User authentication",
    "branch": "feature/auth"
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:3001/api/v1/workflows/trigger \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "template": "feature_development",
    "parameters": {
      "feature_name": "User authentication"
    }
  }'
```

**Response**:
```json
{
  "success": true,
  "data": {
    "workflow_id": "workflow_12345",
    "status": "running",
    "started_at": "2025-01-08T14:00:00Z",
    "estimated_completion": "2025-01-08T20:00:00Z"
  }
}
```

---

## Request/Response Format

### Standard Response Structure

**Success Response**:
```json
{
  "success": true,
  "data": {
    // Response data
  },
  "meta": {
    "timestamp": "2025-01-08T14:00:00Z",
    "request_id": "req_abc123"
  }
}
```

**Error Response**:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": [
      {
        "field": "due_date",
        "message": "must be a valid ISO 8601 date"
      }
    ]
  },
  "meta": {
    "timestamp": "2025-01-08T14:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### Date/Time Format

All dates use **ISO 8601 format** with UTC timezone:

```
2025-01-08T14:30:00Z
```

**Examples**:
- `2025-01-08T14:30:00Z` - January 8, 2025, 2:30 PM UTC
- `2025-12-31T23:59:59Z` - December 31, 2025, 11:59:59 PM UTC

### Pagination

**Request**:
```bash
curl "http://localhost:3001/api/v1/tasks?page=2&limit=20"
```

**Response**:
```json
{
  "success": true,
  "data": {
    "tasks": [ /* ... */ ],
    "pagination": {
      "current_page": 2,
      "total_pages": 10,
      "total_items": 187,
      "per_page": 20,
      "has_next": true,
      "has_previous": true
    }
  }
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request successful, no content to return |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `AUTHENTICATION_REQUIRED` | Missing authentication token |
| `INVALID_TOKEN` | Token is invalid or expired |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| `RESOURCE_NOT_FOUND` | Requested resource does not exist |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INTERNAL_ERROR` | Internal server error |

### Example Error Handling

**JavaScript**:
```javascript
try {
  const response = await fetch('http://localhost:3001/api/v1/tasks', {
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    }
  });

  const json = await response.json();

  if (!json.success) {
    console.error('API Error:', json.error.message);

    if (json.error.code === 'INVALID_TOKEN') {
      // Refresh token or redirect to login
    } else if (json.error.code === 'VALIDATION_ERROR') {
      // Show validation errors to user
      json.error.details.forEach(detail => {
        console.error(`${detail.field}: ${detail.message}`);
      });
    }
  } else {
    // Process successful response
    console.log('Tasks:', json.data.tasks);
  }
} catch (error) {
  console.error('Network error:', error);
}
```

---

## Rate Limiting

**Limits**:
- **Authenticated requests**: 1000 requests per hour
- **Unauthenticated requests**: 100 requests per hour

**Headers** (included in every response):
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 987
X-RateLimit-Reset: 1704724800
```

**When rate limit is exceeded**:

**Status**: 429 Too Many Requests

**Response**:
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please try again in 45 minutes.",
    "retry_after": 2700
  }
}
```

**Best Practices**:
- Cache responses when possible
- Use pagination to reduce request count
- Implement exponential backoff for retries
- Monitor `X-RateLimit-Remaining` header

---

## Code Examples

### JavaScript (Node.js with Axios)

**Installation**:
```bash
npm install axios
```

**Example**:
```javascript
const axios = require('axios');

const API_BASE_URL = 'http://localhost:3001/api/v1';
const API_TOKEN = 'your-jwt-token';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Authorization': `Bearer ${API_TOKEN}`,
    'Content-Type': 'application/json'
  }
});

// List tasks
async function listTasks() {
  try {
    const response = await api.get('/tasks', {
      params: {
        priority: 'high',
        status: 'pending',
        limit: 10
      }
    });

    console.log('Tasks:', response.data.data.tasks);
  } catch (error) {
    console.error('Error:', error.response.data.error);
  }
}

// Create task
async function createTask() {
  try {
    const response = await api.post('/tasks', {
      title: 'New task from API',
      description: 'Created via REST API',
      priority: 'high',
      due_date: '2025-01-15T14:00:00Z'
    });

    console.log('Created task:', response.data.data);
  } catch (error) {
    console.error('Error:', error.response.data.error);
  }
}

// Update task
async function updateTask(taskId) {
  try {
    const response = await api.patch(`/tasks/${taskId}`, {
      status: 'in_progress'
    });

    console.log('Updated task:', response.data.data);
  } catch (error) {
    console.error('Error:', error.response.data.error);
  }
}

// Delete task
async function deleteTask(taskId) {
  try {
    await api.delete(`/tasks/${taskId}`);
    console.log('Task deleted');
  } catch (error) {
    console.error('Error:', error.response.data.error);
  }
}

// Run examples
listTasks();
createTask();
```

### Python (with requests)

**Installation**:
```bash
pip install requests
```

**Example**:
```python
import requests
from datetime import datetime, timedelta

API_BASE_URL = 'http://localhost:3001/api/v1'
API_TOKEN = 'your-jwt-token'

headers = {
    'Authorization': f'Bearer {API_TOKEN}',
    'Content-Type': 'application/json'
}

# List tasks
def list_tasks():
    response = requests.get(
        f'{API_BASE_URL}/tasks',
        headers=headers,
        params={
            'priority': 'high',
            'status': 'pending',
            'limit': 10
        }
    )

    if response.status_code == 200:
        data = response.json()
        print('Tasks:', data['data']['tasks'])
    else:
        print('Error:', response.json()['error'])

# Create task
def create_task():
    due_date = (datetime.now() + timedelta(days=7)).isoformat() + 'Z'

    response = requests.post(
        f'{API_BASE_URL}/tasks',
        headers=headers,
        json={
            'title': 'New task from Python',
            'description': 'Created via REST API',
            'priority': 'high',
            'due_date': due_date
        }
    )

    if response.status_code == 201:
        data = response.json()
        print('Created task:', data['data'])
    else:
        print('Error:', response.json()['error'])

# Update task
def update_task(task_id):
    response = requests.patch(
        f'{API_BASE_URL}/tasks/{task_id}',
        headers=headers,
        json={'status': 'in_progress'}
    )

    if response.status_code == 200:
        print('Updated task:', response.json()['data'])
    else:
        print('Error:', response.json()['error'])

# Delete task
def delete_task(task_id):
    response = requests.delete(
        f'{API_BASE_URL}/tasks/{task_id}',
        headers=headers
    )

    if response.status_code == 200:
        print('Task deleted')
    else:
        print('Error:', response.json()['error'])

# Run examples
list_tasks()
create_task()
```

### cURL Examples

**Create task**:
```bash
curl -X POST http://localhost:3001/api/v1/tasks \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Deploy to production",
    "priority": "critical",
    "due_date": "2025-01-10T09:00:00Z",
    "tags": ["deployment", "production"]
  }'
```

**Update task status**:
```bash
curl -X PATCH http://localhost:3001/api/v1/tasks/task_abc123 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}'
```

**List high-priority tasks**:
```bash
curl "http://localhost:3001/api/v1/tasks?priority=high&status=pending" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## WebSocket API

For real-time updates, connect to the WebSocket server.

### Connection

**Endpoint**: `ws://localhost:3002`

**Authentication**: Send token in initial message

**JavaScript Example**:
```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3002');

ws.on('open', () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'your-jwt-token'
  }));
});

ws.on('message', (data) => {
  const message = JSON.parse(data);

  switch (message.type) {
    case 'auth_success':
      console.log('Authenticated');

      // Subscribe to task updates
      ws.send(JSON.stringify({
        type: 'subscribe',
        channel: 'tasks'
      }));
      break;

    case 'task_created':
      console.log('New task:', message.data);
      break;

    case 'task_updated':
      console.log('Task updated:', message.data);
      break;

    case 'agent_status':
      console.log('Agent status:', message.data);
      break;
  }
});

ws.on('error', (error) => {
  console.error('WebSocket error:', error);
});
```

### Available Channels

- `tasks`: Task creation, updates, deletions
- `projects`: Project changes
- `agents`: Agent status updates
- `workflows`: Workflow execution updates

### Message Types

**From client**:
- `auth`: Authenticate with token
- `subscribe`: Subscribe to channel
- `unsubscribe`: Unsubscribe from channel

**From server**:
- `auth_success`: Authentication successful
- `auth_error`: Authentication failed
- `task_created`: New task created
- `task_updated`: Task updated
- `task_deleted`: Task deleted
- `agent_status`: Agent status changed
- `workflow_started`: Workflow started
- `workflow_completed`: Workflow completed

---

## Best Practices

### 1. Use Pagination

Always paginate large result sets:

```javascript
async function getAllTasks() {
  let allTasks = [];
  let page = 1;
  let hasMore = true;

  while (hasMore) {
    const response = await api.get('/tasks', {
      params: { page, limit: 100 }
    });

    allTasks = allTasks.concat(response.data.data.tasks);
    hasMore = response.data.data.pagination.has_next;
    page++;
  }

  return allTasks;
}
```

### 2. Handle Rate Limits

Implement exponential backoff:

```javascript
async function fetchWithRetry(url, options, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url, options);

      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After') || Math.pow(2, i);
        await sleep(retryAfter * 1000);
        continue;
      }

      return response;
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await sleep(Math.pow(2, i) * 1000);
    }
  }
}
```

### 3. Cache Responses

Cache static or infrequently changing data:

```javascript
const cache = new Map();

async function getCachedTasks(cacheKey, ttl = 60000) {
  const cached = cache.get(cacheKey);

  if (cached && Date.now() - cached.timestamp < ttl) {
    return cached.data;
  }

  const response = await api.get('/tasks');
  const data = response.data.data.tasks;

  cache.set(cacheKey, {
    data,
    timestamp: Date.now()
  });

  return data;
}
```

### 4. Validate Before Sending

Validate data client-side before API calls:

```javascript
function validateTask(task) {
  const errors = [];

  if (!task.title || task.title.length < 3) {
    errors.push('Title must be at least 3 characters');
  }

  if (task.due_date && !isValidISO8601(task.due_date)) {
    errors.push('Invalid due date format');
  }

  if (!['low', 'medium', 'high', 'critical'].includes(task.priority)) {
    errors.push('Invalid priority');
  }

  return errors;
}
```

### 5. Use WebSockets for Real-Time Updates

Combine REST API with WebSockets:

```javascript
// Initial load via REST API
const tasks = await api.get('/tasks');

// Subscribe to real-time updates via WebSocket
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'tasks'
}));

ws.on('message', (data) => {
  const message = JSON.parse(data);

  if (message.type === 'task_updated') {
    // Update local state without full refresh
    updateTaskInState(message.data);
  }
});
```

---

## Support

For API-related questions:

- **Documentation**: This guide
- **API Reference**: https://api.ruv-sparc.io/docs (auto-generated)
- **GitHub Issues**: https://github.com/yourusername/ruv-sparc-ui-dashboard/issues
- **Discord**: #api-help channel

---

**Happy coding!** 🚀
