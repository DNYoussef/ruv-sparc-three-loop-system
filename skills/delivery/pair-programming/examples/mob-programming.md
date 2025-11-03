# Mob Programming Example
Collaborative mob programming session with rotating roles and multiple specialists

## Scenario: Building Real-Time Chat Feature

This example demonstrates mob programming (ensemble programming) where multiple AI specialists rotate through the driver role while the team navigates together.

---

## Mob Configuration

```bash
# Start mob programming session with multiple AI specialists
claude-flow pair --start \
  --mode mob \
  --specialists "backend-dev,frontend-dev,security-expert,test-engineer" \
  --rotation 7m \
  --focus implementation \
  --verify \
  --test \
  --continuous-review
```

**Mob Setup:**
- **Driver Rotation**: Every 7 minutes
- **Navigator**: You (constant strategic guidance)
- **Specialists**:
  - Backend Developer (Node.js/WebSocket)
  - Frontend Developer (React)
  - Security Expert (Auth/Encryption)
  - Test Engineer (Testing strategy)

**Workflow:**
1. Plan together as a mob
2. Rotate drivers every 7 minutes
3. Navigator provides continuous direction
4. Each specialist brings domain expertise
5. Continuous integration and testing

---

## Phase 1: Mob Planning (All Participants)

### Initial Discussion

**You (Navigator):**

```
We need to build a real-time chat feature with:
- User authentication
- WebSocket connections
- Message persistence
- Typing indicators
- Read receipts
- Security (encryption, rate limiting)

Let's break this down and decide our approach.
```

**Backend Developer:**

```
🔧 BACKEND PERSPECTIVE:

Architecture Recommendation:
- Socket.IO for WebSocket handling
- Redis for pub/sub and presence tracking
- MongoDB for message persistence
- JWT for authentication
- Rate limiting with express-rate-limit

Key Concerns:
- Scalability (multiple server instances)
- Connection handling
- Message delivery guarantees
```

**Frontend Developer:**

```
💻 FRONTEND PERSPECTIVE:

Tech Stack Recommendation:
- React with hooks
- Socket.IO client
- Context API for chat state
- Optimistic UI updates
- Local message caching

Key Concerns:
- Real-time state synchronization
- Reconnection handling
- Offline support
- UX for typing indicators
```

**Security Expert:**

```
🔒 SECURITY PERSPECTIVE:

Critical Requirements:
- End-to-end encryption for messages
- Rate limiting on all endpoints
- Input sanitization (XSS prevention)
- CSRF protection
- Secure WebSocket (WSS)
- Message validation

Threats to Mitigate:
- Message injection
- DoS attacks
- Man-in-the-middle
- Data leakage
```

**Test Engineer:**

```
🧪 TESTING PERSPECTIVE:

Testing Strategy:
- Unit tests for message handling
- Integration tests for WebSocket flow
- E2E tests for chat scenarios
- Load testing for scalability
- Security testing

Coverage Goals:
- 90%+ code coverage
- All security paths tested
- Connection edge cases covered
```

### Mob Decision

**Agreed Architecture:**

```
┌─────────────────┐
│   React Client  │
│  (Socket.IO)    │
└────────┬────────┘
         │ WSS
         ▼
┌─────────────────┐
│  Node.js Server │
│  (Socket.IO)    │
└────┬────────┬───┘
     │        │
     ▼        ▼
┌────────┐ ┌──────┐
│ Redis  │ │ Mongo│
│ Pub/Sub│ │  DB  │
└────────┘ └──────┘
```

**Implementation Order:**
1. Backend: Socket.IO server + authentication
2. Backend: Message handling and persistence
3. Frontend: Socket connection and UI
4. Security: Encryption and rate limiting
5. Testing: Comprehensive test suite
6. Integration: Connect all pieces

---

## Rotation 1: Backend Developer as Driver (0-7 min)

### Driver: Backend Developer
### Navigator: You + Other Specialists

**Your Instructions:**

```
Let's start with the Socket.IO server setup and authentication.
Create the basic server structure with:
- Socket.IO server initialization
- JWT authentication middleware
- Connection/disconnection handling
- Basic event structure
```

**Backend Developer (Driver) Implements:**

```javascript
// server/socket-server.js

const socketIO = require('socket.io');
const jwt = require('jsonwebtoken');
const { createAdapter } = require('@socket.io/redis-adapter');
const { createClient } = require('redis');

class ChatSocketServer {
  constructor(httpServer) {
    this.io = socketIO(httpServer, {
      cors: {
        origin: process.env.CLIENT_URL,
        credentials: true
      },
      transports: ['websocket', 'polling']
    });

    this.setupRedisAdapter();
    this.setupMiddleware();
    this.setupEventHandlers();
  }

  async setupRedisAdapter() {
    // For horizontal scaling
    const pubClient = createClient({ url: process.env.REDIS_URL });
    const subClient = pubClient.duplicate();

    await Promise.all([
      pubClient.connect(),
      subClient.connect()
    ]);

    this.io.adapter(createAdapter(pubClient, subClient));
  }

  setupMiddleware() {
    // JWT authentication middleware
    this.io.use(async (socket, next) => {
      try {
        const token = socket.handshake.auth.token;

        if (!token) {
          return next(new Error('Authentication required'));
        }

        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        socket.userId = decoded.userId;
        socket.username = decoded.username;

        next();
      } catch (error) {
        next(new Error('Invalid token'));
      }
    });
  }

  setupEventHandlers() {
    this.io.on('connection', (socket) => {
      console.log(`User connected: ${socket.username}`);

      this.handleJoinRoom(socket);
      this.handleSendMessage(socket);
      this.handleTyping(socket);
      this.handleDisconnect(socket);
    });
  }

  handleJoinRoom(socket) {
    socket.on('join-room', async (roomId) => {
      socket.join(roomId);
      socket.currentRoom = roomId;

      // Notify room members
      socket.to(roomId).emit('user-joined', {
        userId: socket.userId,
        username: socket.username
      });

      // Send room history
      const messages = await this.getMessageHistory(roomId);
      socket.emit('room-history', messages);
    });
  }

  handleSendMessage(socket) {
    socket.on('send-message', async (data) => {
      const message = {
        id: generateMessageId(),
        roomId: socket.currentRoom,
        userId: socket.userId,
        username: socket.username,
        text: sanitize(data.text),
        timestamp: Date.now()
      };

      // Save to database
      await this.saveMessage(message);

      // Broadcast to room
      this.io.to(socket.currentRoom).emit('new-message', message);
    });
  }

  handleTyping(socket) {
    socket.on('typing-start', () => {
      socket.to(socket.currentRoom).emit('user-typing', {
        userId: socket.userId,
        username: socket.username
      });
    });

    socket.on('typing-stop', () => {
      socket.to(socket.currentRoom).emit('user-stopped-typing', {
        userId: socket.userId
      });
    });
  }

  handleDisconnect(socket) {
    socket.on('disconnect', () => {
      console.log(`User disconnected: ${socket.username}`);

      if (socket.currentRoom) {
        socket.to(socket.currentRoom).emit('user-left', {
          userId: socket.userId,
          username: socket.username
        });
      }
    });
  }

  async getMessageHistory(roomId) {
    // TODO: Implement database query
    return [];
  }

  async saveMessage(message) {
    // TODO: Implement database save
  }
}

module.exports = ChatSocketServer;
```

**Security Expert (Navigator Feedback):**

```
🔒 SECURITY REVIEW:

✅ Good:
- JWT authentication on connection
- Input sanitization mentioned
- CORS configuration

⚠️  Needs Attention:
1. Add rate limiting per socket
2. Validate message length (prevent DoS)
3. Escape user input before broadcasting
4. Add room access control (authorization)
5. Implement message encryption

Priority: Add rate limiting next rotation.
```

**Test Engineer (Navigator Feedback):**

```
🧪 TESTING NOTES:

Need to test:
- Connection with valid/invalid tokens
- Multiple users in same room
- Message broadcasting
- Disconnection handling
- Room join/leave scenarios

Suggest: Create mock Socket.IO for testing
```

---

## Rotation 2: Frontend Developer as Driver (7-14 min)

### ⏰ Role Switch

**Handoff Summary:**

```
╔════════════════════════════════════════╗
║       ROLE HANDOFF - Rotation 2       ║
╚════════════════════════════════════════╝

Completed: Socket.IO server with authentication
Next: React chat UI with Socket.IO client

Files Created:
- server/socket-server.js (+120 lines)

Context:
- Server handles authentication, rooms, messages
- Need frontend to connect and display chat
- Security concerns noted
- Testing strategy outlined
```

**Your Instructions:**

```
Create React chat component with:
- Socket.IO client connection
- Message list display
- Message input and send
- Typing indicators
- Connection status
- Error handling
```

**Frontend Developer (Driver) Implements:**

```jsx
// client/src/components/ChatRoom.jsx

import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import { useAuth } from '../contexts/AuthContext';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import TypingIndicator from './TypingIndicator';

function ChatRoom({ roomId }) {
  const { user, token } = useAuth();
  const [messages, setMessages] = useState([]);
  const [typingUsers, setTypingUsers] = useState(new Set());
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const socketRef = useRef(null);
  const typingTimeoutRef = useRef(null);

  useEffect(() => {
    // Initialize socket connection
    socketRef.current = io(process.env.REACT_APP_SOCKET_URL, {
      auth: { token },
      transports: ['websocket', 'polling']
    });

    const socket = socketRef.current;

    // Connection event handlers
    socket.on('connect', () => {
      setConnectionStatus('connected');
      socket.emit('join-room', roomId);
    });

    socket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      setConnectionStatus('error');
    });

    socket.on('disconnect', () => {
      setConnectionStatus('disconnected');
    });

    // Message event handlers
    socket.on('room-history', (history) => {
      setMessages(history);
    });

    socket.on('new-message', (message) => {
      setMessages(prev => [...prev, message]);
    });

    // Typing indicators
    socket.on('user-typing', ({ userId, username }) => {
      setTypingUsers(prev => new Set([...prev, username]));
    });

    socket.on('user-stopped-typing', ({ userId }) => {
      setTypingUsers(prev => {
        const updated = new Set(prev);
        // Find and remove by userId (would need user mapping)
        return updated;
      });
    });

    // User presence
    socket.on('user-joined', ({ username }) => {
      // Show notification
      console.log(`${username} joined`);
    });

    socket.on('user-left', ({ username }) => {
      console.log(`${username} left`);
    });

    // Cleanup
    return () => {
      socket.disconnect();
    };
  }, [roomId, token]);

  const handleSendMessage = (text) => {
    if (!text.trim() || connectionStatus !== 'connected') {
      return;
    }

    socketRef.current.emit('send-message', { text });
  };

  const handleTyping = () => {
    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    } else {
      // First keystroke, emit typing-start
      socketRef.current.emit('typing-start');
    }

    // Set timeout to emit typing-stop
    typingTimeoutRef.current = setTimeout(() => {
      socketRef.current.emit('typing-stop');
      typingTimeoutRef.current = null;
    }, 1000);
  };

  return (
    <div className="chat-room">
      <div className="chat-header">
        <h2>Chat Room</h2>
        <ConnectionStatus status={connectionStatus} />
      </div>

      <MessageList
        messages={messages}
        currentUserId={user.id}
      />

      {typingUsers.size > 0 && (
        <TypingIndicator users={Array.from(typingUsers)} />
      )}

      <MessageInput
        onSend={handleSendMessage}
        onTyping={handleTyping}
        disabled={connectionStatus !== 'connected'}
      />
    </div>
  );
}

export default ChatRoom;
```

**Backend Developer (Navigator Feedback):**

```
✅ Good React patterns!

Suggestions:
- Add reconnection logic with exponential backoff
- Implement message queue for offline sending
- Add optimistic UI updates
```

---

## Rotation 3: Security Expert as Driver (14-21 min)

**Your Instructions:**

```
Add security layers:
- Input sanitization
- Rate limiting
- Message validation
- XSS prevention
```

**Security Expert (Driver) Implements:**

```javascript
// server/security/rate-limiter.js

const rateLimit = require('express-rate-limit');

// Socket.IO rate limiting
class SocketRateLimiter {
  constructor() {
    this.limits = new Map();
    this.windowMs = 60000; // 1 minute
    this.maxRequests = 30;  // 30 messages per minute
  }

  checkLimit(socketId) {
    const now = Date.now();
    const userLimits = this.limits.get(socketId) || { count: 0, resetTime: now + this.windowMs };

    if (now > userLimits.resetTime) {
      // Reset window
      this.limits.set(socketId, { count: 1, resetTime: now + this.windowMs });
      return true;
    }

    if (userLimits.count >= this.maxRequests) {
      return false; // Rate limit exceeded
    }

    userLimits.count++;
    return true;
  }

  cleanup() {
    // Remove old entries
    const now = Date.now();
    for (const [socketId, limits] of this.limits.entries()) {
      if (now > limits.resetTime) {
        this.limits.delete(socketId);
      }
    }
  }
}

// server/security/message-validator.js

const DOMPurify = require('isomorphic-dompurify');

class MessageValidator {
  static validate(message) {
    const errors = [];

    // Check required fields
    if (!message.text) {
      errors.push('Message text is required');
    }

    // Check length
    if (message.text && message.text.length > 1000) {
      errors.push('Message too long (max 1000 characters)');
    }

    if (message.text && message.text.length < 1) {
      errors.push('Message too short');
    }

    // Sanitize HTML/scripts
    message.text = this.sanitize(message.text);

    // Check for spam patterns
    if (this.isSpam(message.text)) {
      errors.push('Message appears to be spam');
    }

    return {
      valid: errors.length === 0,
      errors,
      message
    };
  }

  static sanitize(text) {
    // Remove HTML tags and scripts
    return DOMPurify.sanitize(text, {
      ALLOWED_TAGS: [],
      KEEP_CONTENT: true
    });
  }

  static isSpam(text) {
    // Basic spam detection
    const spamPatterns = [
      /(.)\1{10,}/,  // Repeated characters
      /http[s]?:\/\/.*\s+http[s]?:\/\//,  // Multiple URLs
      /BUY NOW/gi,
      /CLICK HERE/gi
    ];

    return spamPatterns.some(pattern => pattern.test(text));
  }
}

module.exports = { SocketRateLimiter, MessageValidator };
```

**Test Engineer (Navigator Feedback):**

```
🧪 SECURITY TESTING NEEDED:

Critical Tests:
1. Rate limiting enforcement
2. XSS prevention
3. Long message rejection
4. Spam detection accuracy
5. SQL injection prevention

Will implement in next rotation.
```

---

## Rotation 4: Test Engineer as Driver (21-28 min)

**Your Instructions:**

```
Create comprehensive test suite:
- Unit tests for message validation
- Integration tests for Socket.IO flow
- E2E tests for chat scenarios
```

**Test Engineer (Driver) Implements:**

```javascript
// tests/chat.integration.test.js

const io = require('socket.io-client');
const { createServer } = require('http');
const ChatSocketServer = require('../server/socket-server');

describe('Chat Integration Tests', () => {
  let httpServer;
  let chatServer;
  let clientSocket;

  beforeAll((done) => {
    httpServer = createServer();
    chatServer = new ChatSocketServer(httpServer);
    httpServer.listen(() => {
      const port = httpServer.address().port;
      done();
    });
  });

  afterAll(() => {
    chatServer.io.close();
    httpServer.close();
  });

  afterEach(() => {
    if (clientSocket) {
      clientSocket.close();
    }
  });

  describe('Authentication', () => {
    it('should reject connection without token', (done) => {
      clientSocket = io(`http://localhost:${port}`);

      clientSocket.on('connect_error', (err) => {
        expect(err.message).toBe('Authentication required');
        done();
      });
    });

    it('should accept connection with valid token', (done) => {
      const token = generateValidToken({ userId: '123', username: 'test' });

      clientSocket = io(`http://localhost:${port}`, {
        auth: { token }
      });

      clientSocket.on('connect', () => {
        expect(clientSocket.connected).toBe(true);
        done();
      });
    });
  });

  describe('Message Flow', () => {
    it('should broadcast message to room', (done) => {
      const token = generateValidToken({ userId: '123', username: 'user1' });
      const client1 = io(`http://localhost:${port}`, { auth: { token } });
      const client2 = io(`http://localhost:${port}`, { auth: { token } });

      Promise.all([
        new Promise(resolve => client1.on('connect', resolve)),
        new Promise(resolve => client2.on('connect', resolve))
      ]).then(() => {
        client1.emit('join-room', 'room1');
        client2.emit('join-room', 'room1');

        client2.on('new-message', (message) => {
          expect(message.text).toBe('Hello!');
          client1.close();
          client2.close();
          done();
        });

        client1.emit('send-message', { text: 'Hello!' });
      });
    });
  });

  describe('Security', () => {
    it('should sanitize XSS attempts', (done) => {
      const token = generateValidToken({ userId: '123', username: 'user' });
      clientSocket = io(`http://localhost:${port}`, { auth: { token } });

      clientSocket.on('connect', () => {
        clientSocket.emit('join-room', 'room1');

        clientSocket.on('new-message', (message) => {
          expect(message.text).not.toContain('<script>');
          expect(message.text).toBe('alert("xss")');
          done();
        });

        clientSocket.emit('send-message', {
          text: '<script>alert("xss")</script>'
        });
      });
    });

    it('should enforce rate limits', (done) => {
      const token = generateValidToken({ userId: '123', username: 'user' });
      clientSocket = io(`http://localhost:${port}`, { auth: { token } });

      clientSocket.on('connect', () => {
        clientSocket.emit('join-room', 'room1');

        // Send 31 messages rapidly
        let sentCount = 0;
        for (let i = 0; i < 31; i++) {
          clientSocket.emit('send-message', { text: `Message ${i}` });
          sentCount++;
        }

        // Should receive rate limit error
        clientSocket.on('error', (error) => {
          expect(error).toBe('Rate limit exceeded');
          done();
        });
      });
    });
  });
});
```

---

## Final Mob Session Summary

### Completed Features

✅ **Backend**: Socket.IO server with authentication
✅ **Frontend**: React chat UI with real-time updates
✅ **Security**: Rate limiting, input sanitization, XSS prevention
✅ **Testing**: Comprehensive test suite

### Mob Metrics

```json
{
  "mob": {
    "duration": 1680,
    "rotations": 4,
    "specialists": 4,
    "rotationDuration": 420
  },

  "contributions": {
    "backendDev": {
      "linesAdded": 120,
      "filesCreated": 1,
      "primaryFocus": "server-architecture"
    },
    "frontendDev": {
      "linesAdded": 95,
      "filesCreated": 1,
      "primaryFocus": "ui-components"
    },
    "securityExpert": {
      "linesAdded": 78,
      "filesCreated": 2,
      "primaryFocus": "security-hardening"
    },
    "testEngineer": {
      "linesAdded": 102,
      "filesCreated": 1,
      "primaryFocus": "testing"
    }
  },

  "quality": {
    "testCoverage": 92,
    "truthScore": 0.965,
    "securityScore": 0.94,
    "codeReviews": 4
  }
}
```

### Mob Programming Benefits Demonstrated

✅ **Knowledge Sharing**: All specialists learned from each other
✅ **Higher Quality**: Continuous review caught issues early
✅ **Faster Integration**: No merge conflicts or integration delays
✅ **Better Design**: Multiple perspectives improved architecture
✅ **Reduced Bugs**: Issues caught during implementation
✅ **Team Building**: Collaborative problem-solving

### When to Use Mob Programming

- Complex features requiring multiple expertise areas
- Knowledge transfer and team learning
- Critical features needing high quality
- Architectural decisions
- Problem-solving difficult bugs
- Building shared understanding

**Mob Programming Complete: Real-Time Chat Feature** ✅
