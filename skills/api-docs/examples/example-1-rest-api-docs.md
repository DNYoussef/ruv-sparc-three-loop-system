# Example 1: REST API Documentation with OpenAPI

**Scenario**: Generate comprehensive OpenAPI 3.0 documentation for an Express.js REST API with authentication, CRUD operations, and error handling.

**Difficulty**: Basic
**Time**: 30-45 minutes
**Prerequisites**: Express.js REST API with routes and controllers

## 🎯 Objective

Transform an existing Express.js REST API into a fully documented OpenAPI 3.0 specification with:
- Complete endpoint documentation
- Request/response schemas
- Authentication flows (JWT)
- Error response examples
- Interactive Swagger UI

## 📋 Starting Point

### Existing Code Structure
```javascript
// routes/users.js
const express = require('express');
const router = express.Router();
const userController = require('../controllers/userController');
const auth = require('../middleware/auth');

router.get('/users', auth, userController.getAllUsers);
router.get('/users/:id', auth, userController.getUserById);
router.post('/users', auth, userController.createUser);
router.put('/users/:id', auth, userController.updateUser);
router.delete('/users/:id', auth, userController.deleteUser);

module.exports = router;
```

### Controller Example
```javascript
// controllers/userController.js
exports.getAllUsers = async (req, res) => {
  try {
    const users = await User.find();
    res.json({ success: true, data: users });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

exports.getUserById = async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (!user) {
      return res.status(404).json({ success: false, error: 'User not found' });
    }
    res.json({ success: true, data: user });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};
```

## 🚀 Step-by-Step Process

### Step 1: Initialize OpenAPI Configuration

Create `swagger.config.js`:

```javascript
const swaggerJsDoc = require('swagger-jsdoc');
const swaggerUi = require('swagger-ui-express');

const swaggerOptions = {
  definition: {
    openapi: '3.0.0',
    info: {
      title: 'User Management API',
      version: '1.0.0',
      description: 'Complete REST API for user management with JWT authentication',
      contact: {
        name: 'API Support',
        email: 'api@example.com'
      },
      license: {
        name: 'MIT',
        url: 'https://opensource.org/licenses/MIT'
      }
    },
    servers: [
      {
        url: 'http://localhost:3000/api/v1',
        description: 'Development server'
      },
      {
        url: 'https://api.example.com/v1',
        description: 'Production server'
      }
    ],
    components: {
      securitySchemes: {
        bearerAuth: {
          type: 'http',
          scheme: 'bearer',
          bearerFormat: 'JWT',
          description: 'Enter JWT token in format: Bearer {token}'
        }
      },
      schemas: {
        User: {
          type: 'object',
          required: ['email', 'name'],
          properties: {
            id: {
              type: 'string',
              format: 'uuid',
              description: 'Auto-generated UUID',
              example: '123e4567-e89b-12d3-a456-426614174000'
            },
            email: {
              type: 'string',
              format: 'email',
              description: 'User email address',
              example: 'user@example.com'
            },
            name: {
              type: 'string',
              description: 'Full name',
              example: 'John Doe'
            },
            role: {
              type: 'string',
              enum: ['user', 'admin', 'moderator'],
              default: 'user',
              description: 'User role'
            },
            createdAt: {
              type: 'string',
              format: 'date-time',
              description: 'Account creation timestamp',
              example: '2025-01-15T10:30:00Z'
            }
          }
        },
        Error: {
          type: 'object',
          properties: {
            success: {
              type: 'boolean',
              example: false
            },
            error: {
              type: 'string',
              description: 'Error message',
              example: 'User not found'
            },
            code: {
              type: 'string',
              description: 'Error code',
              example: 'USER_NOT_FOUND'
            }
          }
        },
        SuccessResponse: {
          type: 'object',
          properties: {
            success: {
              type: 'boolean',
              example: true
            },
            data: {
              description: 'Response data'
            },
            message: {
              type: 'string',
              description: 'Success message',
              example: 'Operation completed successfully'
            }
          }
        }
      }
    },
    security: [
      {
        bearerAuth: []
      }
    ]
  },
  apis: ['./routes/*.js', './controllers/*.js']
};

const swaggerSpec = swaggerJsDoc(swaggerOptions);

module.exports = { swaggerUi, swaggerSpec };
```

### Step 2: Add JSDoc Annotations to Routes

Update `routes/users.js`:

```javascript
/**
 * @swagger
 * /users:
 *   get:
 *     summary: Retrieve all users
 *     description: Get a list of all users in the system. Requires authentication.
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: page
 *         schema:
 *           type: integer
 *           default: 1
 *         description: Page number for pagination
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 10
 *         description: Number of items per page
 *       - in: query
 *         name: role
 *         schema:
 *           type: string
 *           enum: [user, admin, moderator]
 *         description: Filter by user role
 *     responses:
 *       200:
 *         description: Successfully retrieved users
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   type: array
 *                   items:
 *                     $ref: '#/components/schemas/User'
 *                 pagination:
 *                   type: object
 *                   properties:
 *                     page:
 *                       type: integer
 *                     limit:
 *                       type: integer
 *                     total:
 *                       type: integer
 *       401:
 *         description: Unauthorized - Invalid or missing token
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *       500:
 *         description: Internal server error
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 */
router.get('/users', auth, userController.getAllUsers);

/**
 * @swagger
 * /users/{id}:
 *   get:
 *     summary: Get user by ID
 *     description: Retrieve detailed information about a specific user
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *           format: uuid
 *         description: User ID
 *         example: 123e4567-e89b-12d3-a456-426614174000
 *     responses:
 *       200:
 *         description: User found
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   $ref: '#/components/schemas/User'
 *       404:
 *         description: User not found
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *             example:
 *               success: false
 *               error: "User not found"
 *               code: "USER_NOT_FOUND"
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Server error
 */
router.get('/users/:id', auth, userController.getUserById);

/**
 * @swagger
 * /users:
 *   post:
 *     summary: Create new user
 *     description: Register a new user in the system
 *     tags: [Users]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - email
 *               - name
 *               - password
 *             properties:
 *               email:
 *                 type: string
 *                 format: email
 *                 example: newuser@example.com
 *               name:
 *                 type: string
 *                 example: Jane Smith
 *               password:
 *                 type: string
 *                 format: password
 *                 minLength: 8
 *                 example: SecureP@ssw0rd
 *               role:
 *                 type: string
 *                 enum: [user, admin, moderator]
 *                 default: user
 *     responses:
 *       201:
 *         description: User created successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 data:
 *                   $ref: '#/components/schemas/User'
 *                 message:
 *                   type: string
 *                   example: "User created successfully"
 *       400:
 *         description: Bad request - Invalid input
 *         content:
 *           application/json:
 *             schema:
 *               $ref: '#/components/schemas/Error'
 *       409:
 *         description: Conflict - Email already exists
 *       500:
 *         description: Server error
 */
router.post('/users', auth, userController.createUser);
```

### Step 3: Mount Swagger UI in Server

Update `server.js`:

```javascript
const express = require('express');
const { swaggerUi, swaggerSpec } = require('./swagger.config');

const app = express();

// Middleware
app.use(express.json());

// API Documentation
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerSpec, {
  explorer: true,
  customCss: '.swagger-ui .topbar { display: none }',
  customSiteTitle: "User Management API Docs"
}));

// Serve OpenAPI JSON
app.get('/api-docs.json', (req, res) => {
  res.setHeader('Content-Type', 'application/json');
  res.send(swaggerSpec);
});

// Routes
app.use('/api/v1', require('./routes/users'));

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`API Docs available at http://localhost:${PORT}/api-docs`);
});
```

### Step 4: Install Dependencies

```bash
npm install --save swagger-jsdoc swagger-ui-express
```

### Step 5: Test Documentation

1. **Start server**:
```bash
npm start
```

2. **Access Swagger UI**:
```
http://localhost:3000/api-docs
```

3. **Test authentication**:
   - Click "Authorize" button
   - Enter JWT token: `Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`
   - Click "Authorize"

4. **Try endpoints**:
   - Expand GET /users
   - Click "Try it out"
   - Modify parameters
   - Click "Execute"

## ✅ Success Criteria

Your API documentation is complete when:

1. **OpenAPI Spec Valid**:
   - Validate at https://editor.swagger.io
   - All schemas defined
   - All endpoints documented

2. **Swagger UI Functional**:
   - All endpoints visible
   - Try-it-out works for each endpoint
   - Authentication flow works
   - Examples are accurate

3. **Documentation Complete**:
   - All CRUD operations documented
   - Error responses included
   - Request/response examples present
   - Security requirements clear

## 📊 Generated Artifacts

```
project/
├── swagger.config.js          # OpenAPI configuration
├── routes/
│   └── users.js               # Annotated routes
├── server.js                  # Swagger UI mounted
└── docs/
    └── openapi.json           # Generated spec
```

## 🎯 Key Takeaways

1. **JSDoc Annotations**: Document routes inline with code
2. **Schema Reusability**: Define common schemas in components
3. **Security Documentation**: Clearly document auth requirements
4. **Error Handling**: Document all possible error responses
5. **Interactive Testing**: Swagger UI enables API testing

## 🚀 Next Steps

1. **Add Redoc**: Alternative documentation viewer
2. **Generate Client SDKs**: Use OpenAPI Generator
3. **API Versioning**: Document v1, v2 differences
4. **Rate Limiting**: Document rate limit headers
5. **Webhooks**: Document webhook events

## 🔗 Related Examples

- **Example 2**: GraphQL schema documentation
- **Example 3**: API versioning strategies
