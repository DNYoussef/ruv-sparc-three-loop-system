# Example 1: Generate Node.js Express API Template

## Scenario
You need to start a new REST API project using Node.js and Express with modern best practices, but don't want to spend time setting up the basic structure.

## User Request
> "Generate a Node.js Express API template with TypeScript, basic routing, and error handling"

## Step-by-Step Process

### Step 1: Analyze Requirements
**Skill automatically identifies**:
- Language: Node.js with TypeScript
- Framework: Express
- Features: Basic routing, error handling
- Structure: RESTful API pattern

### Step 2: Generate Directory Structure
```
my-api/
├── src/
│   ├── routes/
│   │   ├── index.ts          # Route aggregator
│   │   └── health.ts         # Health check endpoint
│   ├── middleware/
│   │   ├── errorHandler.ts   # Global error handling
│   │   └── logger.ts         # Request logging
│   ├── controllers/
│   │   └── healthController.ts
│   ├── types/
│   │   └── express.d.ts      # Type definitions
│   ├── config/
│   │   └── index.ts          # Configuration management
│   ├── app.ts                # Express app setup
│   └── server.ts             # Server entry point
├── tests/
│   ├── integration/
│   │   └── health.test.ts
│   └── unit/
│       └── errorHandler.test.ts
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── .eslintrc.json            # ESLint configuration
├── .prettierrc               # Prettier configuration
├── tsconfig.json             # TypeScript configuration
├── package.json              # Dependencies and scripts
├── jest.config.js            # Jest testing configuration
└── README.md                 # Project documentation
```

### Step 3: Core Files Content

#### package.json
```json
{
  "name": "my-api",
  "version": "1.0.0",
  "description": "Express TypeScript API",
  "main": "dist/server.js",
  "scripts": {
    "dev": "tsx watch src/server.ts",
    "build": "tsc",
    "start": "node dist/server.js",
    "test": "jest",
    "test:watch": "jest --watch",
    "lint": "eslint src/**/*.ts",
    "format": "prettier --write src/**/*.ts"
  },
  "dependencies": {
    "express": "^4.18.2",
    "dotenv": "^16.3.1",
    "helmet": "^7.1.0",
    "cors": "^2.8.5"
  },
  "devDependencies": {
    "@types/express": "^4.17.21",
    "@types/node": "^20.10.0",
    "typescript": "^5.3.0",
    "tsx": "^4.7.0",
    "jest": "^29.7.0",
    "@types/jest": "^29.5.11",
    "eslint": "^8.56.0",
    "prettier": "^3.1.1"
  }
}
```

#### src/server.ts
```typescript
import app from './app';
import { config } from './config';

const PORT = config.port;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Environment: ${config.nodeEnv}`);
});
```

#### src/app.ts
```typescript
import express, { Application } from 'express';
import helmet from 'helmet';
import cors from 'cors';
import routes from './routes';
import { errorHandler } from './middleware/errorHandler';
import { requestLogger } from './middleware/logger';

const app: Application = express();

// Security middleware
app.use(helmet());
app.use(cors());

// Body parsing middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Request logging
app.use(requestLogger);

// Routes
app.use('/api', routes);

// Error handling (must be last)
app.use(errorHandler);

export default app;
```

#### src/routes/health.ts
```typescript
import { Router } from 'express';
import { healthCheck } from '../controllers/healthController';

const router = Router();

router.get('/health', healthCheck);

export default router;
```

#### src/controllers/healthController.ts
```typescript
import { Request, Response } from 'express';

export const healthCheck = (req: Request, res: Response): void => {
  res.status(200).json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
};
```

#### src/middleware/errorHandler.ts
```typescript
import { Request, Response, NextFunction } from 'express';

export const errorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  console.error('Error:', err);

  res.status(500).json({
    error: {
      message: process.env.NODE_ENV === 'production'
        ? 'Internal server error'
        : err.message,
      ...(process.env.NODE_ENV !== 'production' && { stack: err.stack })
    }
  });
};
```

#### .env.example
```bash
NODE_ENV=development
PORT=3000
LOG_LEVEL=info
```

#### tsconfig.json
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "moduleResolution": "node"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "tests"]
}
```

### Step 4: Setup Instructions

The generated README.md includes:

```markdown
# My API

Express TypeScript REST API

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Copy environment variables:
   ```bash
   cp .env.example .env
   ```

3. Start development server:
   ```bash
   npm run dev
   ```

## Available Scripts

- `npm run dev` - Start development server with hot reload
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm test` - Run tests
- `npm run lint` - Lint code
- `npm run format` - Format code

## API Endpoints

- `GET /api/health` - Health check endpoint

## Project Structure

See directory tree above for complete structure.
```

## Result

**Generated in**: ~2 minutes
**Files created**: 20+ files
**Ready to use**: Immediately runnable with `npm install && npm run dev`

## Next Steps

After generation, you can:
1. Add business logic in new controllers
2. Create additional routes for your API
3. Implement database connections
4. Add authentication/authorization
5. Deploy using generated structure

## Benefits

✅ **Time saved**: ~30-60 minutes of setup time
✅ **Best practices**: ESLint, Prettier, TypeScript configured
✅ **Testing ready**: Jest setup included
✅ **Production ready**: Error handling, logging, security headers
✅ **Clean architecture**: Organized by concern (routes, controllers, middleware)

## Common Variations

### Add database support
> "Add PostgreSQL with TypeORM to this template"

### Add authentication
> "Include JWT authentication middleware"

### Different framework
> "Generate the same but with Fastify instead of Express"
