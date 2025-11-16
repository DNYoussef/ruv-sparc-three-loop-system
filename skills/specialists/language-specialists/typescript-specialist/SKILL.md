---
name: typescript-specialist
description: Modern TypeScript development specialist for Node.js backends, Express/Nest.js
  frameworks, type-safe frontend development, npm package creation, and monorepo management
  with Turborepo/nx. Use when building TypeScript APIs, implementing type-safe full-stack
  applications, creating npm libraries, or requiring TypeScript best practices. Handles
  strict mode, advanced types, build tooling, and ESM/CommonJS modules.
category: Language Specialists
complexity: Medium
triggers:
- typescript
- node.js
- express
- nest.js
- npm package
- typescript api
- type safety
- monorepo
- turborepo
- ts-node
version: 1.0.0
tags:
- specialists
- domain-expert
author: ruv
---

# TypeScript Specialist

Expert TypeScript development for type-safe, scalable backend and full-stack applications with modern tooling.

## Purpose

This skill provides comprehensive TypeScript expertise including advanced type systems, modern Node.js patterns, framework integration (Express, Nest.js), and production-grade TypeScript configuration. It ensures TypeScript code leverages the full power of static typing while maintaining developer productivity.

## When to Use This Skill

Activate this skill when:
- Building backend APIs with Express or Nest.js
- Creating type-safe frontend applications
- Developing npm packages or libraries
- Setting up monorepos with Turborepo or nx
- Migrating JavaScript projects to TypeScript
- Configuring strict TypeScript compiler options
- Implementing advanced TypeScript patterns (generics, mapped types, conditional types)
- Optimizing TypeScript build performance

## Prerequisites

**Required Knowledge**:
- JavaScript ES6+ syntax and concepts
- Node.js runtime and npm/yarn/pnpm
- Basic understanding of static typing

**Required Tools**:
- Node.js 18+ installed
- npm, yarn, or pnpm package manager
- Code editor with TypeScript support (VS Code recommended)

**Agent Assignments**:
- `backend-dev`: Primary TypeScript API implementation
- `coder`: General TypeScript development
- `base-template-generator`: Project scaffolding
- `tester`: Jest/Vitest test suite creation
- `code-analyzer`: Type safety and quality analysis

## Core Workflows

### Workflow 1: Nest.js Backend API Development

**Step 1: Initialize Nest.js Project**

Create a production-ready Nest.js project with TypeScript:

```bash
# Install Nest CLI globally
npm install -g @nestjs/cli

# Create new project
nest new my-api --package-manager pnpm

# Navigate to project
cd my-api

# Install additional dependencies
pnpm add @nestjs/config @nestjs/typeorm typeorm pg class-validator class-transformer
pnpm add -D @types/node
```

**Step 2: Configure TypeScript Strict Mode**

```json
// tsconfig.json
{
  "compilerOptions": {
    "module": "commonjs",
    "declaration": true,
    "removeComments": true,
    "emitDecoratorMetadata": true,
    "experimentalDecorators": true,
    "allowSyntheticDefaultImports": true,
    "target": "ES2021",
    "sourceMap": true,
    "outDir": "./dist",
    "baseUrl": "./",
    "incremental": true,
    "skipLibCheck": true,
    "strict": true,
    "strictNullChecks": true,
    "noImplicitAny": true,
    "strictBindCallApply": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "esModuleInterop": true,
    "resolveJsonModule": true
  }
}
```

**Step 3: Create Type-Safe DTOs with Class Validator**

```typescript
// src/users/dto/create-user.dto.ts
import { IsEmail, IsString, MinLength, MaxLength } from 'class-validator';
import { ApiProperty } from '@nestjs/swagger';

export class CreateUserDto {
  @ApiProperty({ description: 'User email address' })
  @IsEmail()
  email: string;

  @ApiProperty({ description: 'Username', minLength: 3, maxLength: 50 })
  @IsString()
  @MinLength(3)
  @MaxLength(50)
  username: string;

  @ApiProperty({ description: 'User password', minLength: 8 })
  @IsString()
  @MinLength(8)
  password: string;
}
```

**Step 4: Implement Service with Dependency Injection**

```typescript
// src/users/users.service.ts
import { Injectable, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { User } from './entities/user.entity';
import { CreateUserDto } from './dto/create-user.dto';

@Injectable()
export class UsersService {
  constructor(
    @InjectRepository(User)
    private readonly usersRepository: Repository<User>,
  ) {}

  async create(createUserDto: CreateUserDto): Promise<User> {
    const user = this.usersRepository.create(createUserDto);
    return await this.usersRepository.save(user);
  }

  async findAll(): Promise<User[]> {
    return await this.usersRepository.find();
  }

  async findOne(id: number): Promise<User> {
    const user = await this.usersRepository.findOne({ where: { id } });
    if (!user) {
      throw new NotFoundException(`User with ID ${id} not found`);
    }
    return user;
  }
}
```

**Step 5: Create Controller with Type-Safe Routing**

```typescript
// src/users/users.controller.ts
import { Controller, Get, Post, Body, Param, ParseIntPipe } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { UsersService } from './users.service';
import { CreateUserDto } from './dto/create-user.dto';
import { User } from './entities/user.entity';

@ApiTags('users')
@Controller('users')
export class UsersController {
  constructor(private readonly usersService: UsersService) {}

  @Post()
  @ApiOperation({ summary: 'Create user' })
  @ApiResponse({ status: 201, description: 'User created', type: User })
  async create(@Body() createUserDto: CreateUserDto): Promise<User> {
    return await this.usersService.create(createUserDto);
  }

  @Get()
  @ApiOperation({ summary: 'Get all users' })
  @ApiResponse({ status: 200, description: 'Users retrieved', type: [User] })
  async findAll(): Promise<User[]> {
    return await this.usersService.findAll();
  }

  @Get(':id')
  @ApiOperation({ summary: 'Get user by ID' })
  @ApiResponse({ status: 200, description: 'User found', type: User })
  @ApiResponse({ status: 404, description: 'User not found' })
  async findOne(@Param('id', ParseIntPipe) id: number): Promise<User> {
    return await this.usersService.findOne(id);
  }
}
```

### Workflow 2: Advanced TypeScript Types

**Step 1: Utility Types and Mapped Types**

```typescript
// types/utils.ts
type ReadonlyUser = Readonly<User>;
type PartialUser = Partial<User>;
type RequiredUser = Required<User>;
type PickedUser = Pick<User, 'id' | 'email'>;
type OmittedUser = Omit<User, 'password'>;

// Custom mapped type
type NullableFields<T> = {
  [P in keyof T]: T[P] | null;
};

type UserWithNullableFields = NullableFields<User>;
```

**Step 2: Conditional Types**

```typescript
// types/conditionals.ts
type IsArray<T> = T extends any[] ? true : false;
type Unwrap<T> = T extends Promise<infer U> ? U : T;

// Extract function return type
type ReturnTypeOf<T> = T extends (...args: any[]) => infer R ? R : never;

// Deep readonly
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};
```

**Step 3: Generic Constraints**

```typescript
// utils/repository.ts
interface Entity {
  id: number;
}

class GenericRepository<T extends Entity> {
  private items: Map<number, T> = new Map();

  async save(item: T): Promise<T> {
    this.items.set(item.id, item);
    return item;
  }

  async findById(id: number): Promise<T | undefined> {
    return this.items.get(id);
  }

  async findAll(): Promise<T[]> {
    return Array.from(this.items.values());
  }
}
```

**Step 4: Template Literal Types**

```typescript
// types/routes.ts
type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE';
type ApiVersion = 'v1' | 'v2';
type Resource = 'users' | 'posts' | 'comments';

type ApiRoute = `/${ApiVersion}/${Resource}`;
type ApiEndpoint = `${HttpMethod} ${ApiRoute}`;

// Usage
const endpoint: ApiEndpoint = 'GET /v1/users'; // Valid
// const invalid: ApiEndpoint = 'PATCH /v3/users'; // Error
```

### Workflow 3: Express TypeScript API

**Step 1: Setup Express with TypeScript**

```bash
mkdir express-api && cd express-api
pnpm init
pnpm add express
pnpm add -D typescript @types/express @types/node ts-node-dev
npx tsc --init
```

**Step 2: Create Type-Safe Request Handlers**

```typescript
// src/types/express.ts
import { Request, Response, NextFunction } from 'express';

export interface TypedRequest<T> extends Request {
  body: T;
}

export interface TypedResponse<T> extends Response {
  json: (data: T) => this;
}

export type AsyncHandler<ReqBody = any, ResBody = any> = (
  req: TypedRequest<ReqBody>,
  res: TypedResponse<ResBody>,
  next: NextFunction
) => Promise<void>;
```

**Step 3: Implement Middleware with Types**

```typescript
// src/middleware/validation.ts
import { Request, Response, NextFunction } from 'express';
import { AnyZodObject } from 'zod';

export const validate = (schema: AnyZodObject) =>
  async (req: Request, res: Response, next: NextFunction) => {
    try {
      await schema.parseAsync({
        body: req.body,
        query: req.query,
        params: req.params,
      });
      next();
    } catch (error) {
      res.status(400).json({ error: 'Validation failed' });
    }
  };
```

**Step 4: Create Routes with Zod Validation**

```typescript
// src/routes/users.ts
import { Router } from 'express';
import { z } from 'zod';
import { validate } from '../middleware/validation';

const createUserSchema = z.object({
  body: z.object({
    email: z.string().email(),
    username: z.string().min(3).max(50),
    password: z.string().min(8),
  }),
});

const router = Router();

router.post('/', validate(createUserSchema), async (req, res) => {
  const { email, username, password } = req.body;
  // Type-safe: TypeScript knows the shape from Zod schema
  const user = await createUser({ email, username, password });
  res.status(201).json(user);
});

export default router;
```

### Workflow 4: NPM Package Development

**Step 1: Initialize Package with TypeScript**

```bash
mkdir my-package && cd my-package
pnpm init
pnpm add -D typescript @types/node tsup
```

**Step 2: Configure Package.json for Dual Module Support**

```json
{
  "name": "my-package",
  "version": "1.0.0",
  "type": "module",
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "require": "./dist/index.cjs",
      "import": "./dist/index.js",
      "types": "./dist/index.d.ts"
    }
  },
  "scripts": {
    "build": "tsup src/index.ts --format cjs,esm --dts",
    "dev": "tsup src/index.ts --format cjs,esm --dts --watch",
    "prepublishOnly": "pnpm build"
  },
  "files": [
    "dist"
  ]
}
```

**Step 3: Write Package Code with Type Exports**

```typescript
// src/index.ts
export interface Config {
  apiKey: string;
  endpoint: string;
}

export class ApiClient {
  private config: Config;

  constructor(config: Config) {
    this.config = config;
  }

  async fetch<T>(path: string): Promise<T> {
    const response = await fetch(`${this.config.endpoint}${path}`, {
      headers: { 'X-API-Key': this.config.apiKey },
    });
    return response.json();
  }
}

export { type Config as ApiConfig };
```

**Step 4: Build and Publish**

```bash
# Build package
pnpm build

# Test locally
pnpm link --global

# Publish to npm
pnpm publish
```

## Best Practices

**1. Always Enable Strict Mode**
```json
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitReturns": true
  }
}
```

**2. Use Const Assertions for Immutability**
```typescript
// ✅ GOOD: Const assertion creates readonly array of specific strings
const ROLES = ['admin', 'user', 'guest'] as const;
type Role = typeof ROLES[number]; // 'admin' | 'user' | 'guest'

// ❌ BAD: Mutable array, type is just string[]
const ROLES = ['admin', 'user', 'guest'];
```

**3. Avoid `any`, Use `unknown` Instead**
```typescript
// ✅ GOOD: Force type checking
function process(data: unknown): string {
  if (typeof data === 'string') {
    return data.toUpperCase();
  }
  return String(data);
}

// ❌ BAD: No type safety
function process(data: any): string {
  return data.toUpperCase(); // Runtime error if not string
}
```

**4. Use Type Guards**
```typescript
interface User {
  id: number;
  name: string;
}

function isUser(value: unknown): value is User {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    typeof value.id === 'number' &&
    'name' in value &&
    typeof value.name === 'string'
  );
}
```

**5. Leverage Discriminated Unions**
```typescript
type ApiResponse<T> =
  | { status: 'success'; data: T }
  | { status: 'error'; error: string }
  | { status: 'loading' };

function handleResponse<T>(response: ApiResponse<T>) {
  switch (response.status) {
    case 'success':
      console.log(response.data); // TypeScript knows data exists
      break;
    case 'error':
      console.error(response.error); // TypeScript knows error exists
      break;
    case 'loading':
      console.log('Loading...'); // No data or error here
      break;
  }
}
```

## Quality Criteria

**Type Safety**:
- ✅ No `any` types (exceptions documented)
- ✅ Strict mode enabled
- ✅ All public APIs have type definitions
- ✅ 100% type coverage

**Code Quality**:
- ✅ ESLint passes with TypeScript rules
- ✅ Prettier formatting applied
- ✅ No TypeScript compiler errors
- ✅ Test coverage ≥85%

**Build Configuration**:
- ✅ Dual module support (ESM + CJS)
- ✅ Source maps generated
- ✅ Declaration files (.d.ts) emitted
- ✅ Build time <10s for incremental builds

## Agent Coordination

**backend-dev Agent**:
```bash
# Pre-task: Initialize TypeScript environment
npx claude-flow@alpha hooks pre-task --description "TypeScript Nest.js API implementation"

# During: Track changes
npx claude-flow@alpha hooks post-edit --file "src/users/users.service.ts" --memory-key "typescript-api/nest-service"

# Post-task: Run type checking and tests
pnpm tsc --noEmit && pnpm test
npx claude-flow@alpha hooks post-task --task-id "typescript-api-implementation"
```

## Common Patterns

**Pattern 1: Factory Pattern with Generics**
```typescript
interface Factory<T> {
  create(): T;
}

class UserFactory implements Factory<User> {
  create(): User {
    return { id: Math.random(), name: 'New User' };
  }
}
```

**Pattern 2: Builder Pattern**
```typescript
class QueryBuilder<T> {
  private filters: Array<(item: T) => boolean> = [];

  where(predicate: (item: T) => boolean): this {
    this.filters.push(predicate);
    return this;
  }

  execute(items: T[]): T[] {
    return items.filter(item => this.filters.every(f => f(item)));
  }
}
```

**Pattern 3: Decorator Pattern with Metadata**
```typescript
function Log(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = async function(...args: any[]) {
    console.log(`Calling ${propertyKey} with args:`, args);
    const result = await originalMethod.apply(this, args);
    console.log(`${propertyKey} returned:`, result);
    return result;
  };
}
```

## Troubleshooting

**Issue**: "Cannot find module" errors despite correct imports
**Solution**: Check `tsconfig.json` `baseUrl` and `paths` settings, ensure `moduleResolution` is "node"

**Issue**: Slow TypeScript compilation
**Solution**: Enable `incremental: true`, use `skipLibCheck: true`, consider using `ts-node-dev` or `tsup` for faster builds

**Issue**: Type errors with third-party libraries
**Solution**: Install `@types/*` packages, check DefinitelyTyped, or create custom type declarations in `types/` directory

## Related Skills

- `base-template-generator`: Project scaffolding
- `react-specialist`: Frontend TypeScript
- `testing-quality`: Jest/Vitest testing
- `docker-containerization`: Containerizing Node.js apps

## Tools and Resources

**Frameworks**:
- Nest.js: https://nestjs.com/
- Express: https://expressjs.com/
- Fastify: https://www.fastify.io/

**Build Tools**:
- tsup: Fast TypeScript bundler
- esbuild: Extremely fast bundler
- Vite: Fast dev server

**Validation**:
- Zod: TypeScript-first schema validation
- class-validator: Decorator-based validation

**Type Utilities**:
- type-fest: Useful type utilities
- ts-pattern: Pattern matching for TypeScript

## MCP Tools

- `mcp__flow-nexus__sandbox_create` with `template: "node"` for Node.js testing
- `mcp__flow-nexus__sandbox_execute` for running TypeScript code
- `mcp__memory-mcp__memory_store` for persisting TypeScript patterns

## Success Metrics

**Development Speed**:
- New API endpoint: 10-20 minutes
- Service with business logic: 30-40 minutes
- Full feature with tests: 1.5-2 hours

**Type Safety**:
- Compiler errors: 0
- Type coverage: 100%
- Runtime type errors: <1% of bugs

**Build Performance**:
- Full build: <30 seconds
- Incremental build: <3 seconds
- Dev server startup: <2 seconds

---

**Skill Version**: 1.0.0
**Last Updated**: 2025-11-02
**Maintained By**: typescript-specialist agent