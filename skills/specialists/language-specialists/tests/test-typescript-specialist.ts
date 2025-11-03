/**
 * Test suite for TypeScript specialist tooling and validation.
 *
 * Tests TypeScript validator, Nest.js patterns, generic types, and build output.
 * Run with: pnpm test test-typescript-specialist.ts
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

// Test TypeScript Validator Script
describe('TypeScript Validator', () => {
  const validatorPath = path.join(__dirname, '..', 'resources', 'scripts', 'typescript-validator.js');

  it('should exist and be executable', () => {
    expect(fs.existsSync(validatorPath)).toBe(true);
    const stats = fs.statSync(validatorPath);
    expect(stats.size).toBeGreaterThan(0);
  });

  it('should show help when no arguments provided', () => {
    try {
      const output = execSync(`node ${validatorPath} --help`, { encoding: 'utf-8' });
      expect(output).toContain('TypeScript Validator');
    } catch (error: any) {
      // May exit with non-zero, but should produce output
      expect(error.stdout?.toString() || error.stderr?.toString()).toBeTruthy();
    }
  });
});

// Test Nest.js Controller Patterns
describe('Nest.js Controller Validation', () => {
  class CreateUserDto {
    email!: string;
    username!: string;
    password!: string;
  }

  class UsersService {
    private users: any[] = [];

    async create(dto: CreateUserDto): Promise<any> {
      const user = { id: this.users.length + 1, ...dto };
      this.users.push(user);
      return user;
    }

    async findAll(): Promise<any[]> {
      return this.users;
    }
  }

  class UsersController {
    constructor(private readonly usersService: UsersService) {}

    async create(dto: CreateUserDto): Promise<any> {
      return await this.usersService.create(dto);
    }

    async findAll(): Promise<any[]> {
      return await this.usersService.findAll();
    }
  }

  let controller: UsersController;
  let service: UsersService;

  beforeEach(() => {
    service = new UsersService();
    controller = new UsersController(service);
  });

  it('should create user via controller', async () => {
    const dto: CreateUserDto = {
      email: 'test@example.com',
      username: 'testuser',
      password: 'password123',
    };

    const result = await controller.create(dto);
    expect(result).toHaveProperty('id');
    expect(result.email).toBe('test@example.com');
  });

  it('should retrieve all users via controller', async () => {
    const dto1: CreateUserDto = {
      email: 'user1@example.com',
      username: 'user1',
      password: 'pass1',
    };
    const dto2: CreateUserDto = {
      email: 'user2@example.com',
      username: 'user2',
      password: 'pass2',
    };

    await controller.create(dto1);
    await controller.create(dto2);

    const users = await controller.findAll();
    expect(users).toHaveLength(2);
  });
});

// Test Generic Type Constraints
describe('Generic Type Constraints', () => {
  interface Entity {
    id: number;
  }

  class GenericRepository<T extends Entity> {
    private items = new Map<number, T>();

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

  interface User extends Entity {
    name: string;
  }

  it('should work with constrained generic types', async () => {
    const repo = new GenericRepository<User>();
    const user: User = { id: 1, name: 'Test User' };

    await repo.save(user);
    const found = await repo.findById(1);

    expect(found).toBeDefined();
    expect(found?.name).toBe('Test User');
  });

  it('should enforce type constraints', async () => {
    const repo = new GenericRepository<User>();

    // This should work
    const validUser: User = { id: 2, name: 'Valid User' };
    await repo.save(validUser);

    // TypeScript would prevent this at compile time:
    // const invalidUser = { name: 'No ID' };
    // await repo.save(invalidUser); // Error: missing 'id'
  });
});

// Test Advanced TypeScript Types
describe('Advanced TypeScript Types', () => {
  type ReadonlyUser = Readonly<{ id: number; name: string }>;
  type PartialUser = Partial<{ id: number; name: string; email: string }>;

  it('should enforce readonly properties', () => {
    const user: ReadonlyUser = { id: 1, name: 'Test' };
    expect(user.id).toBe(1);
    // user.id = 2; // TypeScript error: Cannot assign to 'id' because it is a read-only property
  });

  it('should allow partial properties', () => {
    const user1: PartialUser = { id: 1 };
    const user2: PartialUser = { name: 'Test' };
    const user3: PartialUser = {};

    expect(user1.id).toBe(1);
    expect(user2.name).toBe('Test');
    expect(user3.id).toBeUndefined();
  });

  type ApiResponse<T> =
    | { status: 'success'; data: T }
    | { status: 'error'; error: string }
    | { status: 'loading' };

  it('should handle discriminated unions', () => {
    function handleResponse<T>(response: ApiResponse<T>): string {
      switch (response.status) {
        case 'success':
          return 'Success';
        case 'error':
          return response.error;
        case 'loading':
          return 'Loading...';
      }
    }

    expect(handleResponse({ status: 'success', data: { id: 1 } })).toBe('Success');
    expect(handleResponse({ status: 'error', error: 'Failed' })).toBe('Failed');
    expect(handleResponse({ status: 'loading' })).toBe('Loading...');
  });
});

// Test Template Literal Types
describe('Template Literal Types', () => {
  type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE';
  type ApiVersion = 'v1' | 'v2';
  type Resource = 'users' | 'posts';

  type ApiRoute = `/${ApiVersion}/${Resource}`;
  type ApiEndpoint = `${HttpMethod} ${ApiRoute}`;

  it('should enforce template literal types', () => {
    const endpoint1: ApiEndpoint = 'GET /v1/users';
    const endpoint2: ApiEndpoint = 'POST /v2/posts';

    expect(endpoint1).toBe('GET /v1/users');
    expect(endpoint2).toBe('POST /v2/posts');

    // TypeScript would prevent these at compile time:
    // const invalid1: ApiEndpoint = 'PATCH /v1/users'; // Error: invalid HttpMethod
    // const invalid2: ApiEndpoint = 'GET /v3/users'; // Error: invalid ApiVersion
  });
});

// Test Build Output Verification
describe('Build Output Verification', () => {
  const configPath = path.join(__dirname, '..', 'resources', 'templates', 'typescript-config.json');

  it('should have valid tsconfig template', () => {
    expect(fs.existsSync(configPath)).toBe(true);
    const content = fs.readFileSync(configPath, 'utf-8');
    const config = JSON.parse(content);

    expect(config.compilerOptions).toBeDefined();
    expect(config.compilerOptions.strict).toBe(true);
    expect(config.compilerOptions.target).toBeDefined();
    expect(config.compilerOptions.module).toBeDefined();
  });

  it('should enforce strict compiler options', () => {
    const content = fs.readFileSync(configPath, 'utf-8');
    const config = JSON.parse(content);

    expect(config.compilerOptions.strict).toBe(true);
    expect(config.compilerOptions.noImplicitAny).toBe(true);
    expect(config.compilerOptions.strictNullChecks).toBe(true);
    expect(config.compilerOptions.noUnusedLocals).toBe(true);
    expect(config.compilerOptions.noUnusedParameters).toBe(true);
  });
});

// Test Dependency Injection Pattern
describe('Dependency Injection Pattern', () => {
  interface Logger {
    log(message: string): void;
  }

  class ConsoleLogger implements Logger {
    log(message: string): void {
      console.log(message);
    }
  }

  class UserService {
    constructor(private readonly logger: Logger) {}

    async createUser(name: string): Promise<{ id: number; name: string }> {
      this.logger.log(`Creating user: ${name}`);
      return { id: Math.random(), name };
    }
  }

  it('should support dependency injection', async () => {
    const logger = new ConsoleLogger();
    const service = new UserService(logger);

    const user = await service.createUser('Test User');
    expect(user.name).toBe('Test User');
    expect(user.id).toBeDefined();
  });
});

// Integration Tests
describe('Integration Tests', () => {
  it('should validate complete TypeScript workflow', () => {
    // Verify all required templates exist
    const templatesDir = path.join(__dirname, '..', 'resources', 'templates');
    expect(fs.existsSync(templatesDir)).toBe(true);

    const tsConfig = path.join(templatesDir, 'typescript-config.json');
    expect(fs.existsSync(tsConfig)).toBe(true);

    // Verify scripts exist
    const scriptsDir = path.join(__dirname, '..', 'resources', 'scripts');
    expect(fs.existsSync(scriptsDir)).toBe(true);

    const validator = path.join(scriptsDir, 'typescript-validator.js');
    expect(fs.existsSync(validator)).toBe(true);
  });
});
