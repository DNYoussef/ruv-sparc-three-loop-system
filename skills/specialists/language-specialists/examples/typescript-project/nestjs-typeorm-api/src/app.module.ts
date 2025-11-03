/**
 * Nest.js + TypeORM API Example - Root Module
 * Complete production-ready API with dependency injection, Swagger docs, and E2E tests.
 *
 * Features:
 * - Nest.js for scalable architecture
 * - TypeORM for type-safe database operations
 * - class-validator for DTO validation
 * - Swagger for API documentation
 * - Jest for testing
 *
 * Total: ~220 lines across all files
 */

import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { UsersModule } from './users/users.module';
import { ConfigModule } from '@nestjs/config';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
    }),
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: process.env.DB_HOST || 'localhost',
      port: parseInt(process.env.DB_PORT || '5432'),
      username: process.env.DB_USER || 'postgres',
      password: process.env.DB_PASSWORD || 'postgres',
      database: process.env.DB_NAME || 'nestjs_db',
      entities: [__dirname + '/**/*.entity{.ts,.js}'],
      synchronize: process.env.NODE_ENV !== 'production', // Don't use in production!
      logging: process.env.NODE_ENV === 'development',
    }),
    UsersModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
