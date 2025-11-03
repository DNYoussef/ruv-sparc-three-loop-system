import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn } from 'typeorm';
import { ApiProperty } from '@nestjs/swagger';

@Entity('users')
export class User {
  @ApiProperty({ description: 'User ID' })
  @PrimaryGeneratedColumn()
  id!: number;

  @ApiProperty({ description: 'User email address' })
  @Column({ unique: true })
  email!: string;

  @ApiProperty({ description: 'Username' })
  @Column({ unique: true })
  username!: string;

  @Column()
  password!: string; // Hashed password (not exposed in API)

  @ApiProperty({ description: 'Account creation date' })
  @CreateDateColumn()
  createdAt!: Date;
}
