#!/usr/bin/env node

/**
 * Flow Nexus Authentication Manager
 *
 * Handles user authentication workflows including registration, login,
 * password management, and profile updates via Flow Nexus MCP tools.
 *
 * Usage:
 *   node auth-manager.js register <email> <password> [fullName]
 *   node auth-manager.js login <email> <password>
 *   node auth-manager.js logout
 *   node auth-manager.js status [--detailed]
 *   node auth-manager.js reset-password <email>
 *   node auth-manager.js update-profile <userId> <key=value> [key=value...]
 *   node auth-manager.js upgrade <userId> <tier>
 */

const { execSync } = require('child_process');

class AuthManager {
  constructor() {
    this.mcpPrefix = 'mcp__flow-nexus__';
  }

  /**
   * Execute MCP tool and return parsed result
   */
  executeMCP(toolName, params = {}) {
    const command = `claude mcp call ${this.mcpPrefix}${toolName} '${JSON.stringify(params)}'`;

    try {
      const result = execSync(command, { encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] });
      return JSON.parse(result);
    } catch (error) {
      console.error(`MCP call failed: ${error.message}`);
      if (error.stderr) {
        console.error(`Error details: ${error.stderr.toString()}`);
      }
      process.exit(1);
    }
  }

  /**
   * Register new user account
   */
  register(email, password, fullName = null, username = null) {
    console.log(`Registering user: ${email}`);

    const params = { email, password };
    if (fullName) params.full_name = fullName;
    if (username) params.username = username;

    const result = this.executeMCP('user_register', params);

    if (result.success) {
      console.log('✓ Registration successful!');
      console.log(`User ID: ${result.user_id}`);
      console.log(`Email verification sent to: ${email}`);
      return result;
    } else {
      console.error('✗ Registration failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Login user and create session
   */
  login(email, password) {
    console.log(`Logging in: ${email}`);

    const result = this.executeMCP('user_login', { email, password });

    if (result.success) {
      console.log('✓ Login successful!');
      console.log(`Session token: ${result.token.substring(0, 20)}...`);
      console.log(`Expires: ${new Date(result.expires_at).toLocaleString()}`);
      return result;
    } else {
      console.error('✗ Login failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Logout user and clear session
   */
  logout() {
    console.log('Logging out...');

    const result = this.executeMCP('user_logout');

    if (result.success) {
      console.log('✓ Logout successful');
      return result;
    } else {
      console.error('✗ Logout failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Check authentication status
   */
  status(detailed = false) {
    console.log('Checking authentication status...');

    const result = this.executeMCP('auth_status', { detailed });

    if (result.authenticated) {
      console.log('✓ Authenticated');
      console.log(`User: ${result.user.email}`);
      console.log(`Tier: ${result.user.tier}`);

      if (detailed && result.permissions) {
        console.log('\nPermissions:');
        result.permissions.forEach(perm => console.log(`  - ${perm}`));
      }

      return result;
    } else {
      console.log('✗ Not authenticated');
      return result;
    }
  }

  /**
   * Request password reset
   */
  resetPassword(email) {
    console.log(`Requesting password reset for: ${email}`);

    const result = this.executeMCP('user_reset_password', { email });

    if (result.success) {
      console.log('✓ Password reset email sent');
      console.log('Check your email for reset instructions');
      return result;
    } else {
      console.error('✗ Password reset request failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Update user profile
   */
  updateProfile(userId, updates) {
    console.log(`Updating profile for user: ${userId}`);

    const result = this.executeMCP('user_update_profile', { user_id: userId, updates });

    if (result.success) {
      console.log('✓ Profile updated successfully');
      console.log('Updated fields:', Object.keys(updates).join(', '));
      return result;
    } else {
      console.error('✗ Profile update failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Upgrade user tier
   */
  upgrade(userId, tier) {
    console.log(`Upgrading user ${userId} to ${tier} tier`);

    const validTiers = ['pro', 'enterprise'];
    if (!validTiers.includes(tier.toLowerCase())) {
      console.error(`✗ Invalid tier: ${tier}. Must be one of: ${validTiers.join(', ')}`);
      process.exit(1);
    }

    const result = this.executeMCP('user_upgrade', { user_id: userId, tier: tier.toLowerCase() });

    if (result.success) {
      console.log(`✓ Upgraded to ${tier} tier`);
      console.log(`New features: ${result.features.join(', ')}`);
      return result;
    } else {
      console.error('✗ Upgrade failed:', result.error);
      process.exit(1);
    }
  }
}

// CLI Interface
function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log(`
Flow Nexus Authentication Manager

Usage:
  node auth-manager.js register <email> <password> [fullName]
  node auth-manager.js login <email> <password>
  node auth-manager.js logout
  node auth-manager.js status [--detailed]
  node auth-manager.js reset-password <email>
  node auth-manager.js update-profile <userId> <key=value> [key=value...]
  node auth-manager.js upgrade <userId> <tier>

Examples:
  node auth-manager.js register user@example.com SecurePass123 "John Doe"
  node auth-manager.js login user@example.com SecurePass123
  node auth-manager.js status --detailed
  node auth-manager.js update-profile user123 bio="AI Developer" github_username=johndoe
  node auth-manager.js upgrade user123 pro
    `);
    process.exit(0);
  }

  const manager = new AuthManager();
  const command = args[0];

  try {
    switch (command) {
      case 'register':
        if (args.length < 3) {
          console.error('Error: register requires <email> <password> [fullName]');
          process.exit(1);
        }
        manager.register(args[1], args[2], args[3], args[4]);
        break;

      case 'login':
        if (args.length < 3) {
          console.error('Error: login requires <email> <password>');
          process.exit(1);
        }
        manager.login(args[1], args[2]);
        break;

      case 'logout':
        manager.logout();
        break;

      case 'status':
        const detailed = args.includes('--detailed');
        manager.status(detailed);
        break;

      case 'reset-password':
        if (args.length < 2) {
          console.error('Error: reset-password requires <email>');
          process.exit(1);
        }
        manager.resetPassword(args[1]);
        break;

      case 'update-profile':
        if (args.length < 3) {
          console.error('Error: update-profile requires <userId> <key=value> [key=value...]');
          process.exit(1);
        }
        const userId = args[1];
        const updates = {};
        for (let i = 2; i < args.length; i++) {
          const [key, value] = args[i].split('=');
          if (!key || !value) {
            console.error(`Error: Invalid update format: ${args[i]}. Use key=value`);
            process.exit(1);
          }
          updates[key] = value;
        }
        manager.updateProfile(userId, updates);
        break;

      case 'upgrade':
        if (args.length < 3) {
          console.error('Error: upgrade requires <userId> <tier>');
          process.exit(1);
        }
        manager.upgrade(args[1], args[2]);
        break;

      default:
        console.error(`Unknown command: ${command}`);
        console.error('Run without arguments to see usage');
        process.exit(1);
    }
  } catch (error) {
    console.error(`Fatal error: ${error.message}`);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = AuthManager;
