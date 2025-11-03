#!/usr/bin/env node

/**
 * Flow Nexus Sandbox Manager
 *
 * Manages sandbox lifecycle including creation, configuration, execution,
 * monitoring, and cleanup via Flow Nexus MCP tools.
 *
 * Usage:
 *   node sandbox-manager.js create <template> <name> [--env KEY=VALUE] [--packages pkg1,pkg2]
 *   node sandbox-manager.js list [--status running|stopped|all]
 *   node sandbox-manager.js status <sandboxId>
 *   node sandbox-manager.js execute <sandboxId> <code|@file>
 *   node sandbox-manager.js upload <sandboxId> <localPath> <remotePath>
 *   node sandbox-manager.js logs <sandboxId> [--lines N]
 *   node sandbox-manager.js stop <sandboxId>
 *   node sandbox-manager.js delete <sandboxId>
 *   node sandbox-manager.js cleanup-all [--older-than-hours N]
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class SandboxManager {
  constructor() {
    this.mcpPrefix = 'mcp__flow-nexus__';
    this.validTemplates = ['node', 'python', 'react', 'nextjs', 'vanilla', 'base', 'claude-code'];
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
   * Create new sandbox
   */
  create(template, name, options = {}) {
    if (!this.validTemplates.includes(template)) {
      console.error(`Invalid template: ${template}`);
      console.error(`Valid templates: ${this.validTemplates.join(', ')}`);
      process.exit(1);
    }

    console.log(`Creating ${template} sandbox: ${name}`);

    const params = { template, name };

    if (options.env) {
      params.env_vars = options.env;
    }

    if (options.packages) {
      params.install_packages = options.packages;
    }

    if (options.timeout) {
      params.timeout = parseInt(options.timeout);
    }

    const result = this.executeMCP('sandbox_create', params);

    if (result.success) {
      console.log('✓ Sandbox created successfully');
      console.log(`Sandbox ID: ${result.sandbox_id}`);
      console.log(`Status: ${result.status}`);
      console.log(`Template: ${template}`);
      return result;
    } else {
      console.error('✗ Sandbox creation failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * List sandboxes
   */
  list(status = 'all') {
    console.log(`Listing ${status} sandboxes...`);

    const result = this.executeMCP('sandbox_list', { status });

    if (result.sandboxes && result.sandboxes.length > 0) {
      console.log(`\nFound ${result.sandboxes.length} sandbox(es):\n`);

      result.sandboxes.forEach(sandbox => {
        console.log(`  ID: ${sandbox.id}`);
        console.log(`  Name: ${sandbox.name}`);
        console.log(`  Template: ${sandbox.template}`);
        console.log(`  Status: ${sandbox.status}`);
        console.log(`  Created: ${new Date(sandbox.created_at).toLocaleString()}`);
        console.log('  ---');
      });

      return result.sandboxes;
    } else {
      console.log('No sandboxes found');
      return [];
    }
  }

  /**
   * Get sandbox status
   */
  status(sandboxId) {
    console.log(`Checking status for sandbox: ${sandboxId}`);

    const result = this.executeMCP('sandbox_status', { sandbox_id: sandboxId });

    if (result.sandbox) {
      const s = result.sandbox;
      console.log('\nSandbox Details:');
      console.log(`  ID: ${s.id}`);
      console.log(`  Name: ${s.name}`);
      console.log(`  Template: ${s.template}`);
      console.log(`  Status: ${s.status}`);
      console.log(`  Created: ${new Date(s.created_at).toLocaleString()}`);
      console.log(`  Uptime: ${s.uptime || 'N/A'}`);

      if (s.env_vars) {
        console.log('\n  Environment Variables:');
        Object.entries(s.env_vars).forEach(([key, val]) => {
          console.log(`    ${key}=${val}`);
        });
      }

      return s;
    } else {
      console.error('✗ Sandbox not found');
      process.exit(1);
    }
  }

  /**
   * Execute code in sandbox
   */
  execute(sandboxId, codeOrFile, language = 'javascript') {
    let code = codeOrFile;

    // Check if code is a file reference (@file)
    if (codeOrFile.startsWith('@')) {
      const filePath = codeOrFile.substring(1);
      if (!fs.existsSync(filePath)) {
        console.error(`File not found: ${filePath}`);
        process.exit(1);
      }
      code = fs.readFileSync(filePath, 'utf-8');
      console.log(`Executing code from file: ${filePath}`);
    } else {
      console.log(`Executing code in sandbox: ${sandboxId}`);
    }

    const result = this.executeMCP('sandbox_execute', {
      sandbox_id: sandboxId,
      code,
      language,
      capture_output: true
    });

    if (result.success) {
      console.log('\n✓ Execution successful\n');

      if (result.stdout) {
        console.log('STDOUT:');
        console.log(result.stdout);
      }

      if (result.stderr) {
        console.log('\nSTDERR:');
        console.log(result.stderr);
      }

      console.log(`\nExecution time: ${result.execution_time}ms`);
      return result;
    } else {
      console.error('✗ Execution failed:', result.error);
      if (result.stderr) {
        console.error('\nSTDERR:');
        console.error(result.stderr);
      }
      process.exit(1);
    }
  }

  /**
   * Upload file to sandbox
   */
  upload(sandboxId, localPath, remotePath) {
    if (!fs.existsSync(localPath)) {
      console.error(`Local file not found: ${localPath}`);
      process.exit(1);
    }

    console.log(`Uploading ${localPath} to ${remotePath} in sandbox ${sandboxId}`);

    const content = fs.readFileSync(localPath, 'utf-8');

    const result = this.executeMCP('sandbox_upload', {
      sandbox_id: sandboxId,
      file_path: remotePath,
      content
    });

    if (result.success) {
      console.log('✓ File uploaded successfully');
      return result;
    } else {
      console.error('✗ Upload failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Get sandbox logs
   */
  logs(sandboxId, lines = 100) {
    console.log(`Fetching logs for sandbox: ${sandboxId} (last ${lines} lines)`);

    const result = this.executeMCP('sandbox_logs', {
      sandbox_id: sandboxId,
      lines: parseInt(lines)
    });

    if (result.logs) {
      console.log('\n--- Sandbox Logs ---\n');
      console.log(result.logs);
      return result.logs;
    } else {
      console.log('No logs available');
      return null;
    }
  }

  /**
   * Stop sandbox
   */
  stop(sandboxId) {
    console.log(`Stopping sandbox: ${sandboxId}`);

    const result = this.executeMCP('sandbox_stop', { sandbox_id: sandboxId });

    if (result.success) {
      console.log('✓ Sandbox stopped');
      return result;
    } else {
      console.error('✗ Stop failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Delete sandbox
   */
  delete(sandboxId) {
    console.log(`Deleting sandbox: ${sandboxId}`);

    const result = this.executeMCP('sandbox_delete', { sandbox_id: sandboxId });

    if (result.success) {
      console.log('✓ Sandbox deleted');
      return result;
    } else {
      console.error('✗ Delete failed:', result.error);
      process.exit(1);
    }
  }

  /**
   * Cleanup old/stopped sandboxes
   */
  cleanupAll(olderThanHours = null) {
    console.log('Cleaning up sandboxes...');

    const allSandboxes = this.executeMCP('sandbox_list', { status: 'all' });

    if (!allSandboxes.sandboxes || allSandboxes.sandboxes.length === 0) {
      console.log('No sandboxes to clean up');
      return;
    }

    let deletedCount = 0;
    const now = Date.now();
    const cutoffTime = olderThanHours ? now - (olderThanHours * 60 * 60 * 1000) : null;

    allSandboxes.sandboxes.forEach(sandbox => {
      const createdAt = new Date(sandbox.created_at).getTime();
      const shouldDelete =
        sandbox.status === 'stopped' ||
        (cutoffTime && createdAt < cutoffTime);

      if (shouldDelete) {
        console.log(`Deleting sandbox: ${sandbox.id} (${sandbox.name})`);
        try {
          this.executeMCP('sandbox_delete', { sandbox_id: sandbox.id });
          deletedCount++;
        } catch (error) {
          console.error(`Failed to delete ${sandbox.id}:`, error.message);
        }
      }
    });

    console.log(`✓ Cleanup complete. Deleted ${deletedCount} sandbox(es)`);
  }
}

// CLI Interface
function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log(`
Flow Nexus Sandbox Manager

Usage:
  node sandbox-manager.js create <template> <name> [--env KEY=VALUE] [--packages pkg1,pkg2]
  node sandbox-manager.js list [--status running|stopped|all]
  node sandbox-manager.js status <sandboxId>
  node sandbox-manager.js execute <sandboxId> <code|@file>
  node sandbox-manager.js upload <sandboxId> <localPath> <remotePath>
  node sandbox-manager.js logs <sandboxId> [--lines N]
  node sandbox-manager.js stop <sandboxId>
  node sandbox-manager.js delete <sandboxId>
  node sandbox-manager.js cleanup-all [--older-than-hours N]

Templates: node, python, react, nextjs, vanilla, base, claude-code

Examples:
  node sandbox-manager.js create node my-api --env PORT=3000 --packages express,cors
  node sandbox-manager.js list --status running
  node sandbox-manager.js execute sbx_123 "console.log('Hello')"
  node sandbox-manager.js execute sbx_123 @script.js
  node sandbox-manager.js cleanup-all --older-than-hours 24
    `);
    process.exit(0);
  }

  const manager = new SandboxManager();
  const command = args[0];

  try {
    switch (command) {
      case 'create': {
        if (args.length < 3) {
          console.error('Error: create requires <template> <name>');
          process.exit(1);
        }

        const options = {};
        for (let i = 3; i < args.length; i++) {
          if (args[i] === '--env' && args[i + 1]) {
            options.env = options.env || {};
            const [key, value] = args[i + 1].split('=');
            options.env[key] = value;
            i++;
          } else if (args[i] === '--packages' && args[i + 1]) {
            options.packages = args[i + 1].split(',');
            i++;
          } else if (args[i] === '--timeout' && args[i + 1]) {
            options.timeout = args[i + 1];
            i++;
          }
        }

        manager.create(args[1], args[2], options);
        break;
      }

      case 'list': {
        const statusIdx = args.indexOf('--status');
        const status = statusIdx !== -1 && args[statusIdx + 1] ? args[statusIdx + 1] : 'all';
        manager.list(status);
        break;
      }

      case 'status':
        if (args.length < 2) {
          console.error('Error: status requires <sandboxId>');
          process.exit(1);
        }
        manager.status(args[1]);
        break;

      case 'execute':
        if (args.length < 3) {
          console.error('Error: execute requires <sandboxId> <code|@file>');
          process.exit(1);
        }
        manager.execute(args[1], args[2]);
        break;

      case 'upload':
        if (args.length < 4) {
          console.error('Error: upload requires <sandboxId> <localPath> <remotePath>');
          process.exit(1);
        }
        manager.upload(args[1], args[2], args[3]);
        break;

      case 'logs': {
        if (args.length < 2) {
          console.error('Error: logs requires <sandboxId>');
          process.exit(1);
        }
        const linesIdx = args.indexOf('--lines');
        const lines = linesIdx !== -1 && args[linesIdx + 1] ? args[linesIdx + 1] : 100;
        manager.logs(args[1], lines);
        break;
      }

      case 'stop':
        if (args.length < 2) {
          console.error('Error: stop requires <sandboxId>');
          process.exit(1);
        }
        manager.stop(args[1]);
        break;

      case 'delete':
        if (args.length < 2) {
          console.error('Error: delete requires <sandboxId>');
          process.exit(1);
        }
        manager.delete(args[1]);
        break;

      case 'cleanup-all': {
        const hoursIdx = args.indexOf('--older-than-hours');
        const hours = hoursIdx !== -1 && args[hoursIdx + 1] ? parseInt(args[hoursIdx + 1]) : null;
        manager.cleanupAll(hours);
        break;
      }

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

module.exports = SandboxManager;
