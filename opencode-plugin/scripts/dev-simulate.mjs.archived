#!/usr/bin/env node
import path from 'path';
import url from 'url';
import { fileURLToPath } from 'url';
import plugin from '../dist/index.js';

class FakeHost {
  constructor() {
    this.commands = new Map();
    this.system = [];
    this.onUserMessageHandler = null;
  }

  registerCommand(cmd) {
    this.commands.set(cmd.name.replace(/^\\//, ''), cmd);
  }

  onUserMessage(handler) {
    this.onUserMessageHandler = handler;
  }

  addSystemContext(text) {
    this.system.push(text);
  }
}

async function main() {
  const host = new FakeHost();
  await plugin(host, {});

  const [mode, ...rest] = process.argv.slice(2);
  if (!mode || mode === 'help') {
    printHelp();
    process.exit(0);
  }

  if (mode === 'list') {
    for (const [name, cmd] of host.commands.entries()) {
      console.log(`/${name} - ${cmd.description}`);
    }
    return;
  }

  if (mode === 'run') {
    const [cmdName, ...args] = rest;
    if (!cmdName) {
      console.error('Usage: npm run dev -- run <command> \"args\"');
      process.exit(1);
    }
    const command = host.commands.get(cmdName.replace(/^\\//, ''));
    if (!command) {
      console.error(`Command not found: ${cmdName}`);
      process.exit(1);
    }
    const output = await command.run(args.join(' '), {});
    console.log(output);
    return;
  }

  if (mode === 'match') {
    const text = rest.join(' ');
    if (!host.onUserMessageHandler) {
      console.error('Skill injection is not enabled or supported in this harness.');
      process.exit(1);
    }
    await host.onUserMessageHandler(text, {});
    console.log(host.system.join('\\n\\n'));
    return;
  }

  printHelp();
}

function printHelp() {
  console.log(`Usage: npm run dev -- <command>
Commands:
  list                    List registered commands
  run <name> \"args\"        Run a command and print the expanded prompt
  match \"text\"             Simulate skill matching/injection (when enabled)
`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
