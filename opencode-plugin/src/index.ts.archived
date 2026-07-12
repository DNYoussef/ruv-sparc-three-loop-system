import path from 'path';
import { loadCommands, loadSkills } from './loader.js';
import { getRepoRoot } from './markdown.js';
import { OpenCodeHost, PluginOptions } from './registry.js';
import { SkillIndex } from './skill_index.js';

export interface PluginInitOptions extends PluginOptions {}

export default async function plugin(host: OpenCodeHost, options: PluginInitOptions = {}): Promise<void> {
  const repoRoot =
    options.repoRoot ??
    process.env.CONTEXT_CASCADE_ROOT ??
    (await getRepoRoot(path.resolve(process.cwd(), '..'))); // assume invoked from opencode-plugin/

  const commandPrefix = options.commandPrefix ?? process.env.CC_COMMAND_PREFIX ?? 'cc';
  const enableSkillInjection = options.enableSkillInjection ?? process.env.CC_ENABLE_SKILLS === '1';

  const commands = await loadCommands({ repoRoot });
  const skills = await loadSkills({ repoRoot });
  const skillIndex = new SkillIndex(skills, {
    matchCount: options.skillMatchCount,
    threshold: options.skillMatchThreshold,
  });

  registerListCommand(host, commandPrefix, commands, skills);

  for (const cmd of commands) {
    const name = withPrefix(commandPrefix, cmd.name);
    host.registerCommand({
      name,
      description: cmd.description,
      run: async (args) => expandCommandBody(cmd.body, args),
    });

    if (cmd.alias) {
      host.registerCommand({
        name: withPrefix(commandPrefix, cmd.alias),
        description: `${cmd.description} (alias)`,
        run: async (args) => expandCommandBody(cmd.body, args),
      });
    }
  }

  if (enableSkillInjection && host.onUserMessage && host.addSystemContext) {
    host.onUserMessage(async (message) => {
      const matches = skillIndex.match(message);
      for (const match of matches) {
        const inject = `\[Context Cascade Skill Injected: ${match.skill.name}]\n${match.skill.body}`;
        host.addSystemContext?.(inject);
      }
    });
  } else if (enableSkillInjection) {
    // fallback command to request injection manually
    host.registerCommand({
      name: withPrefix(commandPrefix, '/inject'),
      description: 'Injects relevant Context Cascade skills into the prompt based on the provided text.',
      run: async (args) => {
        const matches = skillIndex.match(args);
        if (!matches.length) return 'No relevant skills found.';
        return matches
          .map((m) => `\[Context Cascade Skill Injected: ${m.skill.name}]\n${m.skill.body}`)
          .join('\n\n');
      },
    });
  }
}

function withPrefix(prefix: string, name: string): string {
  const trimmed = name.startsWith('/') ? name.slice(1) : name;
  return `/${prefix}:${trimmed}`;
}

function expandCommandBody(body: string, args: string): string {
  const tokens = args.trim().split(/\s+/);
  let output = body.replace(/\$ARGUMENTS/g, args);
  tokens.forEach((token, idx) => {
    const placeholder = new RegExp(`\\$${idx + 1}`, 'g');
    output = output.replace(placeholder, token);
  });
  return output;
}

function registerListCommand(
  host: OpenCodeHost,
  prefix: string,
  commands: { name: string; alias?: string; description: string }[],
  skills: { name: string; description: string }[],
): void {
  host.registerCommand({
    name: `/${prefix}`,
    description: 'Lists available Context Cascade commands and top skills.',
    run: async () => {
      const commandLines = commands
        .map((cmd) => `- ${withPrefix(prefix, cmd.name)} — ${cmd.description}${cmd.alias ? ` (alias ${withPrefix(prefix, cmd.alias)})` : ''}`)
        .join('\n');
      const skillLines = skills
        .slice(0, 10)
        .map((s) => `- ${s.name}: ${s.description}`)
        .join('\n');
      return `Available commands:\n${commandLines}\n\nTop skills sample:\n${skillLines}`;
    },
  });
}
