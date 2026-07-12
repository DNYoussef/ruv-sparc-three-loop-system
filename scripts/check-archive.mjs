import { access, readFile, readdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const errors = [];

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function walk(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await walk(path)));
    else files.push(path);
  }
  return files;
}

const archivePath = join(root, "ARCHIVED.md");
if (!(await exists(archivePath))) errors.push("ARCHIVED.md is missing");
else if (!(await readFile(archivePath, "utf8")).includes("ARCHIVE_STATUS: FROZEN"))
  errors.push("ARCHIVED.md is missing ARCHIVE_STATUS: FROZEN");

const expectedReadme = `# Context Cascade (archived)

ARCHIVE_STATUS: FROZEN

This repository is preserved as historical implementation evidence. It is not maintained, installable, published, or production-ready.

The original README, project instructions, MCP configuration, Claude/OpenCode plugin entrypoints, and legacy verifier remain byte-for-byte preserved under \`.archived\` names. The 52-file 12FA hook tree is retained unchanged.

\`npm run verify\` checks the non-operational archive boundary and preservation manifests. It does not start the retired plugin or execute its hooks.

See [ARCHIVED.md](ARCHIVED.md) for the archive decision.
`;
if ((await readFile(join(root, "README.md"), "utf8")).replaceAll("\r\n", "\n") !== expectedReadme)
  errors.push("README.md must remain the archive-only non-operational notice");

const inventoryPath = join(root, "docs", "archive", "12fa-hooks-inventory.txt");
const inventory = (await readFile(inventoryPath, "utf8")).split(/\r?\n/).filter(Boolean).sort();
if (inventory.length !== 52) errors.push(`12FA inventory must contain 52 paths, found ${inventory.length}`);
const actualHooks = (await walk(join(root, "hooks", "12fa")))
  .map((path) => relative(root, path).replaceAll("\\", "/"))
  .sort();
if (inventory.join("\n") !== actualHooks.join("\n")) errors.push("12FA inventory differs from preserved hook tree");

const activeEntrypoints = [];
const archivedEntrypoints = [
  "README.md.archived",
  "CLAUDE.md.archived",
  ".mcp.json.archived",
  ".claude-plugin/plugin.json.archived",
  ".claude-plugin/marketplace.json.archived",
  "plugins/12fa-core/plugin.json.archived",
  "plugins/12fa-security/plugin.json.archived",
  "plugins/12fa-swarm/plugin.json.archived",
  "plugins/12fa-three-loop/plugin.json.archived",
  "plugins/12fa-visual-docs/plugin.json.archived",
  "scripts/verify-audit-system.js.archived",
  "opencode-plugin/package.json.archived",
  "opencode-plugin/src/index.ts.archived",
  "opencode-plugin/scripts/dev-simulate.mjs.archived",
  "opencode-plugin/README.md.archived",
];
for (const path of archivedEntrypoints) {
  if (!(await exists(join(root, path)))) errors.push(`archived entrypoint is missing: ${path}`);
}

const hashManifestPath = join(root, "docs", "archive", "archived-entrypoints.sha256");
if (!(await exists(hashManifestPath))) errors.push("archived entrypoint hash manifest is missing");
else {
  const hashRows = (await readFile(hashManifestPath, "utf8")).split(/\r?\n/).filter(Boolean);
  const expectedPaths = [...archivedEntrypoints].sort();
  const recordedPaths = [];
  for (const row of hashRows) {
    const match = row.match(/^([0-9a-f]{64})  (.+)$/);
    if (!match) {
      errors.push(`invalid archived hash row: ${row}`);
      continue;
    }
    const [, expectedHash, path] = match;
    recordedPaths.push(path);
    if (!(await exists(join(root, path)))) continue;
    const actualHash = createHash("sha256").update(await readFile(join(root, path))).digest("hex");
    if (actualHash !== expectedHash) errors.push(`archived hash mismatch: ${path}`);
  }
  if (recordedPaths.sort().join("\n") !== expectedPaths.join("\n"))
    errors.push("archived hash manifest paths differ from required entrypoints");
}

const expectedLiveHead = "fbe77cffbd8d1df5588d220e2450e6fd6fd389d0";
const expectedOriginMain = "12bc124ee69b524b17cd78ea7186204c5fd935b8";
const identityPattern = /^(?:git:[0-9a-f]{40}|(?:sha256|tree-sha256):[0-9a-f]{64})$/;

const dirtyLedger = (await readFile(join(root, "docs", "archive", "live-dirty-inventory.tsv"), "utf8")).split(
  /\r?\n/,
);
if (!dirtyLedger.includes(`# live-head: ${expectedLiveHead}`))
  errors.push("live dirty ledger records the wrong HEAD");
if (!dirtyLedger.includes(`# origin-main: ${expectedOriginMain}`))
  errors.push("live dirty ledger records the wrong origin/main");
if (!dirtyLedger.includes("# entries: 218")) errors.push("live dirty ledger entry count header is not 218");
const dirtyRows = dirtyLedger.filter((row) => row && !row.startsWith("#"));
const dirtyPaths = new Set();
for (const row of dirtyRows) {
  const fields = row.split("\t");
  if (fields.length !== 3 || !/^[ MADRCU?!]{2}$/.test(fields[0]) || !fields[1] || !identityPattern.test(fields[2])) {
    errors.push(`invalid live dirty ledger row: ${row}`);
    continue;
  }
  if (dirtyPaths.has(fields[1])) errors.push(`duplicate live dirty ledger path: ${fields[1]}`);
  dirtyPaths.add(fields[1]);
}
if (dirtyRows.length !== 218) errors.push(`live dirty ledger must contain 218 rows, found ${dirtyRows.length}`);

const aheadLedger = (await readFile(join(root, "docs", "archive", "live-ahead-commit.tsv"), "utf8")).split(
  /\r?\n/,
);
if (!aheadLedger.includes(`# commit: ${expectedLiveHead}`)) errors.push("ahead ledger records the wrong commit");
if (!aheadLedger.includes(`# parent: ${expectedOriginMain}`)) errors.push("ahead ledger records the wrong parent");
if (!aheadLedger.includes("# entries: 9")) errors.push("ahead ledger entry count header is not 9");
const aheadRows = aheadLedger.filter((row) => row && !row.startsWith("#"));
const aheadPaths = new Set();
for (const row of aheadRows) {
  const fields = row.split("\t");
  if (
    fields.length !== 4 ||
    !/^[AMD]$/.test(fields[0]) ||
    !fields[1] ||
    !/^git:[0-9a-f]{40}$/.test(fields[2]) ||
    !/^git:[0-9a-f]{40}$/.test(fields[3])
  ) {
    errors.push(`invalid ahead ledger row: ${row}`);
    continue;
  }
  if (aheadPaths.has(fields[1])) errors.push(`duplicate ahead ledger path: ${fields[1]}`);
  aheadPaths.add(fields[1]);
}
if (aheadRows.length !== 9) errors.push(`ahead ledger must contain 9 rows, found ${aheadRows.length}`);
for (const path of [
  join(root, "CLAUDE.md"),
  join(root, ".mcp.json"),
  join(root, ".claude-plugin", "plugin.json"),
  join(root, ".claude-plugin", "marketplace.json"),
  join(root, "opencode-plugin", "package.json"),
  join(root, "opencode-plugin", "src", "index.ts"),
  join(root, "opencode-plugin", "scripts", "dev-simulate.mjs"),
  join(root, "opencode-plugin", "README.md"),
]) {
  if (await exists(path)) activeEntrypoints.push(relative(root, path));
}
for (const path of await walk(join(root, "plugins"))) {
  if (path.endsWith("plugin.json")) activeEntrypoints.push(relative(root, path));
}
if (activeEntrypoints.length > 0) errors.push(`active install/load entrypoints: ${activeEntrypoints.join(", ")}`);

const pkg = JSON.parse(await readFile(join(root, "package.json"), "utf8"));
if (pkg.private !== true) errors.push("package.json must be private");
if (pkg.scripts?.verify !== "node scripts/check-archive.mjs")
  errors.push("package.json verify must run the archive gate");
const allowedPackageKeys = ["description", "license", "name", "private", "repository", "scripts", "version"];
if (Object.keys(pkg).sort().join("\n") !== allowedPackageKeys.join("\n"))
  errors.push("package.json contains non-archive metadata or publication entrypoints");
if (Object.keys(pkg.scripts ?? {}).join("\n") !== "verify")
  errors.push("package.json must expose only the verify script");

const activeLoaders = [];
const archivedHookImport = ["hooks", "12fa"].join("/");
for (const path of await walk(join(root, "scripts"))) {
  if (!/\.(?:c?js|mjs)$/.test(path)) continue;
  if ((await readFile(path, "utf8")).includes(archivedHookImport)) activeLoaders.push(relative(root, path));
}
if (activeLoaders.length > 0) errors.push(`active 12FA loaders: ${activeLoaders.join(", ")}`);

const registeredHooks = await readFile(join(root, "hooks", "hooks.json"), "utf8");
if (/hooks[\\/]12fa/.test(registeredHooks)) errors.push("hooks/hooks.json still registers archived hooks");

if (errors.length > 0) {
  console.error(errors.join("\n"));
  console.error(
    `F013_F035_ARCHIVE_RED errors=${errors.length} hooks=${actualHooks.length} active_entrypoints=${activeEntrypoints.length} active_loaders=${activeLoaders.length}`,
  );
  process.exit(1);
}

console.log(
  `F013_F035_ARCHIVE_GREEN hooks=${actualHooks.length} active_entrypoints=0 active_loaders=0`,
);
