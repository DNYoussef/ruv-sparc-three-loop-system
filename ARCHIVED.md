# Context Cascade archive

ARCHIVE_STATUS: FROZEN

Context Cascade and its legacy 12FA hook system are preserved as historical implementation evidence. They are not maintained, installable plugins and must not be presented as production-ready infrastructure.

The source remains intact for reference. The original README, project instructions, MCP configuration, Claude and OpenCode plugin entrypoints, nested 12FA plugin manifests, and audit verifier are retained beside their former locations with `.archived` suffixes so they cannot be loaded accidentally.

Run `npm run verify` to confirm that the archive boundary remains closed, every archived entrypoint still matches its checked SHA-256 identity, both reconciliation ledgers retain their expected metadata-only shape, and all 52 tracked 12FA hook files are still present.
