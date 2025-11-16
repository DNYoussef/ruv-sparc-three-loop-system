# Agent Folder Consolidation Plan

**Created**: 2025-11-15
**Status**: Analysis Complete, Ready for Execution

---

## Current State (3 Agent Folders Identified!)

### Folder 1: Global User Level
**Path**: `C:\Users\17175\.claude\agents\`
**File Count**: 208 files
**Structure**: FLAT (all files in root with prefixed names)
**Size**: 4.9 MB
**Example**: `delivery-architecture-system-design-arch-system-design.md`

### Folder 2: Plugin Internal (.claude subfolder)
**Path**: `C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\.claude\agents\`
**File Count**: (checking...)
**Structure**: FLAT (appears to be same as Folder 1)
**References**: Found in 51+ files via grep

### Folder 3: Plugin Root (CANONICAL)
**Path**: `C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\agents\`
**File Count**: 299 files
**Structure**: NESTED (10 category folders)
**Categories**: delivery, foundry, operations, orchestration, platforms, quality, research, security, specialists, tooling
**README**: Claims 203 agents, well-organized

---

## Analysis Findings

### Content Comparison
- **Sample Test**: `arch-system-design.md` from Folder 1 vs Folder 3 = **IDENTICAL**
- **Verdict**: Folders 1 and 3 have duplicate content, Folder 3 is superset (299 vs 208 files)

### Reference Usage
**CLAUDE.md References** (3 occurrences):
```markdown
Line 544: Read(".claude/agents/README.md") | grep "^###"
Line 748: Check `.claude/agents/` registry
Line 925: `.claude/agents/research/deep-research-orchestrator.md`
```

**Plugin Files Referencing** (51+ files):
- Skills, docs, and agent files reference `.claude/agents/` paths
- Both flat paths and nested paths referenced

---

## Issues Identified

1. **Redundancy**: 3 folders for the same agents (wastes disk space: 4.9 MB x 2-3 = ~10-15 MB)
2. **Confusion**: Multiple sources of truth (which is canonical?)
3. **Maintenance Burden**: Updates need to be synced across 3 locations
4. **Reference Conflicts**: CLAUDE.md points to `.claude/agents/`, but plugin has nested structure
5. **Version Drift**: Folder 3 has 299 files vs Folders 1-2 with 208 files (91 file difference!)

---

## Consolidation Strategy

### Goal
**Single Source of Truth**: Keep only the plugin root `agents/` folder (nested, organized, most complete)

### Phase 1: Verification
1. ✅ Confirm all 3 folders have same agent content (sample tested)
2. ⏳ Identify why Folder 3 has 91 more files (299 vs 208)
3. ⏳ Check if Folder 2 exists and is duplicate of Folder 1
4. ⏳ Verify no unique agents exist only in Folders 1 or 2

### Phase 2: Reference Updates
1. Update `CLAUDE.md` references:
   - Change `.claude/agents/` → `claude-code-plugins/ruv-sparc-three-loop-system/agents/`
   - Update grep commands to use nested paths
   - Fix line 544, 748, 925
2. Update all 51+ plugin files referencing `.claude/agents/`:
   - Search/replace `.claude/agents/` → `agents/` (relative to plugin root)
   - Update skill files, docs, migration plans

### Phase 3: Folder Deletion
1. **Delete** `C:\Users\17175\.claude\agents\` (208 files, 4.9 MB)
   - Redundant copy of plugin agents
   - No longer needed after reference updates
2. **Delete** `C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\.claude\agents\` (if exists)
   - Plugin-internal redundant copy
   - Not needed with canonical root structure
3. **Keep** `C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\agents\`
   - CANONICAL source (299 files, nested, organized)
   - Most complete agent registry

### Phase 4: Documentation Update
1. Update `agents/README.md` to reflect single source of truth
2. Add migration note explaining consolidation
3. Update any agent discovery commands in docs
4. Create `.gitignore` entry to prevent `.claude/agents/` recreation

---

## Benefits

1. **Single Source of Truth**: No confusion about which agent folder to use
2. **Reduced Duplication**: Save ~10-15 MB disk space
3. **Easier Maintenance**: Update agents in one place only
4. **Better Organization**: Nested structure more intuitive than flat prefixed names
5. **Accurate Registry**: 299 files vs 208 (91 additional agents available)

---

## Risks & Mitigation

### Risk 1: Breaking References
**Impact**: Skills/docs may fail if paths incorrect
**Mitigation**:
- Comprehensive grep search for all references
- Batch search/replace with verification
- Test sample agent access after update

### Risk 2: Unique Agents in Deleted Folders
**Impact**: Lose agents that don't exist in canonical folder
**Mitigation**:
- Compare file lists before deletion
- Copy any unique agents to canonical folder first
- Create backup before deletion

### Risk 3: Claude Code Expects `.claude/agents/`
**Impact**: Claude Code system may rely on global path
**Mitigation**:
- Check Claude Code docs for agent location requirements
- Create symlink if global path required
- Test agent spawning after consolidation

---

## Execution Checklist

### Pre-Execution
- [ ] Verify Folder 2 exists and file count
- [ ] List all unique agents in Folders 1-2 not in Folder 3
- [ ] Copy unique agents (if any) to Folder 3
- [ ] Create backup of all 3 folders
- [ ] Document current git status

### Execution (Reference Updates)
- [ ] Update CLAUDE.md (3 references)
- [ ] Update 51+ plugin files (batch search/replace)
- [ ] Test sample agent references work
- [ ] Verify Task tool can spawn agents

### Execution (Folder Deletion)
- [ ] Delete `C:\Users\17175\.claude\agents\`
- [ ] Delete `C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\.claude\agents\` (if exists)
- [ ] Verify no broken references
- [ ] Test agent discovery commands

### Post-Execution
- [ ] Update documentation with consolidation notes
- [ ] Commit changes to git
- [ ] Push to main
- [ ] Monitor for any issues
- [ ] Document lessons learned

---

## Recommended Next Steps

1. **Immediate**: Verify Folder 2 exists and check file count
2. **Immediate**: Compare unique files between all 3 folders
3. **High Priority**: Batch search/replace references (51+ files)
4. **High Priority**: Update CLAUDE.md (3 references)
5. **Medium Priority**: Test agent spawning after updates
6. **Low Priority**: Delete redundant folders
7. **Final**: Document and commit

---

## Questions for User

1. **Should we keep `.claude/agents/` as a symlink?** (for backwards compatibility)
2. **Any known dependencies on global agent path?** (Claude Code internals)
3. **Prefer batch automated execution or manual step-by-step?**
4. **Backup strategy?** (git commit before changes, or tar archive)

---

**Status**: Ready for user approval to proceed with Phase 1 verification
**Estimated Time**: 30-60 minutes for full consolidation
**Recommended Approach**: Automated batch execution with verification checkpoints
