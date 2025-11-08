#!/usr/bin/env python3
"""
MECE Analysis: Find Missing Skills
Compares filesystem skills with CLAUDE.md documented skills
"""

# Filesystem skills (from directory listing - 123 skills)
filesystem_skills = {
    'agent-creator', 'agentdb', 'agentdb-advanced', 'agentdb-learning', 
    'agentdb-memory-patterns', 'agentdb-optimization', 'agentdb-vector-search',
    'baseline-replication', 'cascade-orchestrator', 'cicd-intelligent-recovery',
    'cloud-platforms', 'code-review-assistant', 'codex-auto', 'codex-reasoning',
    'compliance', 'coordination', 'database-specialists', 'debugging',
    'deep-research-orchestrator', 'dependencies', 'deployment-readiness',
    'documentation', 'dogfooding-system', 'feature-dev-complete', 
    'flow-nexus-neural', 'flow-nexus-platform', 'flow-nexus-swarm',
    'frontend-specialists', 'functionality-audit', 'gate-validation',
    'gemini-extensions', 'gemini-media', 'gemini-megacontext', 'gemini-search',
    'github-code-review', 'github-integration', 'github-multi-repo',
    'github-project-management', 'github-release-management', 'github-workflow-automation',
    'hive-mind-advanced', 'holistic-evaluation', 'hooks-automation', 'i18n-automation',
    'infrastructure', 'intent-analyzer', 'interactive-planner', 'language-specialists',
    'literature-synthesis', 'machine-learning', 'meta-tools', 'method-development',
    'micro-skill-creator', 'ml-expert', 'ml-training-debugger', 'multi-model',
    'network-security-setup', 'observability', 'pair-programming',
    'parallel-swarm-implementation', 'performance', 'performance-analysis',
    'platform', 'pptx-generation', 'production-readiness', 'prompt-architect',
    'quick-quality-check', 'reasoningbank-agentdb', 'reasoningbank-intelligence',
    'reproducibility-audit', 'research-driven-planning', 'research-publication',
    'reverse-engineer-debug', 'reverse-engineering-deep', 'reverse-engineering-firmware',
    'reverse-engineering-quick', 'sandbox-configurator', 'security', 'skill-builder',
    'skill-creator-agent', 'skill-forge', 'slash-command-encoder', 'smart-bug-fix',
    'sop-api-development', 'sop-code-review', 'sop-dogfooding-continuous-improvement',
    'sop-dogfooding-pattern-retrieval', 'sop-dogfooding-quality-detection',
    'sop-product-launch', 'sparc-methodology', 'specialized-tools', 'stream-chain',
    'style-audit', 'swarm-advanced', 'swarm-orchestration', 'testing',
    'testing-quality', 'theater-detection-audit', 'utilities', 'verification-quality',
    'web-cli-teleport', 'when-automating-workflows-use-hooks-automation',
    'when-building-backend-api-orchestrate-api-development',
    'when-collaborative-coding-use-pair-programming',
    'when-developing-complete-feature-use-feature-dev-complete',
    'when-fixing-complex-bug-use-smart-bug-fix', 'when-internationalizing-app-use-i18n-automation',
    'when-releasing-new-product-orchestrate-product-launch',
    'when-reviewing-pull-request-orchestrate-comprehensive-code-review',
    'when-using-sparc-methodology-use-sparc-workflow', 'workflow'
}

# Skills documented in CLAUDE.md (from backtick extraction - 163 skills)
claude_md_skills = {
    'coder', 'coder-enhanced', 'reviewer', 'tester', 'planner', 'researcher',
    'api-designer', 'technical-debt-manager', 'tdd-london-swarm', 'production-validator',
    'e2e-testing-specialist', 'performance-testing-agent', 'security-testing-agent',
    'visual-regression-agent', 'contract-testing-agent', 'chaos-engineering-agent',
    'audit-pipeline-orchestrator', 'react-developer', 'vue-developer',
    'ui-component-builder', 'css-styling-specialist', 'accessibility-specialist',
    'frontend-performance-optimizer', 'database-design-specialist',
    'query-optimization-agent', 'database-migration-agent', 'data-pipeline-engineer',
    'cache-strategy-agent', 'database-backup-recovery-agent', 'data-ml-model',
    'api-documentation-specialist', 'developer-documentation-agent',
    'knowledge-base-manager', 'technical-writing-agent', 'architecture-diagram-generator',
    'docs-api-openapi', 'hierarchical-coordinator', 'mesh-coordinator',
    'adaptive-coordinator', 'collective-intelligence-coordinator', 'swarm-memory-manager',
    'consensus-validator', 'swarm-health-monitor', 'byzantine-coordinator', 'raft-manager',
    'gossip-coordinator', 'consensus-builder', 'crdt-synchronizer', 'quorum-manager',
    'security-manager', 'perf-analyzer', 'performance-benchmarker', 'task-orchestrator',
    'memory-coordinator', 'smart-agent', 'github-modes', 'pr-manager', 'code-review-swarm',
    'issue-tracker', 'release-manager', 'workflow-automation', 'project-board-sync',
    'repo-architect', 'multi-repo-swarm', 'sparc-coord', 'sparc-coder', 'specification',
    'pseudocode', 'architecture', 'refinement', 'backend-dev', 'mobile-dev', 'ml-developer',
    'cicd-engineer', 'api-docs', 'system-architect', 'code-analyzer',
    'base-template-generator', 'migration-planner', 'swarm-init', 'data-steward',
    'ethics-agent', 'archivist', 'evaluator', 'research-driven-planning',
    'sparc-methodology', 'interactive-planner', 'intent-analyzer',
    'parallel-swarm-implementation', 'feature-dev-complete', 'pair-programming',
    'cascade-orchestrator', 'functionality-audit', 'theater-detection-audit',
    'production-readiness', 'quick-quality-check', 'code-review-assistant',
    'testing-quality', 'smart-bug-fix', 'reverse-engineer-debug', 'ml-training-debugger',
    'sop-dogfooding-quality-detection', 'sop-dogfooding-pattern-retrieval',
    'sop-dogfooding-continuous-improvement', 'python-specialist', 'typescript-specialist',
    'react-specialist', 'when-building-backend-api-orchestrate-api-development',
    'sop-api-development', 'sql-database-specialist', 'ml-expert', 'pptx-generation',
    'documentation', 'network-security-setup', 'security', 'sop-code-review',
    'wcag-accessibility', 'agent-creator', 'skill-builder', 'skill-creator-agent',
    'micro-skill-creator', 'slash-command-encoder', 'base-template-generator',
    'prompt-architect', 'skill-forge', 'performance-analysis', 'style-audit',
    'dependencies', 'github-code-review', 'github-project-management',
    'github-workflow-automation', 'github-release-management', 'github-multi-repo',
    'github-integration', 'sop-product-launch', 'gemini-search', 'gemini-megacontext',
    'gemini-media', 'gemini-extensions', 'codex-auto', 'codex-reasoning', 'multi-model',
    'agentdb', 'agentdb-memory-patterns', 'agentdb-learning', 'agentdb-optimization',
    'agentdb-vector-search', 'agentdb-advanced', 'reasoningbank-intelligence',
    'reasoningbank-agentdb', 'swarm-orchestration', 'swarm-advanced', 'hive-mind-advanced',
    'coordination', 'flow-nexus-platform', 'flow-nexus-swarm', 'flow-nexus-neural',
    'hooks-automation', 'workflow', 'i18n-automation', 'stream-chain', 'debugging',
    'verification-quality', 'meta-tools', 'specialized-tools', 'web-cli-teleport',
    'platform', 'sandbox-configurator', 'baseline-replication', 'literature-synthesis',
    'method-development', 'holistic-evaluation', 'deployment-readiness',
    'deep-research-orchestrator', 'reproducibility-audit', 'research-publication',
    'gate-validation', 'reverse-engineering-quick', 'reverse-engineering-deep',
    'reverse-engineering-firmware', 'aws-specialist', 'kubernetes-specialist',
    'docker-containerization', 'terraform-iac', 'opentelemetry-observability',
    'cicd-intelligent-recovery'
}

# Perform MECE Analysis
missing_from_claude = filesystem_skills - claude_md_skills
invalid_references = claude_md_skills - filesystem_skills
correctly_documented = filesystem_skills & claude_md_skills

print("=" * 90)
print("MECE ANALYSIS: MISSING SKILLS AUDIT")
print("=" * 90)
print(f"\nFilesystem Skills Total: {len(filesystem_skills)}")
print(f"CLAUDE.md Skills Total: {len(claude_md_skills)}")
print(f"Correctly Documented: {len(correctly_documented)}")
print(f"\nMissing from CLAUDE.md: {len(missing_from_claude)}")
print(f"Invalid References (in CLAUDE.md but not in filesystem): {len(invalid_references)}")

print("\n" + "=" * 90)
print("MISSING SKILLS CATEGORIZED BY MECE TAXONOMY")
print("=" * 90)

# Categorize missing skills using MECE
categorized = {
    'Self-Improvement & Dogfooding': [],
    'Language & Framework Specialists': [],
    'Cloud & Infrastructure': [],
    'Testing & Validation': [],
    'Utilities & Tools': [],
    'Development Lifecycle': [],
}

for skill in sorted(missing_from_claude):
    if 'dogfooding' in skill or 'audit-pipeline' in skill:
        categorized['Self-Improvement & Dogfooding'].append(skill)
    elif any(x in skill for x in ['cloud', 'infrastructure', 'observability', 'docker', 'kubernetes', 'aws', 'terraform']):
        categorized['Cloud & Infrastructure'].append(skill)
    elif any(x in skill for x in ['frontend', 'database', 'machine-learning', 'language']):
        categorized['Language & Framework Specialists'].append(skill)
    elif any(x in skill for x in ['testing', 'compliance']):
        categorized['Testing & Validation'].append(skill)
    elif any(x in skill for x in ['performance', 'utilities']):
        categorized['Utilities & Tools'].append(skill)
    else:
        categorized['Development Lifecycle'].append(skill)

for category in sorted(categorized.keys()):
    if categorized[category]:
        print(f"\n{category.upper()} ({len(categorized[category])})")
        print("-" * 90)
        for skill in sorted(categorized[category]):
            print(f"  - {skill}")

print("\n" + "=" * 90)
print("INVALID REFERENCES IN CLAUDE.MD")
print("=" * 90)
if invalid_references:
    print(f"\nFound {len(invalid_references)} skills mentioned in CLAUDE.md but not in filesystem:")
    for skill in sorted(invalid_references):
        print(f"  - {skill}")
else:
    print("  ✓ No invalid references found")

print("\n" + "=" * 90)
print(f"SUMMARY")
print("=" * 90)
print(f"Missing Skills to Add: {len(missing_from_claude)}")
print(f"Expected Total (Filesystem): {len(filesystem_skills)}")
print(f"Current CLAUDE.md Coverage: {len(correctly_documented)}/{len(filesystem_skills)} ({100*len(correctly_documented)//len(filesystem_skills)}%)")
