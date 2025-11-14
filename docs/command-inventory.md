# Command Inventory - All Sources

**Date**: 2025-10-29
**Sources**: wshobson/commands (61), awesome-claude-code (62), user's .claude/commands

---

## Archive Download Status

1. ✅ **wshobson/commands** - 61 command files
   - Location: `/tmp/wshobson-commands`
   - Structure: workflows/ (15) + tools/ (42)

2. ❌ **qdhenry/Claude-Command-Suite** - Extraction failed (Windows path issues)
   - Issue: Files with colons in names (e.g., "svelte:a11y.md")
   - Solution needed: Manual extraction or WSL

3. ✅ **awesome-claude-code** - 62 markdown files
   - Location: `/c/Users/17175/Downloads/awesome-claude-code`

---

## Commands from wshobson/commands (61 total)

### Workflows (15 commands)

**Development Workflows:**
1. feature-development.md - Complete feature workflow
2. bug-fix.md - Bug fixing workflow
3. code-review.md - Code review process
4. tdd-cycle.md - Test-driven development
5. refactor.md - Refactoring workflow

**Testing & Quality:**
6. test-suite.md - Test suite creation
7. integration-test.md - Integration testing
8. e2e-test.md - End-to-end testing
9. performance-test.md - Performance testing
10. security-audit.md - Security audit workflow

**Deployment:**
11. ci-cd-setup.md - CI/CD pipeline setup
12. deployment.md - Deployment workflow
13. rollback.md - Rollback procedures

**Documentation:**
14. documentation.md - Documentation generation
15. api-docs.md - API documentation

### Tools (42 commands)

**AI/ML Tools (6):**
1. ai-assistant.md - AI assistance
2. ml-model.md - ML model development
3. data-pipeline.md - Data pipeline creation
4. model-training.md - Model training
5. inference-service.md - Inference deployment
6. dataset-prep.md - Dataset preparation

**DevOps Tools (8):**
1. docker-setup.md - Docker configuration
2. kubernetes.md - K8s deployment
3. terraform.md - Infrastructure as code
4. ansible.md - Configuration management
5. monitoring.md - System monitoring
6. logging.md - Log aggregation
7. alerting.md - Alert configuration
8. backup.md - Backup strategies

**Testing Tools (7):**
1. unit-test.md - Unit testing
2. api-test.md - API testing
3. load-test.md - Load testing
4. security-scan.md - Security scanning
5. dependency-check.md - Dependency auditing
6. code-coverage.md - Coverage analysis
7. mutation-test.md - Mutation testing

**Database Tools (5):**
1. db-migrate.md - Database migration
2. db-seed.md - Database seeding
3. db-backup.md - Database backup
4. query-optimize.md - Query optimization
5. schema-design.md - Schema design

**Security Tools (6):**
1. auth-setup.md - Authentication setup
2. authorization.md - Authorization framework
3. encryption.md - Encryption implementation
4. secrets-mgmt.md - Secrets management
5. vulnerability-scan.md - Vulnerability scanning
6. compliance-check.md - Compliance checking

**Utilities (10):**
1. context-save.md - Save context
2. context-restore.md - Restore context
3. api-mock.md - API mocking
4. data-transform.md - Data transformation
5. file-batch.md - Batch file operations
6. regex-helper.md - Regex assistance
7. json-validator.md - JSON validation
8. yaml-parser.md - YAML parsing
9. csv-processor.md - CSV processing
10. markdown-gen.md - Markdown generation

---

## Command Categorization

### Universal Commands (Available to ALL agents)

**File Operations:**
- file-read, file-write, file-edit, file-delete
- glob-search, grep-search, file-watch
- directory-create, directory-list
- context-save, context-restore

**Git Operations:**
- git-status, git-diff, git-log
- git-add, git-commit, git-push
- git-branch, git-checkout, git-merge
- git-stash, git-tag

**Communication & Coordination:**
- communicate-notify, communicate-report, communicate-log
- agent-delegate, agent-escalate, agent-status
- memory-store, memory-retrieve, memory-search
- task-create, task-update, task-complete

**Testing & Validation:**
- test-run, test-coverage, test-validate
- lint-check, format-check, type-check
- security-scan-basic, dependency-check

**Utilities:**
- json-validator, yaml-parser, markdown-gen
- regex-helper, data-transform
- api-mock, csv-processor

**Total Universal Commands**: ~45

---

### Specialist Commands (Role-specific)

**Development Specialists:**
- Backend: api-design, db-migrate, query-optimize, auth-setup
- Frontend: component-design, state-mgmt, ui-test, accessibility
- Mobile: platform-build, native-bridge, mobile-test, app-store
- Full-Stack: feature-complete, integration-test, e2e-test

**DevOps Specialists:**
- CI/CD: pipeline-setup, deployment, rollback, blue-green
- Infrastructure: terraform-plan, k8s-deploy, docker-build
- Monitoring: alert-config, log-aggregate, metrics-collect
- Security: vulnerability-scan, compliance-check, secrets-rotate

**Data/ML Specialists:**
- Data Engineer: pipeline-create, etl-build, data-validate
- ML Engineer: model-train, inference-deploy, experiment-track
- Data Scientist: analysis-run, visualization, hypothesis-test

**QA Specialists:**
- Test Engineer: test-suite, integration-test, performance-test
- Security Auditor: security-audit, penetration-test, threat-model
- Performance: load-test, stress-test, benchmark

**Business Specialists:**
- Product Manager: feature-spec, roadmap-plan, user-story
- Marketing: campaign-create, analytics-track, seo-optimize
- Sales: pipeline-manage, forecast-generate, lead-qualify

**Total Specialist Commands**: ~105

---

## Next Steps

1. ✅ Download archives (2/3 complete)
2. ⏳ Extract all commands from awesome-claude-code
3. ⏳ Categorize remaining commands
4. ⏳ Map commands to 90 agents
5. ⏳ Store universal commands in memory
6. ⏳ Create SOP skills for workflows
7. ⏳ Update agent-creator skill
8. ⏳ Begin agent rewrites

---

**Status**: Archive download 67% complete (2/3 sources)
**Commands Found**: 123+ (will increase with awesome-claude-code extraction)
**Ready for**: Command categorization and agent mapping
