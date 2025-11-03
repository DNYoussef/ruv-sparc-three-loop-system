# Pull Request Template with Swarm Configuration

## Description

<!-- Provide a brief description of the changes in this PR -->

## Type of Change

<!-- Mark the relevant option with an 'x' -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🎨 Code style update (formatting, renaming)
- [ ] ♻️ Code refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] ✅ Test update
- [ ] 🔧 Build configuration change
- [ ] 🔒 Security fix

## Swarm Configuration

<!-- Configure automated review swarm for this PR -->

### Review Topology

<!-- Select one -->
- [ ] 🔄 Auto (based on PR size and complexity)
- [ ] 🔁 Ring (simple PRs, 2-3 reviewers)
- [ ] 🕸️ Mesh (moderate PRs, 3-5 reviewers)
- [ ] 🏗️ Hierarchical (complex PRs, 5+ reviewers)

### Max Agents
<!-- Number of review agents to spawn (1-10) -->
Max Agents: `5`

### Auto-spawn Agents
<!-- Enable automatic agent spawning based on file changes -->
- [x] Yes
- [ ] No

### Priority
<!-- Review priority level -->
- [ ] 🔴 Critical (requires immediate review)
- [x] 🟡 High (review within 24 hours)
- [ ] 🟢 Medium (review within 48 hours)
- [ ] ⚪ Low (review when available)

### Required Review Agents

<!-- Mark the agents that MUST review this PR -->

- [x] 🔒 Security (always required)
- [x] 🎨 Style (always required)
- [ ] ⚡ Performance
- [ ] 🏗️ Architecture
- [ ] ♿ Accessibility
- [ ] 🌍 i18n
- [ ] 🗄️ Database
- [ ] 📚 Documentation
- [ ] 🧪 Testing

### Optional Review Focus Areas

<!-- Additional areas that should be reviewed if relevant -->

- [ ] API design and contracts
- [ ] Error handling
- [ ] Logging and monitoring
- [ ] Configuration management
- [ ] Cache strategy
- [ ] State management
- [ ] Authentication/Authorization
- [ ] Data validation
- [ ] Migration strategy

## Tasks for Swarm

<!-- Define specific tasks for the review swarm -->

- [ ] Review security implications of changes
- [ ] Validate performance impact
- [ ] Check code style compliance
- [ ] Verify test coverage
- [ ] Review documentation updates
- [ ] Validate API contracts
- [ ] Check for breaking changes
- [ ] Review error handling

## Changes Made

<!-- Provide detailed description of changes -->

### Added
-

### Changed
-

### Removed
-

### Fixed
-

## Testing

<!-- Describe the testing you've performed -->

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated
- [ ] Manual testing completed

### Test Results
```
# Paste test results here
```

## Screenshots (if applicable)

<!-- Add screenshots for UI changes -->

### Before
<!-- Screenshot before changes -->

### After
<!-- Screenshot after changes -->

## Performance Impact

<!-- For performance-related changes -->

### Benchmarks
```
# Paste benchmark results here
```

### Metrics
- Execution time:
- Memory usage:
- Bundle size impact:

## Breaking Changes

<!-- List any breaking changes and migration steps -->

### Breaking Change Details
<!-- Describe what breaks and why -->

### Migration Guide
<!-- Step-by-step migration instructions -->

1.
2.
3.

## Dependencies

<!-- List any new dependencies or dependency updates -->

### Added Dependencies
-

### Updated Dependencies
-

### Removed Dependencies
-

## Related Issues/PRs

<!-- Link to related issues and PRs -->

Closes #
Related to #

## Checklist

<!-- Mark completed items with 'x' -->

### Code Quality
- [ ] Code follows project style guidelines
- [ ] Self-review performed
- [ ] Comments added for complex logic
- [ ] No console.log or debug code remaining
- [ ] Error handling implemented
- [ ] Edge cases handled

### Testing
- [ ] All tests passing locally
- [ ] New tests added for new features
- [ ] Test coverage maintained/improved
- [ ] Manual testing completed

### Documentation
- [ ] README updated (if needed)
- [ ] API documentation updated (if needed)
- [ ] Inline code documentation added
- [ ] CHANGELOG updated (if applicable)

### Security
- [ ] No sensitive data in code
- [ ] No hardcoded credentials
- [ ] Security implications reviewed
- [ ] Input validation added
- [ ] SQL injection prevention verified

### Deployment
- [ ] Database migrations included (if needed)
- [ ] Environment variables documented (if needed)
- [ ] Deployment steps documented (if needed)
- [ ] Rollback strategy defined (if needed)

## Review Instructions

<!-- Special instructions for reviewers -->

### Focus Areas
<!-- Specific areas that need careful review -->

1.
2.
3.

### Known Issues
<!-- Any known issues or limitations -->

-

### Testing Instructions
<!-- How reviewers can test the changes -->

1.
2.
3.

## Additional Context

<!-- Any additional context, background, or information -->

---

<!--
🤖 AUTOMATED REVIEW SWARM WILL BE TRIGGERED WHEN:
- PR is opened
- PR label changes to "ready-for-review"
- PR is marked as ready for review (if draft)

Review status will be posted as a comment below.
-->
