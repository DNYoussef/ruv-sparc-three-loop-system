# Example: Semantic Release Workflow

This example demonstrates a complete semantic versioning release workflow using the GitHub Release Management skill.

## Scenario

You're managing a Node.js library with the following requirements:
- Strict semantic versioning
- Automated changelog generation
- Multi-platform npm packages
- Comprehensive testing before release
- Automated GitHub releases

## Project Structure

```
my-library/
├── packages/
│   ├── core/
│   │   ├── package.json (v1.5.0)
│   │   └── src/
│   ├── cli/
│   │   ├── package.json (v1.3.2)
│   │   └── src/
│   └── utils/
│       ├── package.json (v1.4.1)
│       └── src/
├── .github/
│   └── workflows/
│       └── release.yml
├── package.json (monorepo root)
├── version-config.yaml
└── CHANGELOG.md
```

## Step 1: Analyze Commits and Suggest Version

```bash
#!/bin/bash
# analyze-release.sh

set -euo pipefail

echo "🔍 Analyzing commits for version bump..."

# Navigate to each package and analyze
for pkg in packages/*; do
    echo ""
    echo "📦 Analyzing $pkg..."

    cd "$pkg"

    # Get current version
    CURRENT=$(grep -oP '"version":\s*"\K[^"]+' package.json)
    echo "Current version: $CURRENT"

    # Run semantic versioning analysis
    ../../resources/semantic-versioning.sh --analyze

    cd ../..
done

echo ""
echo "✅ Analysis complete!"
echo ""
echo "Next steps:"
echo "1. Review suggested version bumps"
echo "2. Run ./release-monorepo.sh to execute release"
```

## Step 2: Multi-Package Release Script

```bash
#!/bin/bash
# release-monorepo.sh

set -euo pipefail

# Configuration
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Array to track updated packages
declare -a UPDATED_PACKAGES

# Function to release single package
release_package() {
    local pkg_dir="$1"
    local pkg_name
    pkg_name=$(basename "$pkg_dir")

    log_info "Processing package: $pkg_name"

    cd "$pkg_dir"

    # Get current version
    CURRENT_VERSION=$(grep -oP '"version":\s*"\K[^"]+' package.json)

    # Analyze commits and determine new version
    log_info "Analyzing commits..."
    NEW_VERSION=$(../../resources/semantic-versioning.sh --analyze --dry-run 2>&1 | \
        grep -oP 'Suggested version: .* → \K[0-9]+\.[0-9]+\.[0-9]+' || echo "$CURRENT_VERSION")

    if [[ "$NEW_VERSION" == "$CURRENT_VERSION" ]]; then
        log_warning "No version bump needed for $pkg_name"
        cd ../..
        return 0
    fi

    log_info "Version bump: $CURRENT_VERSION → $NEW_VERSION"

    # Run validation
    if [[ "$SKIP_TESTS" != "true" ]]; then
        log_info "Running tests..."

        npm ci
        npm run lint || { log_warning "Linting failed"; }
        npm run test || { log_warning "Tests failed for $pkg_name"; cd ../..; return 1; }
        npm run build || { log_warning "Build failed for $pkg_name"; cd ../..; return 1; }

        log_success "All checks passed for $pkg_name"
    fi

    # Update version
    if [[ "$DRY_RUN" != "true" ]]; then
        ../../resources/semantic-versioning.sh --set "$NEW_VERSION"

        # Generate changelog for this package
        python ../../resources/changelog-generator.py \
            --from "v$CURRENT_VERSION" \
            --to HEAD \
            --version "$NEW_VERSION" \
            --output "CHANGELOG.md" \
            --no-emoji

        log_success "Updated $pkg_name to $NEW_VERSION"

        UPDATED_PACKAGES+=("$pkg_name:$NEW_VERSION")
    else
        log_info "DRY RUN: Would update $pkg_name to $NEW_VERSION"
    fi

    cd ../..
}

# Main release workflow
main() {
    log_info "Starting monorepo release workflow..."

    # Ensure we're on main branch
    CURRENT_BRANCH=$(git branch --show-current)
    if [[ "$CURRENT_BRANCH" != "main" ]] && [[ "$DRY_RUN" != "true" ]]; then
        log_warning "Not on main branch. Current: $CURRENT_BRANCH"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Pull latest changes
    if [[ "$DRY_RUN" != "true" ]]; then
        log_info "Pulling latest changes..."
        git pull --rebase origin main
    fi

    # Release each package
    for pkg in packages/*; do
        if [[ -d "$pkg" ]] && [[ -f "$pkg/package.json" ]]; then
            release_package "$pkg" || log_warning "Failed to release $(basename "$pkg")"
        fi
    done

    # Create unified changelog
    if [[ "$DRY_RUN" != "true" ]] && [[ ${#UPDATED_PACKAGES[@]} -gt 0 ]]; then
        log_info "Generating unified changelog..."

        # Combine all package changelogs
        {
            echo "# Release $(date +%Y-%m-%d)"
            echo ""
            echo "## Updated Packages"
            echo ""

            for update in "${UPDATED_PACKAGES[@]}"; do
                IFS=':' read -r pkg ver <<< "$update"
                echo "- **$pkg**: $ver"
            done

            echo ""

            for pkg in packages/*; do
                if [[ -f "$pkg/CHANGELOG.md" ]]; then
                    echo "### $(basename "$pkg")"
                    echo ""
                    head -n 50 "$pkg/CHANGELOG.md"
                    echo ""
                fi
            done
        } > RELEASE_NOTES.md

        log_success "Created RELEASE_NOTES.md"
    fi

    # Commit changes
    if [[ "$DRY_RUN" != "true" ]] && [[ ${#UPDATED_PACKAGES[@]} -gt 0 ]]; then
        log_info "Committing version bumps..."

        git add .
        git commit -m "chore(release): version bump

Updated packages:
$(printf '- %s\n' "${UPDATED_PACKAGES[@]}")

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

        # Create tags for each package
        for update in "${UPDATED_PACKAGES[@]}"; do
            IFS=':' read -r pkg ver <<< "$update"
            git tag -a "$pkg-v$ver" -m "Release $pkg $ver"
            log_success "Created tag: $pkg-v$ver"
        done

        # Push changes and tags
        log_info "Pushing changes and tags..."
        git push origin main
        git push --tags

        log_success "Release complete!"

        # Trigger GitHub Actions release workflow
        log_info "Triggering GitHub Actions release workflow..."

        for update in "${UPDATED_PACKAGES[@]}"; do
            IFS=':' read -r pkg ver <<< "$update"

            gh workflow run release.yml \
                -f package="$pkg" \
                -f version="$ver"
        done
    fi

    # Summary
    echo ""
    log_info "═══════════════════════════════════════"
    log_info "Release Summary"
    log_info "═══════════════════════════════════════"

    if [[ ${#UPDATED_PACKAGES[@]} -eq 0 ]]; then
        log_warning "No packages updated"
    else
        log_success "Updated ${#UPDATED_PACKAGES[@]} package(s):"
        for update in "${UPDATED_PACKAGES[@]}"; do
            IFS=':' read -r pkg ver <<< "$update"
            echo "  ✓ $pkg → $ver"
        done
    fi

    log_info "═══════════════════════════════════════"
}

# Run main workflow
main "$@"
```

## Step 3: GitHub Actions Workflow

```yaml
# .github/workflows/release.yml
name: Automated Release

on:
  workflow_dispatch:
    inputs:
      package:
        description: 'Package name (core, cli, utils)'
        required: true
      version:
        description: 'Version to release'
        required: true

  push:
    tags:
      - '*-v*'

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'
          cache: 'npm'
          registry-url: 'https://registry.npmjs.org'

      - name: Install Dependencies
        run: npm ci

      - name: Extract Package and Version
        id: vars
        run: |
          if [[ "${{ github.event_name }}" == "workflow_dispatch" ]]; then
            echo "package=${{ inputs.package }}" >> $GITHUB_OUTPUT
            echo "version=${{ inputs.version }}" >> $GITHUB_OUTPUT
          else
            TAG="${{ github.ref_name }}"
            PACKAGE="${TAG%-v*}"
            VERSION="${TAG#*-v}"
            echo "package=$PACKAGE" >> $GITHUB_OUTPUT
            echo "version=$VERSION" >> $GITHUB_OUTPUT
          fi

      - name: Validate Package
        working-directory: packages/${{ steps.vars.outputs.package }}
        run: |
          npm run lint
          npm run test
          npm run build

      - name: Publish to npm
        working-directory: packages/${{ steps.vars.outputs.package }}
        run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}

      - name: Create GitHub Release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          TAG="${{ steps.vars.outputs.package }}-v${{ steps.vars.outputs.version }}"

          # Generate release notes
          python resources/changelog-generator.py \
            --from "${TAG%-*}" \
            --to HEAD \
            --version "${{ steps.vars.outputs.version }}" \
            > release-notes.md

          # Create release
          gh release create "$TAG" \
            --title "${{ steps.vars.outputs.package }} ${{ steps.vars.outputs.version }}" \
            --notes-file release-notes.md

      - name: Notify Team
        if: success()
        run: |
          echo "✅ Released ${{ steps.vars.outputs.package }} ${{ steps.vars.outputs.version }}"
```

## Step 4: Version Configuration

```yaml
# version-config.yaml
version: 2.0.0

packages:
  - name: core
    version: 1.5.0
    path: packages/core

  - name: cli
    version: 1.3.2
    path: packages/cli

  - name: utils
    version: 1.4.1
    path: packages/utils

release:
  versioning:
    strategy: semantic
    breakingKeywords: ["BREAKING", "BREAKING CHANGE", "!"]

  validation:
    preRelease:
      - name: lint
        command: npm run lint
        required: true
      - name: test
        command: npm run test
        required: true
      - name: build
        command: npm run build
        required: true

  artifacts:
    npm:
      enabled: true
      access: public
```

## Usage Examples

### 1. Dry Run (Preview Changes)

```bash
DRY_RUN=true ./release-monorepo.sh
```

### 2. Full Release

```bash
./release-monorepo.sh
```

### 3. Skip Tests (Use with Caution)

```bash
SKIP_TESTS=true ./release-monorepo.sh
```

### 4. Analyze Only

```bash
./analyze-release.sh
```

## Expected Output

```
🔍 Analyzing commits for version bump...

📦 Analyzing packages/core...
Current version: 1.5.0
[INFO] Analyzing commits...
[INFO] New features detected
[SUCCESS] Suggested version: 1.5.0 → 1.6.0 (minor)

📦 Analyzing packages/cli...
Current version: 1.3.2
[INFO] Bug fixes detected
[SUCCESS] Suggested version: 1.3.2 → 1.3.3 (patch)

✅ Analysis complete!

Starting release workflow...
[INFO] Processing package: core
[INFO] Running tests...
[SUCCESS] All checks passed for core
[SUCCESS] Updated core to 1.6.0

[INFO] Processing package: cli
[SUCCESS] Updated cli to 1.3.3

[SUCCESS] Release complete!

═══════════════════════════════════════
Release Summary
═══════════════════════════════════════
[SUCCESS] Updated 2 package(s):
  ✓ core → 1.6.0
  ✓ cli → 1.3.3
═══════════════════════════════════════
```

## Best Practices

1. **Always run dry-run first**: Preview changes before committing
2. **Review changelogs**: Verify auto-generated changelogs are accurate
3. **Test thoroughly**: Don't skip validation steps
4. **Atomic commits**: One release commit per package batch
5. **Tagging strategy**: Use `package-vX.Y.Z` format for monorepos
6. **CI/CD integration**: Automate publishing through workflows
7. **Rollback plan**: Keep previous versions tagged for quick rollback

## Troubleshooting

### Issue: Version not detected

```bash
# Solution: Manually set version
./resources/semantic-versioning.sh --set 2.0.0 --create-tag
```

### Issue: Tests failing

```bash
# Solution: Fix tests first, then release
npm run test -- --coverage
# Fix issues, then retry release
```

### Issue: npm publish fails

```bash
# Solution: Check authentication
npm whoami
echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > .npmrc
```

## Advanced Scenarios

### Pre-release Versions

```bash
# Create beta release
./resources/semantic-versioning.sh --set 2.0.0-beta.1 --create-tag
```

### Hotfix Release

```bash
# Cherry-pick fix from main
git checkout -b hotfix/v1.5.1 v1.5.0
git cherry-pick <commit-sha>

# Release hotfix
./resources/semantic-versioning.sh --bump patch --create-tag
```

## Summary

This example demonstrates:
- ✅ Automated semantic versioning
- ✅ Multi-package monorepo releases
- ✅ Comprehensive validation
- ✅ Changelog generation
- ✅ GitHub integration
- ✅ CI/CD automation
- ✅ Dry-run testing
