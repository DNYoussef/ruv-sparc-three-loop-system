# Example: Multi-Platform Release Pipeline

This example demonstrates a complete multi-platform release workflow with cross-compilation, artifact packaging, and deployment across npm, Docker, and GitHub Releases.

## Scenario

You're releasing a CLI tool that needs to:
- Build native binaries for Linux, macOS, and Windows
- Support x64 and arm64 architectures
- Publish to npm registry
- Create Docker images for multiple platforms
- Upload signed artifacts to GitHub Releases
- Generate checksums and signatures
- Support staged rollouts

## Project Structure

```
cli-tool/
├── .github/
│   └── workflows/
│       ├── build.yml
│       ├── release.yml
│       └── rollback.yml
├── resources/
│   ├── release-automation.js
│   ├── asset-packager.py
│   └── version-config.yaml
├── scripts/
│   ├── build-binaries.sh
│   ├── build-docker.sh
│   └── sign-artifacts.sh
├── Dockerfile
├── package.json
└── README.md
```

## Step 1: Multi-Platform Build Script

```bash
#!/bin/bash
# scripts/build-binaries.sh

set -euo pipefail

# Configuration
VERSION="${VERSION:-1.0.0}"
PLATFORMS="${PLATFORMS:-linux darwin windows}"
ARCHITECTURES="${ARCHITECTURES:-amd64 arm64}"
OUTPUT_DIR="${OUTPUT_DIR:-dist}"

# Binary name
BINARY_NAME="cli-tool"

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

# Clean and create output directory
clean_output() {
    log_info "Cleaning output directory..."
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
}

# Build for specific platform and architecture
build_binary() {
    local platform="$1"
    local arch="$2"

    log_info "Building for $platform/$arch..."

    # Map Go architectures
    local go_os="$platform"
    local go_arch="$arch"

    if [[ "$arch" == "amd64" ]]; then
        go_arch="amd64"
    elif [[ "$arch" == "arm64" ]]; then
        go_arch="arm64"
    fi

    # Set output filename
    local output_name="$BINARY_NAME-$platform-$arch"
    if [[ "$platform" == "windows" ]]; then
        output_name="$output_name.exe"
    fi

    local output_path="$OUTPUT_DIR/$output_name"

    # Build with Go
    env GOOS="$go_os" GOARCH="$go_arch" CGO_ENABLED=0 \
        go build -trimpath -ldflags="-s -w -X main.version=$VERSION" \
        -o "$output_path" \
        ./cmd/cli

    if [[ -f "$output_path" ]]; then
        local size
        size=$(du -h "$output_path" | cut -f1)
        log_success "Built $output_name ($size)"
    else
        log_warning "Failed to build $output_name"
        return 1
    fi

    # Make executable on Unix
    if [[ "$platform" != "windows" ]]; then
        chmod +x "$output_path"
    fi

    # Strip binaries for smaller size (Unix only)
    if command -v strip &> /dev/null && [[ "$platform" != "windows" ]]; then
        strip "$output_path" 2>/dev/null || true
        local new_size
        new_size=$(du -h "$output_path" | cut -f1)
        log_info "Stripped to $new_size"
    fi
}

# Build for all platforms
build_all() {
    log_info "Starting multi-platform build..."

    local total=0
    local success=0
    local failed=0

    for platform in $PLATFORMS; do
        for arch in $ARCHITECTURES; do
            total=$((total + 1))

            if build_binary "$platform" "$arch"; then
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi
        done
    done

    echo ""
    log_info "═══════════════════════════════════════"
    log_info "Build Summary"
    log_info "═══════════════════════════════════════"
    log_success "Success: $success/$total"

    if [[ $failed -gt 0 ]]; then
        log_warning "Failed: $failed/$total"
    fi

    log_info "═══════════════════════════════════════"

    return $failed
}

# Generate build info
generate_build_info() {
    log_info "Generating build information..."

    cat > "$OUTPUT_DIR/build-info.json" << EOF
{
  "version": "$VERSION",
  "build_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "platforms": [
$(for platform in $PLATFORMS; do
    for arch in $ARCHITECTURES; do
        echo "    {\"os\": \"$platform\", \"arch\": \"$arch\"},"
    done
done | sed '$ s/,$//')
  ]
}
EOF

    log_success "Build info written to $OUTPUT_DIR/build-info.json"
}

# Main execution
main() {
    clean_output
    build_all || exit 1
    generate_build_info

    log_success "Build complete! Artifacts in $OUTPUT_DIR/"

    # List built artifacts
    echo ""
    log_info "Built artifacts:"
    ls -lh "$OUTPUT_DIR" | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
}

main "$@"
```

## Step 2: Docker Multi-Platform Build

```bash
#!/bin/bash
# scripts/build-docker.sh

set -euo pipefail

VERSION="${VERSION:-latest}"
REGISTRY="${REGISTRY:-docker.io}"
IMAGE_NAME="${IMAGE_NAME:-cli-tool}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"

log_info() {
    echo "🐳 [INFO] $*"
}

log_success() {
    echo "✅ [SUCCESS] $*"
}

# Setup buildx
setup_buildx() {
    log_info "Setting up Docker buildx..."

    docker buildx create --use --name multi-arch-builder 2>/dev/null || \
    docker buildx use multi-arch-builder

    docker buildx inspect --bootstrap

    log_success "Buildx ready"
}

# Build multi-platform images
build_images() {
    log_info "Building multi-platform Docker images..."

    docker buildx build \
        --platform "$PLATFORMS" \
        --tag "$REGISTRY/$IMAGE_NAME:$VERSION" \
        --tag "$REGISTRY/$IMAGE_NAME:latest" \
        --build-arg VERSION="$VERSION" \
        --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        --push \
        .

    log_success "Multi-platform images built and pushed"
}

# Generate image manifest
generate_manifest() {
    log_info "Generating image manifest..."

    docker buildx imagetools inspect "$REGISTRY/$IMAGE_NAME:$VERSION" > \
        "dist/docker-manifest.txt"

    log_success "Manifest saved to dist/docker-manifest.txt"
}

main() {
    setup_buildx
    build_images
    generate_manifest

    log_success "Docker build complete!"
}

main "$@"
```

## Step 3: Dockerfile for Multi-Platform

```dockerfile
# Dockerfile
FROM --platform=$BUILDPLATFORM golang:1.21-alpine AS builder

# Build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG VERSION=dev
ARG BUILD_DATE

# Install dependencies
RUN apk add --no-cache git ca-certificates

WORKDIR /build

# Copy go modules
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build binary for target platform
RUN --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=cache,target=/go/pkg \
    case "$TARGETPLATFORM" in \
        "linux/amd64") \
            GOARCH=amd64 ;; \
        "linux/arm64") \
            GOARCH=arm64 ;; \
        *) \
            echo "Unsupported platform: $TARGETPLATFORM" && exit 1 ;; \
    esac && \
    CGO_ENABLED=0 GOOS=linux GOARCH=$GOARCH \
    go build -trimpath -ldflags="-s -w -X main.version=$VERSION -X main.buildDate=$BUILD_DATE" \
    -o cli-tool ./cmd/cli

# Final minimal image
FROM scratch

# Copy CA certificates
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy binary
COPY --from=builder /build/cli-tool /cli-tool

# Metadata
LABEL org.opencontainers.image.title="CLI Tool"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"

ENTRYPOINT ["/cli-tool"]
```

## Step 4: Comprehensive Release Workflow

```yaml
# .github/workflows/release.yml
name: Multi-Platform Release

on:
  push:
    tags:
      - 'v*'

  workflow_dispatch:
    inputs:
      version:
        description: 'Version to release'
        required: true
      dry_run:
        description: 'Dry run (no publishing)'
        type: boolean
        default: false

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-binaries:
    name: Build Binaries
    runs-on: ubuntu-latest

    strategy:
      matrix:
        include:
          - os: linux
            arch: amd64
          - os: linux
            arch: arm64
          - os: darwin
            arch: amd64
          - os: darwin
            arch: arm64
          - os: windows
            arch: amd64
          - os: windows
            arch: arm64

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Go
        uses: actions/setup-go@v4
        with:
          go-version: '1.21'

      - name: Extract Version
        id: version
        run: |
          if [[ "${{ github.event_name }}" == "push" ]]; then
            VERSION="${{ github.ref_name }}"
            VERSION="${VERSION#v}"
          else
            VERSION="${{ inputs.version }}"
          fi
          echo "version=$VERSION" >> $GITHUB_OUTPUT

      - name: Build Binary
        env:
          GOOS: ${{ matrix.os }}
          GOARCH: ${{ matrix.arch }}
          VERSION: ${{ steps.version.outputs.version }}
        run: |
          BINARY_NAME="cli-tool-${{ matrix.os }}-${{ matrix.arch }}"

          if [[ "${{ matrix.os }}" == "windows" ]]; then
            BINARY_NAME="$BINARY_NAME.exe"
          fi

          CGO_ENABLED=0 go build \
            -trimpath \
            -ldflags="-s -w -X main.version=$VERSION" \
            -o "dist/$BINARY_NAME" \
            ./cmd/cli

          # Make executable
          chmod +x "dist/$BINARY_NAME"

      - name: Upload Artifact
        uses: actions/upload-artifact@v3
        with:
          name: binaries-${{ matrix.os }}-${{ matrix.arch }}
          path: dist/*

  package-release:
    name: Package Release
    needs: build-binaries
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Download All Artifacts
        uses: actions/download-artifact@v3
        with:
          path: dist-temp

      - name: Consolidate Artifacts
        run: |
          mkdir -p dist
          find dist-temp -type f -exec cp {} dist/ \;

      - name: Extract Version
        id: version
        run: |
          if [[ "${{ github.event_name }}" == "push" ]]; then
            VERSION="${{ github.ref_name }}"
            VERSION="${VERSION#v}"
          else
            VERSION="${{ inputs.version }}"
          fi
          echo "version=$VERSION" >> $GITHUB_OUTPUT

      - name: Package Artifacts
        run: |
          python resources/asset-packager.py \
            --version "${{ steps.version.outputs.version }}" \
            --source-dir dist \
            --output-dir release-assets \
            --platforms linux,macos,windows \
            --architectures x64,arm64 \
            --sign

      - name: Upload Release Assets
        uses: actions/upload-artifact@v3
        with:
          name: release-assets
          path: release-assets/*

  build-docker:
    name: Build Docker Images
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract Version
        id: version
        run: |
          if [[ "${{ github.event_name }}" == "push" ]]; then
            VERSION="${{ github.ref_name }}"
            VERSION="${VERSION#v}"
          else
            VERSION="${{ inputs.version }}"
          fi
          echo "version=$VERSION" >> $GITHUB_OUTPUT

      - name: Build and Push
        uses: docker/build-push-action@v4
        with:
          context: .
          platforms: linux/amd64,linux/arm64
          push: ${{ !inputs.dry_run }}
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.version.outputs.version }}
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          build-args: |
            VERSION=${{ steps.version.outputs.version }}
            BUILD_DATE=${{ github.event.head_commit.timestamp }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  publish-npm:
    name: Publish to npm
    needs: build-binaries
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'
          registry-url: 'https://registry.npmjs.org'

      - name: Download All Artifacts
        uses: actions/download-artifact@v3
        with:
          path: dist

      - name: Install Dependencies
        run: npm ci

      - name: Publish to npm
        if: ${{ !inputs.dry_run }}
        run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}

  create-release:
    name: Create GitHub Release
    needs: [package-release, build-docker]
    runs-on: ubuntu-latest
    permissions:
      contents: write

    steps:
      - name: Checkout
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Download Release Assets
        uses: actions/download-artifact@v3
        with:
          name: release-assets
          path: release-assets

      - name: Extract Version
        id: version
        run: |
          if [[ "${{ github.event_name }}" == "push" ]]; then
            VERSION="${{ github.ref_name }}"
          else
            VERSION="v${{ inputs.version }}"
          fi
          echo "version=$VERSION" >> $GITHUB_OUTPUT

      - name: Generate Release Notes
        run: |
          # Get previous tag
          PREV_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")

          # Generate changelog
          python resources/changelog-generator.py \
            --from "$PREV_TAG" \
            --to HEAD \
            --version "${{ steps.version.outputs.version }}" \
            > release-notes.md

      - name: Create Release
        if: ${{ !inputs.dry_run }}
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh release create "${{ steps.version.outputs.version }}" \
            --title "Release ${{ steps.version.outputs.version }}" \
            --notes-file release-notes.md \
            release-assets/*

  post-release:
    name: Post-Release Tasks
    needs: create-release
    runs-on: ubuntu-latest

    steps:
      - name: Notify Team
        run: |
          echo "✅ Multi-platform release complete!"
          echo "📦 Artifacts published to npm, Docker, and GitHub Releases"

      - name: Update Documentation
        run: |
          # Trigger documentation update workflow
          echo "Triggering docs update..."
```

## Step 5: Rollback Workflow

```yaml
# .github/workflows/rollback.yml
name: Rollback Release

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Version to rollback to'
        required: true
      reason:
        description: 'Rollback reason'
        required: true

jobs:
  rollback:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      packages: write

    steps:
      - name: Rollback npm
        run: |
          npm unpublish @org/cli-tool@latest
          npm dist-tag add @org/cli-tool@${{ inputs.version }} latest

      - name: Rollback Docker
        run: |
          docker pull ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ inputs.version }}
          docker tag ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ inputs.version }} \
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          docker push ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest

      - name: Create Rollback Notice
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh issue create \
            --title "🔙 Rollback to ${{ inputs.version }}" \
            --body "**Reason**: ${{ inputs.reason }}" \
            --label "rollback,urgent"
```

## Usage

### 1. Local Build

```bash
# Build all platforms
VERSION=2.0.0 ./scripts/build-binaries.sh

# Package artifacts
python resources/asset-packager.py \
  --version 2.0.0 \
  --platforms linux,macos,windows \
  --sign
```

### 2. Docker Build

```bash
VERSION=2.0.0 \
REGISTRY=ghcr.io \
IMAGE_NAME=org/cli-tool \
./scripts/build-docker.sh
```

### 3. Full Release

```bash
# Create and push tag
git tag v2.0.0
git push origin v2.0.0

# Workflow automatically triggers
```

## Summary

This example demonstrates:
- ✅ Multi-platform binary builds (6 platforms)
- ✅ Docker multi-architecture images
- ✅ npm package publishing
- ✅ GitHub Releases with signed artifacts
- ✅ Automated changelog generation
- ✅ Rollback capabilities
- ✅ Checksum verification
- ✅ Comprehensive CI/CD
