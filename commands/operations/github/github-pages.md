# GitHub Pages Deployment Command

**Category**: External Integrations
**Phase**: 4 - Deployment & Integration
**Complexity**: Low

## Purpose

Automated deployment of static sites to GitHub Pages with build optimization, custom domains, and SSL configuration.

## Usage

```bash
/github-pages [action] [options]

Actions:
  deploy              Deploy to GitHub Pages
  configure           Configure Pages settings
  domain              Setup custom domain
  build               Build static site
  preview             Local preview

Options:
  --branch <name>     Deploy branch (default: gh-pages)
  --dir <path>        Build directory (default: build/)
  --framework <name>  Framework (react|vue|next|gatsby|hugo|jekyll)
  --domain <domain>   Custom domain name
  --https             Enable HTTPS (default: true)
  --spa               Single Page Application mode
```

## Examples

```bash
# Deploy React app
/github-pages deploy --framework react --dir build --https

# Deploy with custom domain
/github-pages deploy --domain www.mysite.com --https

# Deploy Next.js static export
/github-pages deploy --framework next --dir out --spa

# Configure Pages settings
/github-pages configure --branch gh-pages --https

# Setup custom domain
/github-pages domain --domain blog.example.com

# Local preview before deployment
/github-pages preview --framework react --port 8080
```

## Workflow

1. **Build Static Site**
   - Run framework-specific build command
   - Optimize assets (minify, compress)
   - Generate sitemap.xml and robots.txt
   - Process images (WebP conversion)

2. **Prepare Deployment**
   - Create/checkout gh-pages branch
   - Clean previous build artifacts
   - Copy build files to branch root
   - Add .nojekyll file (if needed)

3. **Deploy**
   - Commit build files
   - Push to gh-pages branch
   - Wait for GitHub Pages build
   - Verify deployment success

4. **Post-Deployment**
   - Configure custom domain DNS
   - Enable HTTPS via Let's Encrypt
   - Update README with live URL
   - Generate deployment report

## Framework-Specific Configs

### React (Create React App)

```json
// package.json
{
  "homepage": "https://username.github.io/repo-name",
  "scripts": {
    "predeploy": "npm run build",
    "deploy": "gh-pages -d build"
  }
}
```

```bash
npm install --save-dev gh-pages
npm run deploy
```

### Next.js Static Export

```javascript
// next.config.js
module.exports = {
  output: 'export',
  basePath: '/repo-name',
  images: {
    unoptimized: true
  }
}
```

```bash
npm run build
/github-pages deploy --framework next --dir out
```

### Vue.js

```javascript
// vue.config.js
module.exports = {
  publicPath: process.env.NODE_ENV === 'production'
    ? '/repo-name/'
    : '/'
}
```

### Hugo

```toml
# config.toml
baseURL = "https://username.github.io/repo-name/"
publishDir = "public"
```

```bash
hugo
/github-pages deploy --framework hugo --dir public
```

## GitHub Actions Workflow

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build-deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - uses: actions/setup-node@v4
      with:
        node-version: '20'
        cache: 'npm'

    - run: npm ci
    - run: npm run build

    - name: Setup Pages
      uses: actions/configure-pages@v4

    - name: Upload artifact
      uses: actions/upload-pages-artifact@v3
      with:
        path: ./build

    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v4
```

## Custom Domain Configuration

### 1. Add CNAME File

```bash
echo "www.mysite.com" > build/CNAME
```

### 2. Configure DNS Records

```dns
# A Records (GitHub Pages IPs)
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153

# CNAME Record
www.mysite.com -> username.github.io
```

### 3. Enable HTTPS

```bash
# GitHub automatically provisions SSL certificate
# Wait 24 hours for DNS propagation
# Verify HTTPS at: https://www.mysite.com
```

## SPA Configuration

For single-page applications with client-side routing:

```html
<!-- 404.html redirect trick -->
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script>
    sessionStorage.redirect = location.href;
  </script>
  <meta http-equiv="refresh" content="0;URL='/index.html'">
</head>
</html>
```

```javascript
// index.html - restore path
<script>
  (function(){
    var redirect = sessionStorage.redirect;
    delete sessionStorage.redirect;
    if (redirect && redirect != location.href) {
      history.replaceState(null, null, redirect);
    }
  })();
</script>
```

## Optimization

### Asset Optimization

```bash
# Image compression
npm install --save-dev imagemin imagemin-webp

# CSS minification
npm install --save-dev cssnano postcss-cli

# JS minification (built into most frameworks)
# Gzip/Brotli compression (automatic via GitHub Pages)
```

### Performance Checklist

- ✅ Minify HTML/CSS/JS
- ✅ Compress images (WebP, AVIF)
- ✅ Enable browser caching
- ✅ Use CDN for assets
- ✅ Lazy load images
- ✅ Code splitting
- ✅ Tree shaking
- ✅ Remove unused CSS

## Command Output

```json
{
  "deployment_id": "pages-abc123",
  "site_url": "https://username.github.io/repo-name",
  "custom_domain": "www.mysite.com",
  "https_enabled": true,
  "build_time": "1m 12s",
  "deployed_files": 47,
  "total_size": "2.3 MB",
  "status": "SUCCESS"
}
```

## Troubleshooting

**404 Errors**:
- Verify `homepage` in package.json
- Check `basePath` in Next.js config
- Ensure .nojekyll file exists (non-Jekyll sites)

**Build Failures**:
- Check Node.js version compatibility
- Clear npm cache: `npm cache clean --force`
- Verify environment variables

**HTTPS Issues**:
- Wait 24 hours for DNS propagation
- Verify DNS records with `dig` or `nslookup`
- Check CNAME file in repository root

## Best Practices

1. Use GitHub Actions for automated deployments
2. Enable branch protection for main branch
3. Test builds locally before deploying
4. Use semantic versioning for releases
5. Add deployment status badge to README
6. Monitor analytics (Google Analytics, Plausible)
7. Implement sitemap and robots.txt
8. Use service worker for offline support

## Integration Points

- **/github-actions** - Automated CI/CD
- **/docker-compose** - Local development
- **/slack-notify** - Deployment notifications
- **/hook:on-push** - Auto-deploy on push

## Related Commands

- `/github-actions` - CI/CD workflows
- `/aws-deploy` - Alternative hosting
- `/docker-compose` - Local development

---

**Implementation Status**: Production-Ready
**Last Updated**: 2025-11-01
**Maintainer**: Integration Specialist
