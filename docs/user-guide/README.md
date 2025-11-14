# rUv SPARC UI Dashboard Documentation

This directory contains comprehensive user documentation for the rUv SPARC UI Dashboard.

## 📚 Documentation Files

### Core Documentation

1. **[index.md](index.md)** - Documentation homepage
2. **[INSTALL.md](INSTALL.md)** - Installation guide with prerequisites and setup
3. **[USER_GUIDE.md](USER_GUIDE.md)** - Complete user manual with UI walkthroughs
4. **[API_GUIDE.md](API_GUIDE.md)** - REST API reference with code examples
5. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
6. **[FAQ.md](FAQ.md)** - Frequently asked questions
7. **[VIDEO_TUTORIALS.md](VIDEO_TUTORIALS.md)** - Video tutorial scripts and recording guide

### Configuration Files

- **[_config.yml](_config.yml)** - GitHub Pages configuration
- **[.nojekyll](.nojekyll)** - Disable Jekyll processing for certain files

## 🌐 Viewing Documentation

### Local Preview (Recommended)

**Using Jekyll** (GitHub Pages compatible):

```bash
# Install Jekyll
gem install jekyll bundler

# Navigate to docs directory
cd docs/user-guide

# Serve locally
jekyll serve

# Open browser to http://localhost:4000
```

**Using Python HTTP Server** (simple preview):

```bash
# Navigate to docs directory
cd docs/user-guide

# Start server
python -m http.server 8000

# Open browser to http://localhost:8000
```

### GitHub Pages Deployment

**Setup**:

1. Push documentation to GitHub repository
2. Go to repository Settings → Pages
3. Source: Deploy from branch
4. Branch: `main` (or your default branch)
5. Folder: `/docs/user-guide`
6. Click "Save"

**Access**:
- Your documentation will be available at `https://yourusername.github.io/ruv-sparc-ui-dashboard/`

**Custom Domain** (optional):
1. Add `CNAME` file with your domain
2. Configure DNS with GitHub Pages IP
3. Enable HTTPS in settings

### Markdown Editors

**Desktop**:
- **VS Code** with Markdown Preview Enhanced extension
- **Typora** - WYSIWYG Markdown editor
- **MarkText** - Free, open-source Markdown editor

**Online**:
- **HackMD** - Collaborative Markdown editor
- **StackEdit** - In-browser Markdown editor

## 📝 Contributing to Documentation

### Guidelines

1. **Clarity**: Write for beginners, explain jargon
2. **Examples**: Include code examples and screenshots
3. **Structure**: Use headings, lists, and tables
4. **Accuracy**: Test all instructions before documenting
5. **Updates**: Keep documentation in sync with code changes

### Documentation Style

**Headings**:
```markdown
# H1: Page Title (one per page)
## H2: Major Sections
### H3: Subsections
#### H4: Details (use sparingly)
```

**Code Blocks**:
````markdown
```bash
# Shell commands
npm install
```

```javascript
// JavaScript code
const example = 'code';
```
````

**Admonitions**:
```markdown
**⚠️ Warning**: Important cautionary information

**💡 Tip**: Helpful suggestions

**✅ Success**: Positive confirmation
```

**Links**:
```markdown
[Link Text](URL)
[Internal Link](INSTALL.md)
[Anchor Link](#section-name)
```

### Adding Screenshots

**Location**: `docs/user-guide/images/`

**Naming Convention**:
- `dashboard-overview.png` - Descriptive, lowercase, hyphenated
- `calendar-month-view.png`
- `project-kanban-board.png`

**Markdown**:
```markdown
![Alt Text](images/screenshot-name.png)

**Screenshot**: `images/screenshot-name.png`
```

**Best Practices**:
- Use PNG for UI screenshots (lossless)
- Optimize images (use TinyPNG or similar)
- Include descriptive alt text for accessibility
- Highlight important UI elements (arrows, boxes)

### Adding Video Tutorials

**Scripts**: See [VIDEO_TUTORIALS.md](VIDEO_TUTORIALS.md)

**Embedding YouTube**:
```markdown
### Video Title (Duration)
[![Thumbnail Alt](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

Description of video content.
```

**Hosting Options**:
- **YouTube**: Free, unlimited, good SEO
- **Vimeo**: Ad-free, professional
- **Self-hosted**: Not recommended (bandwidth costs)

## 🔄 Updating Documentation

### When Code Changes

**Process**:
1. Make code changes
2. Update relevant documentation files
3. Add screenshots if UI changed
4. Re-record videos if major changes
5. Update changelog/version numbers
6. Commit documentation with code

**Checklist**:
- [ ] Update affected guides
- [ ] Update API reference (if API changed)
- [ ] Update screenshots
- [ ] Test all code examples
- [ ] Update FAQ (if new common questions)
- [ ] Update troubleshooting (if new issues)

### Version Control

**Versioning**:
- Documentation version matches software version
- Use semantic versioning (1.0.0, 1.1.0, 2.0.0)
- Include version number in footer

**Changelog**:
```markdown
## Changelog

### Version 1.1.0 (2025-02-01)
- Added GitLab integration documentation
- Updated API guide with new endpoints
- New troubleshooting section for common errors

### Version 1.0.0 (2025-01-08)
- Initial release
```

## 🧪 Testing Documentation

### Before Publishing

**Checklist**:
- [ ] All links work (no 404s)
- [ ] Code examples run without errors
- [ ] Screenshots are up-to-date
- [ ] No typos or grammar errors
- [ ] Renders correctly in Markdown preview
- [ ] Renders correctly on GitHub Pages

**Tools**:

**Link Checker**:
```bash
# Install markdown-link-check
npm install -g markdown-link-check

# Check all links
markdown-link-check docs/user-guide/*.md
```

**Spell Checker**:
```bash
# Install cspell
npm install -g cspell

# Check spelling
cspell docs/user-guide/*.md
```

**Markdown Linter**:
```bash
# Install markdownlint-cli
npm install -g markdownlint-cli

# Lint Markdown
markdownlint docs/user-guide/*.md
```

## 📊 Analytics & Feedback

### Tracking Improvements

**Metrics**:
- Page views (most popular docs)
- Search queries (what users look for)
- Support ticket categories (common issues)
- Community questions (Discord, GitHub)

**Feedback Collection**:
- Add "Was this helpful?" buttons
- Monitor GitHub Issues for doc-related questions
- Ask for feedback in Discord #documentation channel
- Annual documentation survey

### Continuous Improvement

**Monthly Review**:
- Check analytics for popular/unpopular pages
- Review recent support tickets
- Identify documentation gaps
- Update based on user feedback

**Quarterly Updates**:
- Major content refresh
- New features documented
- Deprecated content removed
- Screenshots updated

## 🎨 Styling & Themes

### GitHub Pages Theme

**Current**: `jekyll-theme-cayman`

**Customization**:
Create `assets/css/style.scss`:

```scss
---
---

@import "{{ site.theme }}";

/* Custom styles */
.highlight {
  background: #f5f5f5;
  border-left: 4px solid #007bff;
  padding: 1rem;
}
```

### Dark Mode Support

**Media Query**:
```css
@media (prefers-color-scheme: dark) {
  body {
    background: #1a1a1a;
    color: #f0f0f0;
  }

  .card {
    background: #2d2d2d;
    border-color: #444;
  }
}
```

## 📞 Documentation Support

**Issues**:
- Report documentation bugs: GitHub Issues with label `documentation`
- Suggest improvements: GitHub Discussions

**Questions**:
- Ask in Discord #documentation channel
- Email: docs@ruv-sparc.io

**Contributing**:
- Submit PRs for documentation fixes
- Help translate documentation (future)

---

## 📂 Directory Structure

```
docs/user-guide/
├── index.md                  # Homepage
├── INSTALL.md                # Installation guide
├── USER_GUIDE.md             # User manual
├── API_GUIDE.md              # API reference
├── TROUBLESHOOTING.md        # Troubleshooting guide
├── FAQ.md                    # FAQ
├── VIDEO_TUTORIALS.md        # Video tutorial scripts
├── README.md                 # This file
├── _config.yml               # GitHub Pages config
├── .nojekyll                 # Jekyll bypass
├── images/                   # Screenshots
│   ├── dashboard-overview.png
│   ├── calendar-views.png
│   └── ...
└── videos/                   # Video tutorials (future)
    ├── 01-getting-started.mp4
    └── ...
```

---

**Documentation Maintainers**: @yourusername
**Last Updated**: January 8, 2025
**Version**: 1.0.0
