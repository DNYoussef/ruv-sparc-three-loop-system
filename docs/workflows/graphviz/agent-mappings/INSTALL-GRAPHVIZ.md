# GraphViz Installation Guide

## Overview

GraphViz is required to render the `.dot` diagram files into viewable formats (PNG, SVG, PDF).

## Installation Instructions

### Windows

**Option 1: Using Chocolatey (Recommended)**
```bash
# Install Chocolatey if not already installed
# Run in PowerShell as Administrator:
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install GraphViz
choco install graphviz -y

# Add to PATH (if not automatic)
# Add: C:\Program Files\Graphviz\bin to System PATH
```

**Option 2: Manual Download**
1. Download from: https://graphviz.org/download/
2. Run the installer (graphviz-X.X.X-win64.exe)
3. Add to PATH:
   - Right-click "This PC" → Properties
   - Advanced system settings → Environment Variables
   - Edit "Path" under System variables
   - Add: `C:\Program Files\Graphviz\bin`
4. Restart terminal/IDE

**Option 3: Using Scoop**
```bash
scoop install graphviz
```

### macOS

**Using Homebrew (Recommended)**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install GraphViz
brew install graphviz
```

**Using MacPorts**
```bash
sudo port install graphviz
```

### Linux

**Ubuntu/Debian**
```bash
sudo apt-get update
sudo apt-get install graphviz -y
```

**Fedora/RHEL/CentOS**
```bash
sudo dnf install graphviz -y
# or
sudo yum install graphviz -y
```

**Arch Linux**
```bash
sudo pacman -S graphviz
```

**openSUSE**
```bash
sudo zypper install graphviz
```

### Python (via pip)

If you prefer Python integration:
```bash
pip install graphviz

# Note: This installs the Python wrapper, but you still need
# the GraphViz binaries installed via one of the methods above
```

## Verify Installation

After installation, verify GraphViz is working:

```bash
# Check version
dot -V

# Expected output:
# dot - graphviz version X.X.X (date)

# Test rendering
cd docs/workflows/graphviz/agent-mappings/
dot -Tpng core-agent-hub.dot -o test.png

# If successful, test.png will be created
```

## Troubleshooting

### "dot: command not found"

**Cause**: GraphViz not in system PATH

**Solutions**:

**Windows**:
```powershell
# Check if installed
where dot

# Manually add to PATH for current session
$env:Path += ";C:\Program Files\Graphviz\bin"

# Permanent: Edit System Environment Variables
```

**macOS/Linux**:
```bash
# Check if installed
which dot

# Check if in PATH
echo $PATH | grep graphviz

# Add to PATH temporarily
export PATH=$PATH:/usr/local/bin

# Add to PATH permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export PATH=$PATH:/usr/local/bin' >> ~/.bashrc
source ~/.bashrc
```

### "Error: syntax error in line X"

**Cause**: Invalid DOT syntax in diagram file

**Solution**:
```bash
# Validate DOT file syntax
dot -Tcanon diagram.dot > /dev/null

# This will show syntax errors if any exist
```

### "Warning: Unable to load plugin"

**Cause**: Missing GraphViz plugins

**Solution**:

**Windows**:
```bash
# Rebuild plugin configuration
dot -c
```

**Linux**:
```bash
sudo apt-get install --reinstall graphviz libgraphviz-dev
dot -c
```

### Memory/Performance Issues with Large Diagrams

**Cause**: Complex diagrams with many nodes/edges

**Solutions**:
```bash
# Increase memory limit
dot -Gmaxiter=10000 -Tpng diagram.dot -o output.png

# Use different layout engine (faster for large graphs)
sfdp -Tpng diagram.dot -o output.png

# Simplify output
dot -Tpng -Gmargin=0 -Gpad=0 diagram.dot -o output.png

# Use lower DPI
dot -Tpng -Gdpi=72 diagram.dot -o output.png
```

## Quick Rendering Script

Create a batch rendering script:

**Windows (PowerShell)**:
```powershell
# Save as: render-all.ps1
$diagrams = Get-ChildItem -Filter *.dot
foreach ($diagram in $diagrams) {
    $base = $diagram.BaseName
    Write-Host "Rendering: $base"

    dot -Tpng "$diagram" -o "$base.png"
    dot -Tsvg "$diagram" -o "$base.svg"
    dot -Tpdf "$diagram" -o "$base.pdf"
}
Write-Host "Done!"
```

**macOS/Linux (Bash)**:
```bash
#!/bin/bash
# Save as: render-all.sh
# chmod +x render-all.sh

for diagram in *.dot; do
    base="${diagram%.dot}"
    echo "Rendering: $base"

    dot -Tpng "$diagram" -o "$base.png"
    dot -Tsvg "$diagram" -o "$base.svg"
    dot -Tpdf "$diagram" -o "$base.pdf"
done
echo "Done!"
```

## Alternative: Online Rendering

If you can't install GraphViz locally, use online tools:

1. **GraphvizOnline**: https://dreampuf.github.io/GraphvizOnline/
   - Paste `.dot` file contents
   - Renders in browser
   - Export as PNG/SVG

2. **Edotor**: https://edotor.net/
   - Online GraphViz editor
   - Live preview
   - Multiple export formats

3. **Graphviz Visual Editor**: https://graphviz.org/resources/

## VS Code Integration

For VS Code users:

1. Install extension: "Graphviz (dot) language support for Visual Studio Code"
   - Extension ID: `joaompinto.vscode-graphviz`

2. Preview diagrams directly in VS Code:
   - Open `.dot` file
   - Right-click → "Open Preview to the Side"
   - Live rendering as you edit

3. Install extension: "Graphviz Preview"
   - Extension ID: `EFanZh.graphviz-preview`
   - Provides better rendering options

## Docker Alternative

Run GraphViz via Docker (no local installation):

```bash
# Pull GraphViz Docker image
docker pull omerosmanilter/graphviz

# Render diagram
docker run --rm -v "$(pwd):/data" omerosmanilter/graphviz \
  dot -Tpng /data/diagram.dot -o /data/output.png

# Create alias for convenience
alias dot='docker run --rm -v "$(pwd):/data" omerosmanilter/graphviz dot'

# Now use normally
dot -Tpng diagram.dot -o output.png
```

## Support

- **Official Documentation**: https://graphviz.org/documentation/
- **DOT Language Guide**: https://graphviz.org/doc/info/lang.html
- **Attribute Reference**: https://graphviz.org/doc/info/attrs.html
- **Gallery**: https://graphviz.org/gallery/

---

**Last Updated**: 2025-11-01
