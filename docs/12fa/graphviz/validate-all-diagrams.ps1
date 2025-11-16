# Graphviz Validation and Rendering Script (PowerShell)
# Phase 3: Validation & Integration
# Date: 2025-11-01

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Phase 3: Graphviz Validation & Rendering" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Counters
$script:Total = 0
$script:Success = 0
$script:Failed = 0
$script:Skipped = 0

# Base directory
$BaseDir = "C:\Users\17175\docs\12fa\graphviz"
Set-Location $BaseDir

# Check if graphviz is installed
$GraphvizInstalled = $false
try {
    $dotVersion = & dot -V 2>&1
    Write-Host "✓ Graphviz found: $dotVersion" -ForegroundColor Green
    $GraphvizInstalled = $true
} catch {
    Write-Host "ERROR: Graphviz 'dot' command not found" -ForegroundColor Red
    Write-Host "Please install Graphviz:"
    Write-Host "  Windows: choco install graphviz"
    Write-Host "  Or download from: https://graphviz.org/download/"
    Write-Host ""
    Write-Host "Skipping SVG generation, but will validate syntax..." -ForegroundColor Yellow
}

Write-Host ""

# Validation function
function Validate-AndRender {
    param(
        [string]$Category,
        [string]$Dir
    )

    Write-Host "----------------------------------------" -ForegroundColor Cyan
    Write-Host "Processing: $Category" -ForegroundColor Cyan
    Write-Host "----------------------------------------" -ForegroundColor Cyan

    if (-not (Test-Path $Dir)) {
        Write-Host "⚠ Directory not found: $Dir" -ForegroundColor Yellow
        return
    }

    Set-Location $Dir
    $dotFiles = Get-ChildItem -Filter "*-process.dot"

    if ($dotFiles.Count -eq 0) {
        Write-Host "⚠ No .dot files found in $Dir" -ForegroundColor Yellow
        Set-Location $BaseDir
        return
    }

    Write-Host "Found: $($dotFiles.Count) diagram(s)"

    foreach ($dotFile in $dotFiles) {
        $script:Total++
        $baseName = $dotFile.BaseName

        # Syntax validation
        $validSyntax = $false
        try {
            $null = & dot -Tsvg $dotFile.Name -o "nul" 2>&1
            $validSyntax = $true
            Write-Host "✓ $($dotFile.Name) - Valid syntax" -ForegroundColor Green

            # Render to SVG if Graphviz installed
            if ($GraphvizInstalled) {
                try {
                    & dot -Tsvg $dotFile.Name -o "$baseName.svg" 2>&1 | Out-Null
                    $script:Success++

                    # Also generate PNG
                    try {
                        & dot -Tpng $dotFile.Name -o "$baseName.png" 2>&1 | Out-Null
                    } catch {
                        # PNG generation is optional
                    }
                } catch {
                    Write-Host "✗ $($dotFile.Name) - SVG generation failed" -ForegroundColor Red
                    $script:Failed++
                }
            } else {
                $script:Skipped++
            }
        } catch {
            Write-Host "✗ $($dotFile.Name) - Invalid syntax" -ForegroundColor Red
            $script:Failed++
        }
    }

    Write-Host ""
    Set-Location $BaseDir
}

# Process all categories
Validate-AndRender -Category "Skills" -Dir (Join-Path $BaseDir "skills")
Validate-AndRender -Category "Agents" -Dir (Join-Path $BaseDir "agents")
Validate-AndRender -Category "Commands" -Dir (Join-Path $BaseDir "commands")

# Summary
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Validation Summary" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Total diagrams: $Total"
Write-Host "✓ Valid: $Success" -ForegroundColor Green
Write-Host "✗ Failed: $Failed" -ForegroundColor Red
if (-not $GraphvizInstalled) {
    Write-Host "⚠ Skipped (no Graphviz): $Skipped" -ForegroundColor Yellow
}
Write-Host ""

# Calculate success rate
if ($Total -gt 0) {
    $successRate = [math]::Round(($Success / $Total) * 100, 1)
    Write-Host "Success Rate: $successRate%"
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Output Files" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
if ($GraphvizInstalled) {
    $svgCount = (Get-ChildItem -Recurse -Filter "*.svg").Count
    $pngCount = (Get-ChildItem -Recurse -Filter "*.png").Count
    Write-Host "SVG files: $svgCount"
    Write-Host "PNG files: $pngCount"
} else {
    Write-Host "Graphviz not installed - no renders generated"
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
