Param(
    [string]$OutputDirectory = "rendered"
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $Root $OutputDirectory

if (-not (Get-Command dot -ErrorAction SilentlyContinue)) {
    Write-Error "Graphviz 'dot' binary is required but was not found in PATH."
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Render-Directory {
    param(
        [string]$Source,
        [string]$Relative
    )

    if (-not (Test-Path $Source)) {
        return
    }

    Get-ChildItem -Path $Source -Filter *.dot -File | ForEach-Object {
        $base = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
        $targetDir = Join-Path $OutDir $Relative
        New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
        $target = Join-Path $targetDir "$base.svg"
        dot -Tsvg $_.FullName -o $target | Out-Null
        Write-Output "Rendered $Relative/$base.svg"
    }
}

Render-Directory (Join-Path $Root 'skills') 'skills'
Render-Directory (Join-Path $Root 'agent-mappings') 'agent-mappings'

Write-Output ""
Write-Output "============================"
Write-Output "Graphviz Validation Complete"
Write-Output "============================"
Write-Output "Source: $Root"
Write-Output "Output: $OutDir"
