# MCP Profile Manager for ruv-SPARC Three-Loop System
# Version: 1.0.0
# Purpose: Generate and switch between MCP configuration profiles to optimize token usage

$ErrorActionPreference = "Stop"

$script:ProfilesDir = "$env:APPDATA\Claude\profiles"
$script:ConfigPath = "$env:APPDATA\Claude\claude_desktop_config.json"
$script:BackupPath = "$env:APPDATA\Claude\claude_desktop_config.json.backup"

# Profile Definitions
$script:Profiles = @{
    "minimal" = @{
        Description = "Minimal global config (11.3k tokens - 89.7% reduction)"
        TokenCost = 11300
        UseCases = @("70% of general development", "Simple features", "Bug fixes", "Planning")
        MCPs = @{
            "fetch" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-fetch")
            }
            "sequential-thinking" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-sequential-thinking")
            }
            "filesystem" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-filesystem", "C:\Users\17175")
            }
        }
    }

    "quality" = @{
        Description = "Code quality workflows (13.1k tokens)"
        TokenCost = 13100
        UseCases = @("Code reviews", "Audits", "Debugging", "Quality checks")
        MCPs = @{
            "fetch" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-fetch")
            }
            "sequential-thinking" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-sequential-thinking")
            }
            "filesystem" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-filesystem", "C:\Users\17175")
            }
            "focused-changes" = @{
                command = "node"
                args = @("C:/Users/17175/Documents/Cline/MCP/focused-changes-server/build/index.js")
            }
        }
    }

    "swarm" = @{
        Description = "Multi-agent swarm coordination (26.8k tokens)"
        TokenCost = 26800
        UseCases = @("Three-Loop System", "Parallel implementation", "Complex coordination")
        MCPs = @{
            "fetch" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-fetch")
            }
            "sequential-thinking" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-sequential-thinking")
            }
            "filesystem" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-filesystem", "C:\Users\17175")
            }
            "ruv-swarm" = @{
                command = "npx"
                args = @("ruv-swarm", "mcp", "start")
            }
        }
    }

    "ml" = @{
        Description = "Machine learning development (24.1k tokens)"
        TokenCost = 24100
        UseCases = @("Neural training", "ML pipelines", "Distributed ML")
        MCPs = @{
            "fetch" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-fetch")
            }
            "sequential-thinking" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-sequential-thinking")
            }
            "filesystem" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-filesystem", "C:\Users\17175")
            }
            "flow-nexus" = @{
                command = "npx"
                args = @("flow-nexus@latest", "mcp", "start")
            }
        }
    }

    "frontend" = @{
        Description = "Frontend testing and UI work (32.8k tokens)"
        TokenCost = 32800
        UseCases = @("UI testing", "Visual regression", "E2E tests", "React/Vue development")
        MCPs = @{
            "fetch" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-fetch")
            }
            "sequential-thinking" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-sequential-thinking")
            }
            "filesystem" = @{
                command = "npx"
                args = @("@modelcontextprotocol/server-filesystem", "C:\Users\17175")
            }
            "playwright" = @{
                command = "npx"
                args = @("playwright-mcp")
            }
            "flow-nexus" = @{
                command = "npx"
                args = @("flow-nexus@latest", "mcp", "start")
            }
        }
    }
}

function Initialize-ProfilesDirectory {
    if (!(Test-Path $script:ProfilesDir)) {
        New-Item -ItemType Directory -Path $script:ProfilesDir -Force | Out-Null
        Write-Host "Created profiles directory: $script:ProfilesDir" -ForegroundColor Green
    }
}

function Show-AvailableProfiles {
    Write-Host "`nAvailable MCP Profiles:" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Gray

    foreach ($profileName in $script:Profiles.Keys | Sort-Object) {
        $profile = $script:Profiles[$profileName]
        Write-Host "`n[$profileName]" -ForegroundColor Yellow
        Write-Host "  Description: $($profile.Description)" -ForegroundColor White
        Write-Host "  Token Cost:  $($profile.TokenCost) tokens" -ForegroundColor Magenta
        Write-Host "  MCPs:        $($profile.MCPs.Count) servers" -ForegroundColor Gray
        Write-Host "  Use Cases:" -ForegroundColor White
        foreach ($useCase in $profile.UseCases) {
            Write-Host "    - $useCase" -ForegroundColor Gray
        }
    }

    Write-Host "`n" ("=" * 80) -ForegroundColor Gray
}

function Get-CurrentProfile {
    if (!(Test-Path $script:ConfigPath)) {
        return "None (config not found)"
    }

    $currentConfig = Get-Content $script:ConfigPath | ConvertFrom-Json
    $currentMCPs = $currentConfig.mcpServers.PSObject.Properties.Name | Sort-Object

    foreach ($profileName in $script:Profiles.Keys) {
        $profileMCPs = $script:Profiles[$profileName].MCPs.Keys | Sort-Object
        if (($currentMCPs -join ',') -eq ($profileMCPs -join ',')) {
            return $profileName
        }
    }

    return "Custom ($($currentMCPs.Count) MCPs)"
}

function New-ProfileConfig {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProfileName
    )

    if (!$script:Profiles.ContainsKey($ProfileName)) {
        throw "Profile '$ProfileName' not found. Use Show-AvailableProfiles to list profiles."
    }

    $profile = $script:Profiles[$ProfileName]
    $config = @{
        mcpServers = $profile.MCPs
    }

    return $config
}

function Export-Profile {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProfileName,

        [Parameter(Mandatory=$false)]
        [string]$OutputPath
    )

    Initialize-ProfilesDirectory

    if (!$OutputPath) {
        $OutputPath = Join-Path $script:ProfilesDir "$ProfileName.json"
    }

    $config = New-ProfileConfig -ProfileName $ProfileName
    $json = $config | ConvertTo-Json -Depth 10

    $json | Out-File -FilePath $OutputPath -Encoding UTF8 -Force

    Write-Host "Exported profile '$ProfileName' to: $OutputPath" -ForegroundColor Green

    $profile = $script:Profiles[$ProfileName]
    Write-Host "Token Cost: $($profile.TokenCost) tokens" -ForegroundColor Magenta
}

function Export-AllProfiles {
    Initialize-ProfilesDirectory

    foreach ($profileName in $script:Profiles.Keys) {
        Export-Profile -ProfileName $profileName
    }

    Write-Host "`nAll profiles exported to: $script:ProfilesDir" -ForegroundColor Green
}

function Switch-Profile {
    param(
        [Parameter(Mandatory=$true)]
        [string]$ProfileName,

        [Parameter(Mandatory=$false)]
        [switch]$NoBackup
    )

    if (!$script:Profiles.ContainsKey($ProfileName)) {
        throw "Profile '$ProfileName' not found. Available: $($script:Profiles.Keys -join ', ')"
    }

    # Backup current config
    if (!$NoBackup -and (Test-Path $script:ConfigPath)) {
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $backupPath = "$script:ConfigPath.backup-$timestamp"
        Copy-Item $script:ConfigPath $backupPath -Force
        Write-Host "Backed up current config to: $backupPath" -ForegroundColor Yellow
    }

    # Generate and save new config
    $profilePath = Join-Path $script:ProfilesDir "$ProfileName.json"

    if (!(Test-Path $profilePath)) {
        Write-Host "Profile file not found, generating..." -ForegroundColor Yellow
        Export-Profile -ProfileName $ProfileName
    }

    Copy-Item $profilePath $script:ConfigPath -Force

    $profile = $script:Profiles[$ProfileName]
    Write-Host "`nSwitched to profile: $ProfileName" -ForegroundColor Green
    Write-Host "Description: $($profile.Description)" -ForegroundColor White
    Write-Host "Token Cost: $($profile.TokenCost) tokens" -ForegroundColor Magenta
    Write-Host "MCPs Loaded: $($profile.MCPs.Count)" -ForegroundColor Gray

    Write-Host "`nIMPORTANT: Restart Claude Code for changes to take effect!" -ForegroundColor Red -BackgroundColor Yellow
}

function Show-TokenComparison {
    $currentProfile = Get-CurrentProfile

    Write-Host "`nToken Usage Comparison:" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Gray

    $maxTokens = 200000
    $currentUsage = 109800 # Full config baseline

    foreach ($profileName in $script:Profiles.Keys | Sort-Object { $script:Profiles[$_].TokenCost }) {
        $profile = $script:Profiles[$profileName]
        $tokens = $profile.TokenCost
        $percentage = [math]::Round(($tokens / $maxTokens) * 100, 1)
        $savings = $currentUsage - $tokens
        $savingsPercentage = [math]::Round(($savings / $currentUsage) * 100, 1)

        $isCurrent = $profileName -eq $currentProfile

        $marker = if ($isCurrent) { ">" } else { " " }
        $color = if ($isCurrent) { "Green" } else { "White" }

        Write-Host "$marker [$profileName]" -ForegroundColor $color -NoNewline
        Write-Host " " -NoNewline
        Write-Host "$tokens tokens" -ForegroundColor Magenta -NoNewline
        Write-Host " ($percentage% of context)" -ForegroundColor Gray -NoNewline
        Write-Host " | Saves: $savings tokens ($savingsPercentage%)" -ForegroundColor Yellow
    }

    Write-Host "`n" ("=" * 80) -ForegroundColor Gray
    Write-Host "Current Profile: $currentProfile" -ForegroundColor Cyan
}

# Export module functions
Export-ModuleMember -Function @(
    'Show-AvailableProfiles',
    'Get-CurrentProfile',
    'Export-Profile',
    'Export-AllProfiles',
    'Switch-Profile',
    'Show-TokenComparison'
)

# Main execution if run as script
if ($MyInvocation.InvocationName -ne '.') {
    Write-Host @"

    __  __ ____ ____    ____             __ _ _         __  __
   |  \/  / ___|  _ \  |  _ \ _ __ ___  / _(_) | ___   |  \/  | __ _ _ __   __ _  __ _  ___ _ __
   | |\/| | |   | |_) | | |_) | '__/ _ \| |_| | |/ _ \  | |\/| |/ _\` | '_ \ / _\` |/ _\` |/ _ \ '__|
   | |  | | |___|  __/  |  __/| | | (_) |  _| | |  __/  | |  | | (_| | | | | (_| | (_| |  __/ |
   |_|  |_|\____|_|     |_|   |_|  \___/|_| |_|_|\___|  |_|  |_|\__,_|_| |_|\__,_|\__, |\___|_|
                                                                                   |___/
    ruv-SPARC Three-Loop System
    Token Optimization: 89.7% reduction possible

"@ -ForegroundColor Cyan

    param(
        [Parameter(Mandatory=$false)]
        [ValidateSet('list', 'current', 'switch', 'export', 'export-all', 'compare')]
        [string]$Action = 'list',

        [Parameter(Mandatory=$false)]
        [string]$Profile
    )

    switch ($Action) {
        'list' {
            Show-AvailableProfiles
        }
        'current' {
            $current = Get-CurrentProfile
            Write-Host "Current Profile: $current" -ForegroundColor Cyan
        }
        'switch' {
            if (!$Profile) {
                Write-Host "ERROR: -Profile parameter required for 'switch' action" -ForegroundColor Red
                Write-Host "Example: .\MCP-Profile-Manager.ps1 -Action switch -Profile minimal" -ForegroundColor Yellow
                exit 1
            }
            Switch-Profile -ProfileName $Profile
        }
        'export' {
            if (!$Profile) {
                Write-Host "ERROR: -Profile parameter required for 'export' action" -ForegroundColor Red
                exit 1
            }
            Export-Profile -ProfileName $Profile
        }
        'export-all' {
            Export-AllProfiles
        }
        'compare' {
            Show-TokenComparison
        }
    }
}
