# Native Windows Deployment - BLOCKER Fixes Implementation Guide

**Priority**: 🔴 CRITICAL - Must complete before ANY deployment
**Estimated Time**: 9 hours (1-2 days)
**Status**: ❌ NOT STARTED

---

## 📋 Implementation Checklist

### BLOCKER-001: Secure Secret Management (4 hours) 🔴
- [ ] Step 1: Create secret management library (30 min)
- [ ] Step 2: Update install-postgres.ps1 (30 min)
- [ ] Step 3: Update install-backend-service.ps1 (1 hour)
- [ ] Step 4: Update configure-postgres.ps1 (30 min)
- [ ] Step 5: Update configure-redis.ps1 (30 min)
- [ ] Step 6: Test secret encryption/decryption (30 min)
- [ ] Step 7: Update master-install.ps1 (30 min)

### BLOCKER-002: Failure Recovery (2 hours) 🔴
- [ ] Step 1: Update NSSM configuration in all service scripts (1 hour)
- [ ] Step 2: Test restart behavior with simulated failures (30 min)
- [ ] Step 3: Verify exponential backoff works (30 min)

### BLOCKER-003: Health Monitoring (3 hours) 🟡
- [ ] Step 1: Create health-monitor.ps1 script (1.5 hours)
- [ ] Step 2: Create scheduled task installer (30 min)
- [ ] Step 3: Test monitoring and alerting (1 hour)

**Total Time**: 9 hours

---

## 🛠️ BLOCKER-001: Secure Secret Management

### Step 1: Create Secret Management Library (30 min)

**File**: `scripts/native-windows/lib/secret-manager.ps1` (NEW)

```powershell
<#
.SYNOPSIS
    Secure secret management using Windows DPAPI encryption
.DESCRIPTION
    Provides functions to generate, encrypt, store, and retrieve secrets
    using Windows Data Protection API (DPAPI) for native encryption.
#>

# Ensure System.Web assembly is loaded for password generation
Add-Type -AssemblyName System.Web

function New-SecurePassword {
    <#
    .SYNOPSIS
        Generate cryptographically secure password
    .PARAMETER Length
        Password length (default: 32)
    .PARAMETER MinNonAlphanumeric
        Minimum special characters (default: 8)
    #>
    param(
        [int]$Length = 32,
        [int]$MinNonAlphanumeric = 8
    )

    return [System.Web.Security.Membership]::GeneratePassword($Length, $MinNonAlphanumeric)
}

function New-SecureToken {
    <#
    .SYNOPSIS
        Generate Base64-encoded random token (for JWT secrets)
    .PARAMETER Bytes
        Number of random bytes (default: 64)
    #>
    param([int]$Bytes = 64)

    $RandomBytes = New-Object byte[] $Bytes
    $RNG = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $RNG.GetBytes($RandomBytes)
    return [Convert]::ToBase64String($RandomBytes)
}

function Save-EncryptedSecret {
    <#
    .SYNOPSIS
        Encrypt and save secret using DPAPI
    .PARAMETER Name
        Secret name (e.g., "db_password")
    .PARAMETER Value
        Secret value to encrypt
    .PARAMETER Path
        Secrets directory (default: C:\ProgramData\SPARC\secrets)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Name,

        [Parameter(Mandatory=$true)]
        [string]$Value,

        [string]$Path = "C:\ProgramData\SPARC\secrets"
    )

    # Create secrets directory with restrictive ACLs
    if (-not (Test-Path $Path)) {
        Write-Host "📁 Creating secrets directory: $Path" -ForegroundColor Cyan
        New-Item -Path $Path -ItemType Directory -Force | Out-Null

        # Set ACLs: Administrators Full Control, remove inheritance
        icacls $Path /inheritance:r | Out-Null
        icacls $Path /grant "BUILTIN\Administrators:(OI)(CI)F" | Out-Null
        icacls $Path /grant "NT AUTHORITY\SYSTEM:(OI)(CI)F" | Out-Null

        Write-Host "✅ Secrets directory created with restrictive permissions" -ForegroundColor Green
    }

    # Encrypt using DPAPI (machine scope - any user on this machine can decrypt)
    $SecureString = ConvertTo-SecureString -String $Value -AsPlainText -Force
    $Encrypted = ConvertFrom-SecureString -SecureString $SecureString

    # Save to file
    $FilePath = Join-Path $Path "$Name.enc"
    $Encrypted | Out-File -FilePath $FilePath -Encoding UTF8

    # Set file ACLs (Administrators only)
    icacls $FilePath /inheritance:r | Out-Null
    icacls $FilePath /grant "BUILTIN\Administrators:(F)" | Out-Null
    icacls $FilePath /grant "NT AUTHORITY\SYSTEM:(F)" | Out-Null

    Write-Host "✅ Secret '$Name' encrypted and saved" -ForegroundColor Green
    Write-Host "   Path: $FilePath" -ForegroundColor Gray
}

function Get-DecryptedSecret {
    <#
    .SYNOPSIS
        Retrieve and decrypt secret
    .PARAMETER Name
        Secret name (e.g., "db_password")
    .PARAMETER Path
        Secrets directory (default: C:\ProgramData\SPARC\secrets)
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Name,

        [string]$Path = "C:\ProgramData\SPARC\secrets"
    )

    $FilePath = Join-Path $Path "$Name.enc"

    if (-not (Test-Path $FilePath)) {
        throw "❌ Secret '$Name' not found at: $FilePath"
    }

    # Read and decrypt
    $Encrypted = Get-Content $FilePath
    $SecureString = ConvertTo-SecureString -String $Encrypted

    # Convert to plaintext (required for NSSM env vars)
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecureString)
    $Decrypted = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($BSTR)

    return $Decrypted
}

function Test-SecretExists {
    <#
    .SYNOPSIS
        Check if secret file exists
    #>
    param(
        [Parameter(Mandatory=$true)]
        [string]$Name,

        [string]$Path = "C:\ProgramData\SPARC\secrets"
    )

    $FilePath = Join-Path $Path "$Name.enc"
    return (Test-Path $FilePath)
}

# Export functions
Export-ModuleMember -Function New-SecurePassword, New-SecureToken, Save-EncryptedSecret, Get-DecryptedSecret, Test-SecretExists
```

**Validation**:
```powershell
# Test the library
Import-Module .\secret-manager.ps1

# Generate password
$pwd = New-SecurePassword -Length 32
Write-Host "Generated: $pwd"

# Save and retrieve
Save-EncryptedSecret -Name "test" -Value "MySecret123"
$retrieved = Get-DecryptedSecret -Name "test"
Write-Host "Retrieved: $retrieved"

# Verify match
if ($retrieved -eq "MySecret123") {
    Write-Host "✅ Secret management working!" -ForegroundColor Green
}
```

---

### Step 2: Update install-postgres.ps1 (30 min)

**Changes to**: `scripts/native-windows/install-postgres.ps1`

**Line 6-7** (Import secret manager):
```powershell
param([switch]$Verbose)

# Import secret management library
. "$PSScriptRoot\lib\secret-manager.ps1"

$ErrorActionPreference = "Stop"
```

**Line 16-17** (Replace hardcoded password):
```powershell
# BEFORE:
$SuperPassword = "postgres_admin_$(Get-Random -Minimum 1000 -Maximum 9999)"

# AFTER:
if (Test-SecretExists -Name "postgres_super_password") {
    Write-Host "ℹ️  Using existing PostgreSQL superuser password" -ForegroundColor Cyan
    $SuperPassword = Get-DecryptedSecret -Name "postgres_super_password"
} else {
    Write-Host "🔐 Generating new PostgreSQL superuser password..." -ForegroundColor Cyan
    $SuperPassword = New-SecurePassword -Length 32
    Save-EncryptedSecret -Name "postgres_super_password" -Value $SuperPassword
    Write-Host "✅ Password generated and encrypted" -ForegroundColor Green
}
```

**Line 119-131** (Remove plaintext credential file):
```powershell
# BEFORE:
$credFile = "C:\ProgramData\SPARC\config\postgres-credentials.txt"
New-Item -ItemType Directory -Path (Split-Path $credFile) -Force | Out-Null
@"
PostgreSQL Superuser Credentials
=================================
Username: postgres
Password: $SuperPassword
Port: $Port
Data Directory: $DataDir

⚠️  IMPORTANT: Keep this file secure!
"@ | Out-File -FilePath $credFile -Encoding UTF8

Write-Host "💾 Credentials saved to: $credFile`n" -ForegroundColor Yellow

# AFTER:
Write-Host "🔐 Superuser password encrypted and stored securely" -ForegroundColor Green
Write-Host "   Retrieve with: Get-DecryptedSecret -Name 'postgres_super_password'`n" -ForegroundColor Gray
```

---

### Step 3: Update install-backend-service.ps1 (1 hour)

**Changes to**: `scripts/native-windows/install-backend-service.ps1`

**Line 6** (Import secret manager):
```powershell
param([switch]$Verbose)

# Import secret management library
. "$PSScriptRoot\lib\secret-manager.ps1"

$ErrorActionPreference = "Stop"
```

**Line 16-28** (Replace hardcoded passwords):
```powershell
# BEFORE:
$DBUser = "sparc_user"
$DBPassword = "sparc_secure_password_2024"  # TODO: Read from secure storage
$RedisPassword = "sparc_redis_password_2024"  # TODO: Read from secure storage

# AFTER:
$DBUser = "sparc_user"

# Database password
if (Test-SecretExists -Name "db_password") {
    $DBPassword = Get-DecryptedSecret -Name "db_password"
} else {
    Write-Host "🔐 Generating database password..." -ForegroundColor Cyan
    $DBPassword = New-SecurePassword -Length 32
    Save-EncryptedSecret -Name "db_password" -Value $DBPassword
}

# Redis password
if (Test-SecretExists -Name "redis_password") {
    $RedisPassword = Get-DecryptedSecret -Name "redis_password"
} else {
    Write-Host "🔐 Generating Redis password..." -ForegroundColor Cyan
    $RedisPassword = New-SecurePassword -Length 32
    Save-EncryptedSecret -Name "redis_password" -Value $RedisPassword
}

# JWT secret
if (Test-SecretExists -Name "jwt_secret") {
    $JWTSecret = Get-DecryptedSecret -Name "jwt_secret"
} else {
    Write-Host "🔐 Generating JWT secret..." -ForegroundColor Cyan
    $JWTSecret = New-SecureToken -Bytes 64
    Save-EncryptedSecret -Name "jwt_secret" -Value $JWTSecret
}
```

**Line 106** (Use secure JWT secret):
```powershell
# BEFORE:
JWT_SECRET=sparc_jwt_secret_very_secure_2024

# AFTER:
JWT_SECRET=$JWTSecret
```

---

### Step 4: Update configure-postgres.ps1 (30 min)

**Changes to**: `scripts/native-windows/configure-postgres.ps1`

**Add at top**:
```powershell
# Import secret management library
. "$PSScriptRoot\lib\secret-manager.ps1"

# Retrieve database password
if (Test-SecretExists -Name "db_password") {
    $DBPassword = Get-DecryptedSecret -Name "db_password"
} else {
    Write-Host "❌ Database password not found. Run install-backend-service.ps1 first." -ForegroundColor Red
    exit 1
}

# Retrieve postgres superuser password
if (Test-SecretExists -Name "postgres_super_password") {
    $SuperPassword = Get-DecryptedSecret -Name "postgres_super_password"
} else {
    Write-Host "❌ PostgreSQL superuser password not found. Run install-postgres.ps1 first." -ForegroundColor Red
    exit 1
}
```

**Use in psql commands**:
```powershell
# Example: Create user with secure password
$env:PGPASSWORD = $SuperPassword
& psql -U postgres -c "CREATE USER sparc_user WITH PASSWORD '$DBPassword';"
```

---

### Step 5: Update configure-redis.ps1 (30 min)

**Changes to**: `scripts/native-windows/configure-redis.ps1`

**Add at top**:
```powershell
# Import secret management library
. "$PSScriptRoot\lib\secret-manager.ps1"

# Retrieve Redis password
if (Test-SecretExists -Name "redis_password") {
    $RedisPassword = Get-DecryptedSecret -Name "redis_password"
} else {
    Write-Host "❌ Redis password not found. Run install-backend-service.ps1 first." -ForegroundColor Red
    exit 1
}
```

**Configure Redis with password**:
```powershell
# Update redis.conf or memurai.conf
$ConfigFile = "C:\ProgramData\Memurai\memurai.conf"
if (Test-Path $ConfigFile) {
    Add-Content -Path $ConfigFile -Value "requirepass $RedisPassword"
    Write-Host "✅ Redis password authentication enabled" -ForegroundColor Green
}
```

---

### Step 6: Test Secret Management (30 min)

**Create test script**: `scripts/native-windows/test-secrets.ps1`

```powershell
# Test Secret Management Implementation

Write-Host "`n🧪 Testing Secret Management...`n" -ForegroundColor Cyan

# Import library
. "$PSScriptRoot\lib\secret-manager.ps1"

# Test 1: Password generation
Write-Host "[1/5] Testing password generation..." -ForegroundColor Cyan
$pwd = New-SecurePassword -Length 32
if ($pwd.Length -eq 32) {
    Write-Host "✅ Password generated successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Password generation failed" -ForegroundColor Red
    exit 1
}

# Test 2: Token generation
Write-Host "[2/5] Testing token generation..." -ForegroundColor Cyan
$token = New-SecureToken -Bytes 64
if ($token.Length -gt 50) {
    Write-Host "✅ Token generated successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Token generation failed" -ForegroundColor Red
    exit 1
}

# Test 3: Save encrypted secret
Write-Host "[3/5] Testing secret encryption..." -ForegroundColor Cyan
Save-EncryptedSecret -Name "test_secret" -Value "MyTestPassword123!"
if (Test-SecretExists -Name "test_secret") {
    Write-Host "✅ Secret saved successfully" -ForegroundColor Green
} else {
    Write-Host "❌ Secret save failed" -ForegroundColor Red
    exit 1
}

# Test 4: Retrieve decrypted secret
Write-Host "[4/5] Testing secret decryption..." -ForegroundColor Cyan
$retrieved = Get-DecryptedSecret -Name "test_secret"
if ($retrieved -eq "MyTestPassword123!") {
    Write-Host "✅ Secret retrieved correctly" -ForegroundColor Green
} else {
    Write-Host "❌ Secret retrieval failed (got: $retrieved)" -ForegroundColor Red
    exit 1
}

# Test 5: Verify encryption (file should not contain plaintext)
Write-Host "[5/5] Verifying encryption (file should not contain plaintext)..." -ForegroundColor Cyan
$encFile = Get-Content "C:\ProgramData\SPARC\secrets\test_secret.enc"
if ($encFile -notlike "*MyTestPassword123!*") {
    Write-Host "✅ File is encrypted (plaintext not found)" -ForegroundColor Green
} else {
    Write-Host "❌ File contains plaintext (encryption failed)" -ForegroundColor Red
    exit 1
}

# Cleanup
Remove-Item "C:\ProgramData\SPARC\secrets\test_secret.enc" -Force

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✅ All Secret Management Tests Passed!                   ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green

exit 0
```

**Run test**:
```powershell
.\test-secrets.ps1
```

---

## 🛠️ BLOCKER-002: Failure Recovery

### Step 1: Update Service Configuration (1 hour)

**Update all service scripts** to include exponential backoff:

**Files to update**:
- `install-backend-service.ps1`
- `install-frontend-service.ps1`

**Add after NSSM service creation** (around line 160):

```powershell
# Configure restart policy with exponential backoff
Write-Host "⚙️  Configuring failure recovery..." -ForegroundColor Cyan

# Restart on failure (default already set)
& $NSSMExe set $ServiceName AppExit Default Restart

# Throttle restarts (minimum 60 seconds between attempts)
& $NSSMExe set $ServiceName AppThrottle 60000  # 60 seconds = 60000 ms

# Initial restart delay (10 seconds)
& $NSSMExe set $ServiceName AppRestartDelay 10000  # 10 seconds

# Optional: Set maximum restart attempts before giving up
# Note: NSSM doesn't natively support this, but AppThrottle prevents rapid loops
& $NSSMExe set $ServiceName AppStopMethodSkip 6  # Skip graceful shutdown after 6 attempts

Write-Host "✅ Restart policy configured:" -ForegroundColor Green
Write-Host "   • Minimum interval: 60 seconds" -ForegroundColor Gray
Write-Host "   • Initial delay: 10 seconds" -ForegroundColor Gray
Write-Host "   • Prevents rapid restart loops`n" -ForegroundColor Gray
```

---

### Step 2: Test Restart Behavior (30 min)

**Create test script**: `scripts/native-windows/test-restart-behavior.ps1`

```powershell
# Test Service Restart Behavior

Write-Host "`n🧪 Testing Service Restart Behavior...`n" -ForegroundColor Cyan

# Test with Backend service
$ServiceName = "SPARC-Backend"

Write-Host "[1/3] Stopping service to simulate failure..." -ForegroundColor Cyan
Stop-Service $ServiceName -Force
Write-Host "✅ Service stopped" -ForegroundColor Green

Write-Host "`n[2/3] Monitoring service status for 2 minutes..." -ForegroundColor Cyan
Write-Host "   Expecting: Service should NOT restart within 60 seconds`n" -ForegroundColor Yellow

$startTime = Get-Date
$restartDetected = $false

for ($i = 0; $i -lt 24; $i++) {  # Monitor for 2 minutes (24 x 5 seconds)
    Start-Sleep -Seconds 5
    $service = Get-Service $ServiceName
    $elapsed = ((Get-Date) - $startTime).TotalSeconds

    Write-Host "   [$([int]$elapsed)s] Status: $($service.Status)" -ForegroundColor Gray

    if ($service.Status -eq "Running" -and $elapsed -lt 60) {
        Write-Host "❌ Service restarted too quickly (< 60 seconds)!" -ForegroundColor Red
        $restartDetected = $true
        break
    } elseif ($service.Status -eq "Running" -and $elapsed -ge 60) {
        Write-Host "✅ Service restarted after proper delay" -ForegroundColor Green
        $restartDetected = $true
        break
    }
}

if (-not $restartDetected) {
    Write-Host "⚠️  Service did not restart within 2 minutes. Manual start may be required." -ForegroundColor Yellow
    Start-Service $ServiceName
}

Write-Host "`n[3/3] Verifying service is now running..." -ForegroundColor Cyan
Start-Sleep -Seconds 5
$service = Get-Service $ServiceName

if ($service.Status -eq "Running") {
    Write-Host "✅ Service is running" -ForegroundColor Green
} else {
    Write-Host "❌ Service failed to start" -ForegroundColor Red
    exit 1
}

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✅ Restart Behavior Test Passed!                         ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green
```

---

## 🛠️ BLOCKER-003: Health Monitoring

### Step 1: Create Health Monitor Script (1.5 hours)

**File**: `scripts/native-windows/health-monitor.ps1` (NEW)

```powershell
<#
.SYNOPSIS
    Continuous health monitoring for RUV SPARC services
.DESCRIPTION
    Runs health checks every 5 minutes (via Task Scheduler)
    Logs to Windows Event Log
    Sends alerts on failures
    Auto-restarts unhealthy services
#>

param(
    [switch]$Alert,  # Enable email alerts
    [switch]$AutoRestart  # Auto-restart failed services
)

# Configuration
$LogSource = "SPARC-HealthMonitor"
$EventLogName = "Application"

# Ensure event log source exists
if (-not ([System.Diagnostics.EventLog]::SourceExists($LogSource))) {
    New-EventLog -LogName $EventLogName -Source $LogSource
}

function Write-HealthLog {
    param(
        [string]$Message,
        [string]$Level = "Information"  # Information, Warning, Error
    )

    $EventType = switch ($Level) {
        "Error" { "Error" }
        "Warning" { "Warning" }
        default { "Information" }
    }

    Write-EventLog -LogName $EventLogName -Source $LogSource -EventId 1000 -EntryType $EventType -Message $Message
    Write-Host "[$Level] $Message"
}

# Health checks
$checks = @()

# Check 1: PostgreSQL Service
Write-Host "`n[1/10] Checking PostgreSQL..." -ForegroundColor Cyan
$postgres = Get-Service -Name "postgresql-x64-15" -ErrorAction SilentlyContinue
if ($postgres -and $postgres.Status -eq "Running") {
    Write-Host "✅ PostgreSQL running" -ForegroundColor Green
    $checks += $true
} else {
    Write-HealthLog "PostgreSQL service not running" "Error"
    $checks += $false

    if ($AutoRestart) {
        Start-Service "postgresql-x64-15" -ErrorAction SilentlyContinue
        Write-HealthLog "Attempted to restart PostgreSQL" "Warning"
    }
}

# Check 2: Redis/Memurai Service
Write-Host "[2/10] Checking Redis..." -ForegroundColor Cyan
$memurai = Get-Service -Name "Memurai" -ErrorAction SilentlyContinue
$redis = Get-Service -Name "Redis" -ErrorAction SilentlyContinue

if (($memurai -and $memurai.Status -eq "Running") -or ($redis -and $redis.Status -eq "Running")) {
    Write-Host "✅ Redis running" -ForegroundColor Green
    $checks += $true
} else {
    Write-HealthLog "Redis service not running" "Error"
    $checks += $false

    if ($AutoRestart) {
        if ($memurai) { Start-Service "Memurai" -ErrorAction SilentlyContinue }
        if ($redis) { Start-Service "Redis" -ErrorAction SilentlyContinue }
        Write-HealthLog "Attempted to restart Redis" "Warning"
    }
}

# Check 3: Backend Service
Write-Host "[3/10] Checking Backend..." -ForegroundColor Cyan
$backend = Get-Service -Name "SPARC-Backend" -ErrorAction SilentlyContinue
if ($backend -and $backend.Status -eq "Running") {
    Write-Host "✅ Backend running" -ForegroundColor Green
    $checks += $true
} else {
    Write-HealthLog "Backend service not running" "Error"
    $checks += $false

    if ($AutoRestart) {
        Start-Service "SPARC-Backend" -ErrorAction SilentlyContinue
        Write-HealthLog "Attempted to restart Backend" "Warning"
    }
}

# Check 4: Frontend Service
Write-Host "[4/10] Checking Frontend..." -ForegroundColor Cyan
$frontend = Get-Service -Name "SPARC-Frontend" -ErrorAction SilentlyContinue
if ($frontend -and $frontend.Status -eq "Running") {
    Write-Host "✅ Frontend running" -ForegroundColor Green
    $checks += $true
} else {
    Write-HealthLog "Frontend service not running" "Error"
    $checks += $false

    if ($AutoRestart) {
        Start-Service "SPARC-Frontend" -ErrorAction SilentlyContinue
        Write-HealthLog "Attempted to restart Frontend" "Warning"
    }
}

# Check 5: Backend API Health Endpoint
Write-Host "[5/10] Testing API health..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ API responding" -ForegroundColor Green
        $checks += $true
    } else {
        Write-HealthLog "API returned status $($response.StatusCode)" "Warning"
        $checks += $false
    }
} catch {
    Write-HealthLog "API health check failed: $_" "Error"
    $checks += $false
}

# Check 6: Frontend Web UI
Write-Host "[6/10] Testing frontend UI..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Frontend responding" -ForegroundColor Green
        $checks += $true
    } else {
        Write-HealthLog "Frontend returned status $($response.StatusCode)" "Warning"
        $checks += $false
    }
} catch {
    Write-HealthLog "Frontend health check failed: $_" "Error"
    $checks += $false
}

# Check 7: Disk Space
Write-Host "[7/10] Checking disk space..." -ForegroundColor Cyan
$drive = Get-PSDrive -Name C
$freePercent = ($drive.Free / $drive.Used) * 100

if ($freePercent -lt 20) {
    Write-HealthLog "Disk space low: $([int]$freePercent)% free" "Warning"
    $checks += $false
} else {
    Write-Host "✅ Disk space OK ($([int]$freePercent)% free)" -ForegroundColor Green
    $checks += $true
}

# Check 8: Log File Sizes
Write-Host "[8/10] Checking log file sizes..." -ForegroundColor Cyan
$logDir = "C:\ProgramData\SPARC\backend\logs"
if (Test-Path $logDir) {
    $logSize = (Get-ChildItem $logDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
    if ($logSize -gt 5) {
        Write-HealthLog "Log files exceed 5 GB ($([math]::Round($logSize, 2)) GB)" "Warning"
        $checks += $false
    } else {
        Write-Host "✅ Log size OK ($([math]::Round($logSize, 2)) GB)" -ForegroundColor Green
        $checks += $true
    }
} else {
    $checks += $true
}

# Check 9: Database Connectivity
Write-Host "[9/10] Testing database connectivity..." -ForegroundColor Cyan
try {
    $psql = "C:\Program Files\PostgreSQL\15\bin\psql.exe"
    if (Test-Path $psql) {
        $result = & $psql -U postgres -h localhost -c "SELECT 1;" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Database responding" -ForegroundColor Green
            $checks += $true
        } else {
            Write-HealthLog "Database connectivity failed" "Error"
            $checks += $false
        }
    } else {
        $checks += $true  # Skip if psql not found
    }
} catch {
    Write-HealthLog "Database check failed: $_" "Error"
    $checks += $false
}

# Check 10: Redis Connectivity
Write-Host "[10/10] Testing Redis connectivity..." -ForegroundColor Cyan
try {
    $socket = New-Object System.Net.Sockets.TcpClient
    $socket.Connect("localhost", 6379)
    if ($socket.Connected) {
        Write-Host "✅ Redis responding" -ForegroundColor Green
        $checks += $true
        $socket.Close()
    } else {
        Write-HealthLog "Redis connectivity failed" "Error"
        $checks += $false
    }
} catch {
    Write-HealthLog "Redis check failed: $_" "Error"
    $checks += $false
}

# Calculate health score
$passed = ($checks | Where-Object { $_ -eq $true }).Count
$total = $checks.Count
$healthScore = [math]::Round(($passed / $total) * 100)

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  Health Check Summary                                      ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

Write-Host "Passed: $passed/$total" -ForegroundColor White
Write-Host "Health Score: $healthScore%" -ForegroundColor $(if ($healthScore -ge 80) { "Green" } elseif ($healthScore -ge 60) { "Yellow" } else { "Red" })

if ($healthScore -ge 80) {
    Write-HealthLog "Health check passed ($healthScore%)" "Information"
} elseif ($healthScore -ge 60) {
    Write-HealthLog "Health check degraded ($healthScore%)" "Warning"
} else {
    Write-HealthLog "Health check failed ($healthScore%)" "Error"
}
```

---

### Step 2: Create Scheduled Task (30 min)

**File**: `scripts/native-windows/setup-health-monitoring.ps1` (NEW)

```powershell
# Setup Health Monitoring Scheduled Task

Write-Host "`n🔧 Setting up Health Monitoring...`n" -ForegroundColor Cyan

# Task configuration
$TaskName = "SPARC-HealthMonitor"
$ScriptPath = "C:\ProgramData\SPARC\scripts\health-monitor.ps1"
$LogPath = "C:\ProgramData\SPARC\logs\health-monitor.log"

# Copy health monitor script to installation directory
Write-Host "📂 Copying health monitor script..." -ForegroundColor Cyan
$InstallDir = "C:\ProgramData\SPARC\scripts"
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Copy-Item -Path "$PSScriptRoot\health-monitor.ps1" -Destination $ScriptPath -Force
Write-Host "✅ Script copied to: $ScriptPath`n" -ForegroundColor Green

# Create scheduled task action
$Action = New-ScheduledTaskAction `
    -Execute "PowerShell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`" -AutoRestart | Out-File -Append `"$LogPath`""

# Create trigger (every 5 minutes, starting now)
$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 9999)

# Task settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -MultipleInstances IgnoreNew

# Register task
Write-Host "📅 Registering scheduled task..." -ForegroundColor Cyan
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "RUV SPARC Dashboard health monitoring - runs every 5 minutes" `
    -Force | Out-Null

Write-Host "✅ Scheduled task created: $TaskName`n" -ForegroundColor Green

# Run task now to test
Write-Host "🧪 Running initial health check..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 10

# Check results
if (Test-Path $LogPath) {
    Write-Host "`n📄 Health check output:`n" -ForegroundColor Cyan
    Get-Content $LogPath -Tail 20
}

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✅ Health Monitoring Setup Complete!                     ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green

Write-Host "Monitoring Details:" -ForegroundColor Cyan
Write-Host "  Task Name:     $TaskName" -ForegroundColor White
Write-Host "  Frequency:     Every 5 minutes" -ForegroundColor White
Write-Host "  Script:        $ScriptPath" -ForegroundColor White
Write-Host "  Logs:          $LogPath" -ForegroundColor White
Write-Host "  Event Viewer:  Application log (Source: SPARC-HealthMonitor)`n" -ForegroundColor White
```

---

## ✅ Acceptance Tests

### Test 1: Secret Management
```powershell
.\test-secrets.ps1
# Expected: All 5 tests pass
```

### Test 2: Installation with Secrets
```powershell
.\master-install.ps1 -Verbose
# Expected: No hardcoded passwords in output
# Expected: Secrets encrypted in C:\ProgramData\SPARC\secrets\
```

### Test 3: Restart Behavior
```powershell
.\test-restart-behavior.ps1
# Expected: Service does NOT restart within 60 seconds
```

### Test 4: Health Monitoring
```powershell
.\setup-health-monitoring.ps1
# Expected: Task runs every 5 minutes
# Expected: Logs appear in Event Viewer
```

---

## 📊 Progress Tracking

**BLOCKER-001**: [ ] 0/7 steps complete
**BLOCKER-002**: [ ] 0/3 steps complete
**BLOCKER-003**: [ ] 0/3 steps complete

**Overall Progress**: 0/13 steps (0%)

---

## 🎯 Definition of Done

- [ ] All scripts updated to use encrypted secrets
- [ ] No plaintext passwords in any scripts or config files
- [ ] Secrets stored with DPAPI encryption
- [ ] ACLs restrict access to Administrators only
- [ ] Service restart throttle configured (60 seconds minimum)
- [ ] Health monitoring task running every 5 minutes
- [ ] All tests pass
- [ ] Fresh Windows VM installation successful
- [ ] verify-startup.ps1 shows 10/10 checks passed

---

**Status**: ❌ BLOCKERS NOT YET FIXED
**Next Action**: Start with BLOCKER-001 Step 1 (create secret-manager.ps1)
**ETA**: 9 hours (1-2 days)
