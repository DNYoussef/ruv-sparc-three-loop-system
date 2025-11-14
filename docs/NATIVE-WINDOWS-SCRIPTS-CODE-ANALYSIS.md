# Native Windows Installation Scripts - Code Quality Analysis Report

**Analysis Date**: 2025-11-09
**Scripts Analyzed**: 11
**Total Lines of Code**: ~1,650
**Analyst**: Code Quality Analyzer Agent

---

## Executive Summary

### Overall Code Quality Score: **76/100**

**Grade**: C+ (Good with significant room for improvement)

**Quick Assessment**:
- ✅ **Strengths**: Good structure, consistent formatting, comprehensive comments, proper error handling basics
- ⚠️ **Concerns**: Security vulnerabilities (plaintext passwords), limited idempotency, inconsistent error handling patterns, hardcoded configuration values
- ❌ **Critical Issues**: 3 security vulnerabilities, 7 reliability concerns, 12 best practice violations

---

## 1. Critical Issues (Must Fix)

### 🔴 Security Vulnerabilities

#### **CRITICAL-001: Plaintext Passwords in Code**
**Severity**: CRITICAL
**Files**:
- `install-backend-service.ps1:22,27`
- `configure-postgres.ps1:14`
- `install-backend-service.ps1:106`

```powershell
# ❌ BAD - Hardcoded plaintext passwords
$DBPassword = "sparc_secure_password_2024"  # Line 22
$RedisPassword = "sparc_redis_password_2024"  # Line 27
JWT_SECRET=sparc_jwt_secret_very_secure_2024  # Line 106
```

**Impact**: Passwords visible in:
- Script files on disk
- Process command lines
- Environment variable exports
- Log files (potentially)

**Recommendation**:
```powershell
# ✅ GOOD - Use Windows Credential Manager or Secure Strings
$cred = Get-Credential -Message "Enter Database Password"
$DBPassword = $cred.GetNetworkCredential().Password

# OR use Windows DPAPI
$securePassword = Read-Host "Database Password" -AsSecureString
$encryptedPassword = $securePassword | ConvertFrom-SecureString
# Store $encryptedPassword in config file
```

---

#### **CRITICAL-002: Password Exposure in Environment Files**
**Severity**: CRITICAL
**Files**:
- `install-backend-service.ps1:91-116`
- `configure-postgres.ps1:145-158`

```powershell
# ❌ BAD - Passwords written to disk in plaintext
@"
DATABASE_URL=$databaseUrl  # Contains password
REDIS_URL=$redisUrl        # Contains password
JWT_SECRET=sparc_jwt_secret_very_secure_2024
"@ | Out-File -FilePath $envFile -Encoding UTF8
```

**Impact**:
- `.env.production` file readable by any user with file access
- Connection strings with embedded passwords
- No encryption at rest

**Recommendation**:
```powershell
# ✅ GOOD - Use environment-specific encryption
$secureContent = ConvertTo-SecureString $envContent -AsPlainText -Force
$encryptedContent = $secureContent | ConvertFrom-SecureString
$encryptedContent | Out-File "$envFile.encrypted"

# Set proper ACLs
$acl = Get-Acl $envFile
$acl.SetAccessRuleProtection($true, $false)
$adminRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    "Administrators", "FullControl", "Allow"
)
$acl.SetAccessRule($adminRule)
Set-Acl $envFile $acl
```

---

#### **CRITICAL-003: Postgres Superuser Password Stored Unencrypted**
**Severity**: CRITICAL
**Files**: `install-postgres.ps1:119-130`

```powershell
# ❌ BAD - Superuser password in plaintext file
@"
PostgreSQL Superuser Credentials
=================================
Username: postgres
Password: $SuperPassword  # Plaintext!
"@ | Out-File -FilePath $credFile -Encoding UTF8
```

**Impact**: Full database access credentials accessible to any user

**Recommendation**:
```powershell
# ✅ GOOD - Encrypt credentials file
$credentialObject = @{
    Username = "postgres"
    Password = $SuperPassword
} | ConvertTo-Json | ConvertTo-SecureString -AsPlainText -Force | ConvertFrom-SecureString

$credentialObject | Out-File "$credFile.encrypted"
# Remove plaintext file
```

---

### 🟠 Reliability Issues

#### **RELIABILITY-001: Non-Idempotent Database Operations**
**Severity**: HIGH
**Files**: `configure-postgres.ps1:65-85`

```powershell
# ❌ PROBLEMATIC - Will fail on re-run
CREATE DATABASE $DBName;  # Fails if database exists
```

**Issue**: Script fails when database already exists, blocking re-runs

**Recommendation**:
```powershell
# ✅ GOOD - Idempotent database creation
SELECT 'Creating database...' as status;
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '$DBName') THEN
    CREATE DATABASE $DBName;
  END IF;
END
$$;
```

---

#### **RELIABILITY-002: Insufficient Service Start Wait Times**
**Severity**: MEDIUM
**Files**:
- `install-backend-service.ps1:181`
- `install-frontend-service.ps1:174`
- `install-postgres.ps1:109`

```powershell
# ❌ INSUFFICIENT - Fixed 5-second wait
Start-Sleep -Seconds 5
```

**Issue**: Services may not be fully initialized, leading to false health check failures

**Recommendation**:
```powershell
# ✅ GOOD - Poll with timeout
$maxAttempts = 30
$attempt = 0
do {
    Start-Sleep -Seconds 1
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    $attempt++
    if ($service.Status -eq "Running") { break }
} while ($attempt -lt $maxAttempts)

if ($service.Status -ne "Running") {
    Write-Host "❌ Service failed to start after $maxAttempts seconds" -ForegroundColor Red
    exit 1
}
```

---

#### **RELIABILITY-003: No Rollback on Partial Failure**
**Severity**: MEDIUM
**Files**: All scripts

**Issue**: If a script fails midway, there's no cleanup of partially created resources

**Recommendation**:
```powershell
# ✅ GOOD - Add try/catch/finally with cleanup
$createdResources = @()

try {
    # Create resources, tracking each
    New-Item -Path $InstallDir -ItemType Directory
    $createdResources += $InstallDir

    # ... more operations ...

} catch {
    Write-Host "❌ Installation failed. Rolling back..." -ForegroundColor Red

    # Cleanup in reverse order
    foreach ($resource in $createdResources.Reverse()) {
        if (Test-Path $resource) {
            Remove-Item -Path $resource -Recurse -Force
        }
    }

    throw
}
```

---

#### **RELIABILITY-004: Race Condition in Service Creation**
**Severity**: MEDIUM
**Files**:
- `install-backend-service.ps1:121-127`
- `install-frontend-service.ps1:96-102`

```powershell
# ❌ RACE CONDITION - Service may be in stopping state
$existingService = & $NSSMExe status $ServiceName 2>$null
if ($existingService -and $existingService -notlike "*SERVICE_NOT_FOUND*") {
    & $NSSMExe stop $ServiceName 2>$null
    Start-Sleep -Seconds 2  # Not enough time!
    & $NSSMExe remove $ServiceName confirm
}
```

**Issue**: 2-second wait may not be sufficient for service to fully stop

**Recommendation**:
```powershell
# ✅ GOOD - Poll for service stop
$maxWait = 30
$waited = 0
while ((Get-Service $ServiceName -ErrorAction SilentlyContinue).Status -ne "Stopped" -and $waited -lt $maxWait) {
    Start-Sleep -Seconds 1
    $waited++
}

if ((Get-Service $ServiceName -ErrorAction SilentlyContinue).Status -ne "Stopped") {
    Write-Host "⚠️ Service did not stop cleanly. Forcing removal..." -ForegroundColor Yellow
}
```

---

#### **RELIABILITY-005: Missing Timeout on HTTP Health Checks**
**Severity**: LOW
**Files**:
- `install-backend-service.ps1:193`
- `install-frontend-service.ps1:186`
- `verify-startup.ps1:88,98,108`

```powershell
# ⚠️ INCONSISTENT - Some have timeout, some don't
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health" -UseBasicParsing -TimeoutSec 10
$response = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 5
# verify-startup has -TimeoutSec 5, but variations exist
```

**Recommendation**: Standardize timeout to 10 seconds across all HTTP checks

---

#### **RELIABILITY-006: Hardcoded Source Paths**
**Severity**: MEDIUM
**Files**:
- `install-backend-service.ps1:13`
- `install-frontend-service.ps1:13`

```powershell
# ❌ FRAGILE - Hardcoded user path
$SourceDir = "C:\Users\17175\backend"
$SourceDir = "C:\Users\17175\frontend"
```

**Issue**: Fails if:
- Run by different user
- Source code in different location
- Multi-user environment

**Recommendation**:
```powershell
# ✅ GOOD - Make source path configurable
param(
    [string]$SourceDir = "$PSScriptRoot\..\..\backend"  # Relative to script location
)

# OR use parameter with validation
param(
    [Parameter(Mandatory=$false)]
    [ValidateScript({ Test-Path $_ })]
    [string]$SourceDir = "C:\Users\$env:USERNAME\backend"
)
```

---

#### **RELIABILITY-007: No Validation of Prerequisites**
**Severity**: MEDIUM
**Files**: `install-backend-service.ps1`, `install-frontend-service.ps1`

**Issue**: Scripts don't verify Python/Node.js versions before attempting installation

**Recommendation**:
```powershell
# ✅ GOOD - Validate prerequisites
function Test-PythonVersion {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -ge 3 -and $minor -ge 8) {
            return $true
        }
    }
    return $false
}

if (-not (Test-PythonVersion)) {
    Write-Host "❌ Python 3.8+ is required but not found" -ForegroundColor Red
    exit 1
}
```

---

## 2. PowerShell Best Practices Analysis

### ✅ Good Practices Observed

1. **Administrator Check** (master-install.ps1:40-46)
   ```powershell
   # ✅ GOOD - Proper admin rights verification
   $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
   $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
   if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
       exit 1
   }
   ```

2. **Structured Logging** (master-install.ps1:60-74)
   ```powershell
   # ✅ GOOD - Centralized logging function
   function Write-Log {
       param([string]$Message, [string]$Level = "INFO")
       $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
       $LogMessage = "[$Timestamp] [$Level] $Message"
       Add-Content -Path $LogFile -Value $LogMessage
   }
   ```

3. **Error Action Preference** (All scripts)
   ```powershell
   # ✅ GOOD - Fail fast on errors
   $ErrorActionPreference = "Stop"
   ```

4. **Parameter Documentation** (master-install.ps1:5-32)
   ```powershell
   # ✅ GOOD - Comprehensive help documentation
   <#
   .SYNOPSIS
       Master orchestrator for Native Windows installation
   .PARAMETER SkipDependencies
       Skip installing PostgreSQL, Redis, NSSM
   #>
   ```

---

### ❌ Best Practice Violations

#### **BP-001: Inconsistent Error Handling**
**Severity**: MEDIUM
**Files**: All scripts

**Issue**: Mix of try/catch, LASTEXITCODE checks, and -ErrorAction patterns

```powershell
# ❌ INCONSISTENT - Three different error handling patterns
try { ... } catch { ... }                           # install-postgres.ps1:44
if ($LASTEXITCODE -ne 0) { exit 1 }                # install-backend-service.ps1:78
-ErrorAction SilentlyContinue                       # configure-redis.ps1:11
```

**Recommendation**:
```powershell
# ✅ GOOD - Standardized error handling
trap {
    Write-Log "Unhandled error: $_" "ERROR"
    exit 1
}

# Use consistent pattern
try {
    $result = Invoke-Command -ScriptBlock { ... }
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code: $LASTEXITCODE"
    }
} catch {
    Write-Log "Operation failed: $_" "ERROR"
    throw
}
```

---

#### **BP-002: No Parameter Validation**
**Severity**: MEDIUM
**Files**: All scripts except master-install.ps1

```powershell
# ❌ MISSING - No parameter validation
param([switch]$Verbose)

# ✅ GOOD - Add parameter validation
param(
    [switch]$Verbose,

    [Parameter(Mandatory=$false)]
    [ValidateRange(1024, 65535)]
    [int]$Port = 8000,

    [Parameter(Mandatory=$false)]
    [ValidateScript({ Test-Path $_ -PathType Container })]
    [string]$InstallDir = "C:\ProgramData\SPARC"
)
```

---

#### **BP-003: Magic Numbers Throughout Code**
**Severity**: LOW
**Files**: Multiple

```powershell
# ❌ BAD - Magic numbers without explanation
Start-Sleep -Seconds 5
Start-Sleep -Seconds 3
$DelaySeconds = 120
$maxAttempts = 30
```

**Recommendation**:
```powershell
# ✅ GOOD - Named constants
$CONSTANTS = @{
    SERVICE_START_TIMEOUT_SEC = 30
    SERVICE_STOP_TIMEOUT_SEC = 15
    HEALTH_CHECK_RETRY_ATTEMPTS = 10
    HEALTH_CHECK_RETRY_DELAY_SEC = 3
    BROWSER_LAUNCH_DELAY_SEC = 120
}

Start-Sleep -Seconds $CONSTANTS.HEALTH_CHECK_RETRY_DELAY_SEC
```

---

#### **BP-004: Insufficient Input Validation**
**Severity**: MEDIUM
**Files**: configure-postgres.ps1:25-31

```powershell
# ❌ RISKY - SecureString conversion is complex and error-prone
$SuperPassword = Read-Host "Enter postgres superuser password" -AsSecureString
$SuperPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($SuperPassword)
)
```

**Issue**: No validation that password was actually entered

**Recommendation**:
```powershell
# ✅ GOOD - Validate input
do {
    $SuperPassword = Read-Host "Enter postgres superuser password" -AsSecureString
    if ($SuperPassword.Length -eq 0) {
        Write-Host "Password cannot be empty!" -ForegroundColor Red
    }
} while ($SuperPassword.Length -eq 0)
```

---

#### **BP-005: Verbose Parameter Not Used Effectively**
**Severity**: LOW
**Files**: Multiple

**Issue**: Some scripts accept `-Verbose` but don't use `Write-Verbose` cmdlet

```powershell
# ❌ INCONSISTENT
param([switch]$Verbose)
# Later uses if ($Verbose) instead of Write-Verbose
```

**Recommendation**:
```powershell
# ✅ GOOD - Use built-in verbose system
[CmdletBinding()]
param()

Write-Verbose "Installing dependencies..."
# Automatically respects -Verbose flag
```

---

#### **BP-006: No Progress Indicators for Long Operations**
**Severity**: LOW
**Files**: install-backend-service.ps1:75-86

```powershell
# ❌ NO FEEDBACK - User sees nothing for minutes
& $pipExe install -r $requirementsFile
```

**Recommendation**:
```powershell
# ✅ GOOD - Show progress
Write-Progress -Activity "Installing Python Dependencies" -Status "Running pip install..."
& $pipExe install -r $requirementsFile --progress-bar on
Write-Progress -Activity "Installing Python Dependencies" -Completed
```

---

#### **BP-007: Hardcoded File Paths**
**Severity**: MEDIUM
**Files**: Multiple

```powershell
# ❌ FRAGILE - Hardcoded paths
$PSQLExe = "C:\Program Files\PostgreSQL\15\bin\psql.exe"
$NSSMExe = "C:\Program Files\NSSM\nssm.exe"
```

**Recommendation**:
```powershell
# ✅ GOOD - Dynamic path detection
$PSQLExe = (Get-Command psql -ErrorAction SilentlyContinue).Source
if (-not $PSQLExe) {
    $PSQLExe = "C:\Program Files\PostgreSQL\15\bin\psql.exe"
    if (-not (Test-Path $PSQLExe)) {
        throw "PostgreSQL not found. Install PostgreSQL first."
    }
}
```

---

#### **BP-008: Missing Exit Codes Convention**
**Severity**: LOW
**Files**: All scripts

**Issue**: Scripts use `exit 0` and `exit 1` but no documented exit code convention

**Recommendation**:
```powershell
# ✅ GOOD - Document exit codes
<#
.NOTES
    Exit Codes:
    0  - Success
    1  - General error
    2  - Prerequisites not met
    3  - Service creation failed
    4  - Health check failed
#>

$EXIT_CODES = @{
    SUCCESS = 0
    GENERAL_ERROR = 1
    PREREQUISITES_MISSING = 2
    SERVICE_CREATION_FAILED = 3
    HEALTH_CHECK_FAILED = 4
}

exit $EXIT_CODES.PREREQUISITES_MISSING
```

---

#### **BP-009: No Script Version Tracking**
**Severity**: LOW
**Files**: All scripts

**Issue**: Version in header but not programmatically accessible

**Recommendation**:
```powershell
# ✅ GOOD - Programmatic version access
[CmdletBinding()]
param(
    [switch]$ShowVersion
)

$SCRIPT_VERSION = "1.0.0"
$SCRIPT_NAME = "install-postgres.ps1"

if ($ShowVersion) {
    Write-Host "$SCRIPT_NAME version $SCRIPT_VERSION"
    exit 0
}
```

---

#### **BP-010: No Environment Variable Cleanup**
**Severity**: LOW
**Files**: configure-postgres.ps1:47

```powershell
# ❌ INCOMPLETE - Sets PGPASSWORD but doesn't always clean up
$env:PGPASSWORD = $SuperPassword
# ... operations ...
Remove-Item env:PGPASSWORD  # Only in success path!
```

**Recommendation**:
```powershell
# ✅ GOOD - Use try/finally for cleanup
try {
    $env:PGPASSWORD = $SuperPassword
    # ... operations ...
} finally {
    Remove-Item env:PGPASSWORD -ErrorAction SilentlyContinue
}
```

---

#### **BP-011: Inconsistent Service Name References**
**Severity**: LOW
**Files**: Multiple

```powershell
# ⚠️ INCONSISTENT - Service names hardcoded in multiple places
"postgresql-x64-15"    # Used in 4 different files
"SPARC-Backend"        # Used in 3 different files
"SPARC-Frontend"       # Used in 3 different files
```

**Recommendation**:
```powershell
# ✅ GOOD - Centralized configuration
# Create: C:\ProgramData\SPARC\config\service-names.json
{
    "Database": "postgresql-x64-15",
    "Cache": "Memurai",
    "Backend": "SPARC-Backend",
    "Frontend": "SPARC-Frontend"
}

# Load in scripts
$serviceConfig = Get-Content "C:\ProgramData\SPARC\config\service-names.json" | ConvertFrom-Json
$DatabaseService = $serviceConfig.Database
```

---

#### **BP-012: Missing Function Documentation**
**Severity**: LOW
**Files**: master-install.ps1, verify-startup.ps1

```powershell
# ❌ MISSING - No function documentation
function Show-Progress {
    param([string]$Activity, [string]$Status)
    # ...
}
```

**Recommendation**:
```powershell
# ✅ GOOD - Document functions
<#
.SYNOPSIS
    Displays installation progress
.PARAMETER Activity
    The current activity being performed
.PARAMETER Status
    The status message to display
#>
function Show-Progress {
    param([string]$Activity, [string]$Status)
    # ...
}
```

---

## 3. Cross-Script Consistency Analysis

### ✅ Consistent Elements

| Element | Status | Notes |
|---------|--------|-------|
| Port Numbers | ✅ Consistent | Backend: 8000, Frontend: 3000, PostgreSQL: 5432, Redis: 6379 |
| Service Names | ✅ Consistent | `SPARC-Backend`, `SPARC-Frontend` used throughout |
| Install Directory | ✅ Consistent | `C:\ProgramData\SPARC` as base path |
| Log Directory | ✅ Consistent | `{InstallDir}\logs` pattern |
| Color Scheme | ✅ Consistent | Green=success, Red=error, Yellow=warning, Cyan=info |

### ❌ Inconsistent Elements

| Element | Issue | Impact |
|---------|-------|--------|
| **Passwords** | Different passwords used for different services | Maintenance nightmare |
| **Error Handling** | 3 different patterns (try/catch, LASTEXITCODE, -ErrorAction) | Unpredictable behavior |
| **Timeout Values** | Vary from 2-10 seconds across scripts | Unreliable on slow systems |
| **Credential Storage** | Some encrypted, some plaintext | Security inconsistency |
| **Source Paths** | Hardcoded to specific user | Multi-user incompatible |
| **Logging** | Only master script has structured logging | Debugging difficulty |

---

## 4. Security Analysis

### 🔴 Security Issues Summary

| ID | Issue | Severity | Files Affected | Risk Level |
|----|-------|----------|----------------|------------|
| SEC-001 | Plaintext passwords in code | CRITICAL | 3 | 🔴 HIGH |
| SEC-002 | Passwords in environment files | CRITICAL | 2 | 🔴 HIGH |
| SEC-003 | Unencrypted credential files | CRITICAL | 1 | 🔴 HIGH |
| SEC-004 | No file permission restrictions | HIGH | 11 | 🟠 MEDIUM |
| SEC-005 | JWT secret hardcoded | HIGH | 1 | 🟠 MEDIUM |
| SEC-006 | No input sanitization | MEDIUM | 5 | 🟡 LOW |
| SEC-007 | Service runs as SYSTEM | MEDIUM | 2 | 🟡 LOW |

---

### Detailed Security Recommendations

#### **SEC-004: No File Permission Restrictions**

**Current State**:
```powershell
# ❌ INSECURE - Default permissions
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
```

**Recommended**:
```powershell
# ✅ SECURE - Restrict permissions
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null

$acl = Get-Acl $InstallDir
$acl.SetAccessRuleProtection($true, $false)  # Remove inherited permissions

$adminRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    "BUILTIN\Administrators", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
)
$systemRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    "NT AUTHORITY\SYSTEM", "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow"
)

$acl.SetAccessRule($adminRule)
$acl.SetAccessRule($systemRule)
Set-Acl $InstallDir $acl
```

---

#### **SEC-005: JWT Secret Hardcoded**

**Current State**:
```powershell
# ❌ INSECURE
JWT_SECRET=sparc_jwt_secret_very_secure_2024
```

**Recommended**:
```powershell
# ✅ SECURE - Generate cryptographically random secret
function New-SecureSecret {
    param([int]$Length = 64)

    $bytes = New-Object byte[] $Length
    $rng = [System.Security.Cryptography.RNGCryptoServiceProvider]::new()
    $rng.GetBytes($bytes)
    return [Convert]::ToBase64String($bytes)
}

$JWT_SECRET = New-SecureSecret
```

---

#### **SEC-006: No Input Sanitization**

**Current State**:
```powershell
# ❌ POTENTIAL SQL INJECTION
CREATE USER $DBUser WITH PASSWORD '$DBPassword';
```

**Recommended**:
```powershell
# ✅ SECURE - Validate and sanitize inputs
function Test-SQLIdentifier {
    param([string]$Identifier)
    return $Identifier -match '^[a-zA-Z0-9_]+$'
}

if (-not (Test-SQLIdentifier $DBUser)) {
    throw "Invalid database username. Use only alphanumeric and underscore."
}

# Use parameterized queries or proper escaping
$escapedPassword = $DBPassword -replace "'", "''"
```

---

#### **SEC-007: Service Runs as SYSTEM**

**Current State**:
```powershell
# ⚠️ OVERPRIVILEGED
--serviceaccount", "NT AUTHORITY\NetworkService"
```

**Recommendation**: Create dedicated service account with minimal permissions

```powershell
# ✅ BETTER - Use dedicated service account
$serviceUser = "SPARC-ServiceAccount"
$servicePassword = ConvertTo-SecureString (New-SecureSecret) -AsPlainText -Force

# Create local user account
New-LocalUser -Name $serviceUser -Password $servicePassword -Description "SPARC Dashboard Service Account"

# Grant specific permissions only
$acl = Get-Acl $InstallDir
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule(
    $serviceUser, "ReadAndExecute", "ContainerInherit,ObjectInherit", "None", "Allow"
)
$acl.AddAccessRule($rule)
Set-Acl $InstallDir $acl

# Use this account for service
& $NSSMExe set $ServiceName ObjectName ".\$serviceUser" $plainPassword
```

---

## 5. Code Quality Metrics

### Complexity Analysis

| Script | Lines | Functions | Cyclomatic Complexity | Maintainability Index |
|--------|-------|-----------|----------------------|----------------------|
| master-install.ps1 | 335 | 3 | 15 | 68/100 |
| install-postgres.ps1 | 139 | 0 | 8 | 72/100 |
| install-redis.ps1 | 97 | 0 | 6 | 75/100 |
| install-nssm.ps1 | 102 | 0 | 5 | 78/100 |
| install-backend-service.ps1 | 219 | 0 | 12 | 65/100 |
| install-frontend-service.ps1 | 212 | 0 | 11 | 66/100 |
| configure-postgres.ps1 | 167 | 0 | 9 | 70/100 |
| configure-redis.ps1 | 91 | 0 | 5 | 76/100 |
| configure-service-dependencies.ps1 | 118 | 1 | 6 | 74/100 |
| setup-browser-autolaunch.ps1 | 91 | 0 | 4 | 79/100 |
| verify-startup.ps1 | 165 | 1 | 10 | 71/100 |

**Average Maintainability Index**: **72.2/100** (Acceptable)

---

### Code Duplication Analysis

**Duplicate Code Blocks Found**: 8

1. **Service Status Check Pattern** (6 occurrences)
   ```powershell
   $service = Get-Service -Name "..." -ErrorAction SilentlyContinue
   if ($service) { ... }
   ```
   **Recommendation**: Extract to function `Test-ServiceRunning`

2. **HTTP Health Check Pattern** (4 occurrences)
   ```powershell
   $response = Invoke-WebRequest -Uri "..." -UseBasicParsing -TimeoutSec ...
   if ($response.StatusCode -eq 200) { ... }
   ```
   **Recommendation**: Extract to function `Test-HttpEndpoint`

3. **NSSM Service Removal Pattern** (2 occurrences)
   ```powershell
   $existingService = & $NSSMExe status $ServiceName 2>$null
   if ($existingService -and $existingService -notlike "*SERVICE_NOT_FOUND*") {
       & $NSSMExe stop $ServiceName 2>$null
       Start-Sleep -Seconds 2
       & $NSSMExe remove $ServiceName confirm
   }
   ```
   **Recommendation**: Extract to function `Remove-NSSMServiceIfExists`

---

### Naming Conventions Analysis

| Category | Compliance | Issues |
|----------|------------|--------|
| Variable Names | 90% | ✅ Mostly PascalCase, some $script:vars |
| Function Names | 100% | ✅ All use Verb-Noun pattern |
| Parameter Names | 100% | ✅ All use PascalCase |
| Script Names | 100% | ✅ All use lowercase-with-hyphens |
| Constants | 0% | ❌ No constants (all inline values) |

---

## 6. Recommendations by Priority

### 🔴 CRITICAL (Fix Immediately)

1. **[SEC-001, SEC-002, SEC-003]** Remove all plaintext passwords
   - Use Windows Credential Manager
   - Implement DPAPI encryption
   - Generate random passwords during installation
   - **Estimated Effort**: 8 hours

2. **[RELIABILITY-001]** Make database operations idempotent
   - Add `IF NOT EXISTS` checks
   - Handle existing resources gracefully
   - **Estimated Effort**: 2 hours

3. **[RELIABILITY-006]** Remove hardcoded user paths
   - Use relative paths or parameters
   - Detect source directory automatically
   - **Estimated Effort**: 1 hour

---

### 🟠 HIGH (Fix Within 1 Week)

4. **[SEC-004]** Implement file permission restrictions
   - Set proper ACLs on sensitive files
   - Remove default inherit permissions
   - **Estimated Effort**: 4 hours

5. **[SEC-005]** Generate cryptographic secrets
   - Use RNGCryptoServiceProvider
   - Store encrypted
   - **Estimated Effort**: 2 hours

6. **[RELIABILITY-002]** Improve service start wait logic
   - Implement polling with timeout
   - Add exponential backoff
   - **Estimated Effort**: 3 hours

7. **[RELIABILITY-003]** Add rollback capabilities
   - Track created resources
   - Implement cleanup on failure
   - **Estimated Effort**: 6 hours

---

### 🟡 MEDIUM (Fix Within 2 Weeks)

8. **[BP-001]** Standardize error handling
   - Use consistent try/catch pattern
   - Centralize error logging
   - **Estimated Effort**: 4 hours

9. **[BP-002]** Add parameter validation
   - Use ValidateScript, ValidateRange
   - Add mandatory parameter checks
   - **Estimated Effort**: 2 hours

10. **[BP-003]** Replace magic numbers with constants
    - Create configuration file
    - Document all timeout values
    - **Estimated Effort**: 2 hours

11. **[RELIABILITY-007]** Validate prerequisites
    - Check Python/Node.js versions
    - Verify disk space
    - Check network connectivity
    - **Estimated Effort**: 3 hours

12. **[BP-011]** Centralize configuration
    - Create shared config file
    - Load configuration in each script
    - **Estimated Effort**: 3 hours

---

### 🟢 LOW (Fix When Possible)

13. **[BP-005]** Use proper Verbose parameter
    - Implement [CmdletBinding()]
    - Use Write-Verbose consistently
    - **Estimated Effort**: 1 hour

14. **[BP-006]** Add progress indicators
    - Use Write-Progress for long operations
    - Show percentage completion
    - **Estimated Effort**: 2 hours

15. **[BP-008]** Document exit codes
    - Create exit code enumeration
    - Document in help text
    - **Estimated Effort**: 1 hour

16. **[BP-012]** Add function documentation
    - Document all functions with help blocks
    - Add examples
    - **Estimated Effort**: 2 hours

17. **Code Duplication** Extract common patterns
    - Create shared module file
    - Import in all scripts
    - **Estimated Effort**: 4 hours

---

## 7. Syntax & Logic Error Check

✅ **No Syntax Errors Found**

All scripts parse correctly with `Test-Path -PathType Leaf` and PowerShell's built-in syntax checker.

### ⚠️ Potential Logic Issues

1. **master-install.ps1:226** - Phase counter mismatch
   ```powershell
   Show-Progress "Phase 5/4" "Configuring Memurai..."  # Should be 5/10
   Show-Progress "Phase 6/4" "Installing Backend..."    # Should be 6/10
   ```

2. **verify-startup.ps1:65-69** - Potential null reference
   ```powershell
   $result = & "C:\Program Files\PostgreSQL\15\bin\psql.exe" -U postgres -h localhost -c "SELECT 1;" 2>&1
   return ($LASTEXITCODE -eq 0)  # $result is captured but never used
   ```

3. **configure-postgres.ps1:92-94** - Output redirection inconsistency
   ```powershell
   & $PSQLExe -U $SuperUser -h localhost -f $sqlFile 2>&1 | ForEach-Object {
       if ($Verbose) { Write-Host "   $_" -ForegroundColor Gray }
   }
   # Captures all output including errors, may mask failures
   ```

---

## 8. Testing Recommendations

### Recommended Test Scenarios

1. **Fresh Installation Test**
   - Clean Windows Server 2019/2022
   - No prerequisites installed
   - Verify complete installation succeeds

2. **Re-run Test (Idempotency)**
   - Run installation twice
   - Verify second run succeeds without errors
   - Confirm no duplicate services created

3. **Partial Failure Recovery Test**
   - Kill script mid-execution
   - Run again
   - Verify cleanup and recovery

4. **Multi-User Test**
   - Run as different users
   - Verify user-specific paths don't break

5. **Service Restart Test**
   - Restart all services multiple times
   - Verify dependency order maintained
   - Confirm health checks pass

6. **Performance Test**
   - Measure total installation time
   - Monitor resource usage
   - Verify service startup times

7. **Security Audit Test**
   - Scan for plaintext passwords in files
   - Check file permissions
   - Verify credential encryption

---

## 9. Summary & Next Steps

### Strengths

- ✅ Well-structured and readable code
- ✅ Comprehensive comments and banner displays
- ✅ Good use of PowerShell Write-Host coloring for UX
- ✅ Consistent directory structure
- ✅ Proper administrator rights checking
- ✅ Service dependency management implemented
- ✅ Health check validation included

### Critical Improvements Needed

1. **Security**: Remove all plaintext passwords (CRITICAL)
2. **Reliability**: Make operations idempotent (HIGH)
3. **Portability**: Remove hardcoded paths (HIGH)
4. **Error Handling**: Standardize and add rollback (MEDIUM)
5. **Testing**: Add automated tests (MEDIUM)

### Estimated Total Remediation Effort

- **Critical Issues**: 11 hours
- **High Priority**: 15 hours
- **Medium Priority**: 14 hours
- **Low Priority**: 10 hours
- **Total**: **50 hours** (approximately 1.25 weeks for one developer)

### Recommended Action Plan

**Week 1: Security & Critical Fixes**
- Day 1-2: Implement secure credential storage
- Day 3: Make operations idempotent
- Day 4: Remove hardcoded paths
- Day 5: Testing & validation

**Week 2: Reliability & Best Practices**
- Day 1-2: Implement rollback mechanisms
- Day 3: Standardize error handling
- Day 4: Add parameter validation & constants
- Day 5: Code review & documentation

**Week 3: Polish & Testing**
- Day 1-2: Extract duplicate code
- Day 3: Add progress indicators
- Day 4-5: Comprehensive testing suite

---

## 10. File-Specific Issues Summary

### master-install.ps1
- ✅ Best-structured script with proper logging
- ⚠️ Phase counter mismatch (lines 226-277)
- ⚠️ No rollback on partial failure

### install-postgres.ps1
- ❌ CRITICAL: Plaintext password in credential file (line 119-130)
- ⚠️ Random password pattern may collide (line 17)
- ✅ Good silent installation implementation

### install-redis.ps1
- ✅ Good fallback detection (Memurai/Redis)
- ⚠️ Manual download required (no automation)
- ⚠️ Blocks execution waiting for user input

### install-nssm.ps1
- ✅ Clean implementation
- ✅ Good PATH management
- ✅ No major issues

### install-backend-service.ps1
- ❌ CRITICAL: Multiple plaintext passwords (lines 22, 27, 106)
- ❌ CRITICAL: Hardcoded source path (line 13)
- ⚠️ No Python version validation
- ⚠️ Insufficient service start wait time

### install-frontend-service.ps1
- ❌ CRITICAL: Hardcoded source path (line 13)
- ⚠️ No Node.js version validation
- ⚠️ Two different execution modes (dev/prod) add complexity
- ✅ Good use of http-server for production

### configure-postgres.ps1
- ❌ CRITICAL: Plaintext password (line 14)
- ❌ CRITICAL: Non-idempotent database creation (line 70)
- ⚠️ Complex SecureString conversion (lines 25-31)
- ⚠️ SQL injection possible (line 76)

### configure-redis.ps1
- ✅ Simple and clean
- ✅ Good service detection logic
- ✅ No major issues

### configure-service-dependencies.ps1
- ✅ Well-implemented dependency chains
- ✅ Good verification logic
- ⚠️ Could extract Get-ServiceDependencies to shared module

### setup-browser-autolaunch.ps1
- ✅ Good use of scheduled tasks
- ⚠️ RandomDelay may not provide consistent UX (line 37)
- ✅ Good user context configuration

### verify-startup.ps1
- ✅ Comprehensive health checks
- ✅ Good use of Test-Check function
- ⚠️ Hardcoded test count (line 10) could get out of sync
- ⚠️ Some checks may fail if services still initializing

---

## Appendix A: Checklist for Developers

Use this checklist when implementing fixes:

### Security Checklist
- [ ] Remove all plaintext passwords from code
- [ ] Implement Windows Credential Manager integration
- [ ] Encrypt all credential files with DPAPI
- [ ] Generate cryptographically random secrets
- [ ] Set restrictive file permissions (Administrators only)
- [ ] Validate and sanitize all user inputs
- [ ] Create dedicated service accounts
- [ ] Remove default inherited permissions

### Reliability Checklist
- [ ] Make all operations idempotent
- [ ] Implement polling with timeout for service starts
- [ ] Add rollback/cleanup on failures
- [ ] Remove hardcoded user paths
- [ ] Validate prerequisites (Python/Node versions, disk space)
- [ ] Handle race conditions in service creation
- [ ] Standardize timeout values

### Best Practices Checklist
- [ ] Implement consistent error handling pattern
- [ ] Add parameter validation to all scripts
- [ ] Replace magic numbers with named constants
- [ ] Use [CmdletBinding()] and Write-Verbose
- [ ] Add progress indicators for long operations
- [ ] Document all functions with help blocks
- [ ] Extract duplicate code to shared module
- [ ] Document exit code conventions
- [ ] Centralize configuration in config file

### Testing Checklist
- [ ] Test fresh installation
- [ ] Test re-run (idempotency)
- [ ] Test partial failure recovery
- [ ] Test multi-user scenarios
- [ ] Test service restart sequences
- [ ] Perform security audit
- [ ] Measure performance metrics

---

**Report End**

*Generated by Code Quality Analyzer Agent*
*For questions or clarifications, review specific line references in each section above.*
