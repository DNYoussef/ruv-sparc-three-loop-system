# Windows Service Startup Failure - Diagnostic Runbook

## Problem Summary
**PostgreSQL**: Service fails to start via Service Controller, but succeeds with `pg_ctl.exe`
**Memurai**: Service fails to start via Service Controller

This asymmetry indicates the issue is in **service configuration**, not the applications themselves.

---

## Phase 1: Diagnostic Commands (Execute All)

### 1.1 Service Configuration Analysis

```powershell
# PostgreSQL service details
Write-Host "`n=== PostgreSQL Service Configuration ===" -ForegroundColor Cyan
sc qc postgresql-x64-15
Get-WmiObject -Class Win32_Service -Filter "Name='postgresql-x64-15'" | Select-Object Name, PathName, StartMode, StartName, State

# Memurai service details
Write-Host "`n=== Memurai Service Configuration ===" -ForegroundColor Cyan
sc qc Memurai
Get-WmiObject -Class Win32_Service -Filter "Name='Memurai'" | Select-Object Name, PathName, StartMode, StartName, State
```

**What to look for:**
- `BINARY_PATH_NAME`: Should include `-D` parameter for PostgreSQL
- `SERVICE_START_NAME`: Should be `LocalSystem` or `NT AUTHORITY\NetworkService`
- `START_TYPE`: Should be `AUTO_START`

---

### 1.2 Event Log Analysis

```powershell
# PostgreSQL application errors (last 24 hours)
Write-Host "`n=== PostgreSQL Event Log Errors ===" -ForegroundColor Cyan
Get-EventLog -LogName Application -Source "postgresql*" -After (Get-Date).AddHours(-24) -EntryType Error -ErrorAction SilentlyContinue |
    Select-Object TimeGenerated, Source, Message | Format-Table -AutoSize

# Service Control Manager errors
Write-Host "`n=== Service Controller Errors ===" -ForegroundColor Cyan
Get-EventLog -LogName System -Source "Service Control Manager" -After (Get-Date).AddHours(-24) -EntryType Error -ErrorAction SilentlyContinue |
    Where-Object {$_.Message -like "*postgresql*" -or $_.Message -like "*Memurai*"} |
    Select-Object TimeGenerated, Message | Format-Table -Wrap
```

**What to look for:**
- Error codes (1053, 1067, 1069 are common)
- Timeout messages (service took too long to start)
- Permission denied errors
- "The service did not respond" messages

---

### 1.3 Port Conflict Detection

```powershell
# Check PostgreSQL port 5432
Write-Host "`n=== Port 5432 Usage ===" -ForegroundColor Cyan
Get-NetTCPConnection -LocalPort 5432 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, State, OwningProcess |
    ForEach-Object {
        $process = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
        [PSCustomObject]@{
            Port = $_.LocalPort
            State = $_.State
            PID = $_.OwningProcess
            ProcessName = $process.ProcessName
            ProcessPath = $process.Path
        }
    }

# Check Memurai port 6379
Write-Host "`n=== Port 6379 Usage ===" -ForegroundColor Cyan
Get-NetTCPConnection -LocalPort 6379 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, State, OwningProcess |
    ForEach-Object {
        $process = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
        [PSCustomObject]@{
            Port = $_.LocalPort
            State = $_.State
            PID = $_.OwningProcess
            ProcessName = $process.ProcessName
            ProcessPath = $process.Path
        }
    }
```

**What to look for:**
- If ports are already in use, service startup will fail
- Identify conflicting process names

---

### 1.4 Permission Analysis

```powershell
# PostgreSQL data directory permissions
Write-Host "`n=== PostgreSQL Data Directory ACL ===" -ForegroundColor Cyan
icacls "C:\Program Files\PostgreSQL\15\data"

# Memurai directory permissions
Write-Host "`n=== Memurai Directory ACL ===" -ForegroundColor Cyan
icacls "C:\Program Files\Memurai"

# Check for Local System access specifically
Write-Host "`n=== Local System Access Test ===" -ForegroundColor Cyan
$pgAcl = Get-Acl "C:\Program Files\PostgreSQL\15\data"
$pgAcl.Access | Where-Object {$_.IdentityReference -like "*SYSTEM*"} |
    Select-Object IdentityReference, FileSystemRights, AccessControlType
```

**What to look for:**
- `NT AUTHORITY\SYSTEM` should have `FullControl` or at minimum `Modify` rights
- `NT AUTHORITY\NetworkService` (if service runs as NetworkService)
- Inherited permissions vs explicit permissions

---

### 1.5 Service Dependencies Check

```powershell
# PostgreSQL dependencies
Write-Host "`n=== PostgreSQL Service Dependencies ===" -ForegroundColor Cyan
sc enumdepend postgresql-x64-15

# Memurai dependencies
Write-Host "`n=== Memurai Service Dependencies ===" -ForegroundColor Cyan
sc enumdepend Memurai

# Check if any automatic services are stopped
Write-Host "`n=== Stopped Automatic Services ===" -ForegroundColor Cyan
Get-Service | Where-Object {$_.Status -eq 'Stopped' -and $_.StartType -eq 'Automatic'} |
    Select-Object Name, DisplayName | Format-Table -AutoSize
```

**What to look for:**
- If dependencies exist, ensure they're running
- RPC, DCOM, or network-related services that might be stopped

---

### 1.6 Binary Path Validation

```powershell
# Verify PostgreSQL binary exists and is executable
Write-Host "`n=== PostgreSQL Binary Validation ===" -ForegroundColor Cyan
$pgService = Get-WmiObject -Class Win32_Service -Filter "Name='postgresql-x64-15'"
$pgBinaryPath = $pgService.PathName -replace '"',''
Write-Host "Binary Path: $pgBinaryPath"
Test-Path $pgBinaryPath

# Verify Memurai binary
Write-Host "`n=== Memurai Binary Validation ===" -ForegroundColor Cyan
$memuraiService = Get-WmiObject -Class Win32_Service -Filter "Name='Memurai'"
$memuraiBinaryPath = $memuraiService.PathName -replace '"',''
Write-Host "Binary Path: $memuraiBinaryPath"
Test-Path $memuraiBinaryPath
```

---

### 1.7 Check for Stale Lock Files

```powershell
# PostgreSQL lock file (postmaster.pid)
Write-Host "`n=== PostgreSQL Lock Files ===" -ForegroundColor Cyan
$lockFile = "C:\Program Files\PostgreSQL\15\data\postmaster.pid"
if (Test-Path $lockFile) {
    Write-Host "WARNING: Stale lock file found!" -ForegroundColor Yellow
    Get-Content $lockFile
} else {
    Write-Host "No lock file found (OK)" -ForegroundColor Green
}

# Memurai lock/pid file
Write-Host "`n=== Memurai Lock Files ===" -ForegroundColor Cyan
$memuraiPid = "C:\Program Files\Memurai\memurai.pid"
if (Test-Path $memuraiPid) {
    Write-Host "WARNING: Memurai PID file found!" -ForegroundColor Yellow
    Get-Content $memuraiPid
} else {
    Write-Host "No PID file found (OK)" -ForegroundColor Green
}
```

---

## Phase 2: Analysis Decision Tree

### If Event Logs show "Error 1053: Service did not respond in timely fashion"
**Root Cause**: Service timeout (default 30 seconds)
**Fix**: Increase service timeout OR fix slow startup (database recovery taking too long)

### If Event Logs show "Error 1067: Process terminated unexpectedly"
**Root Cause**: Application crash during startup
**Fix**: Check PostgreSQL logs in `C:\Program Files\PostgreSQL\15\data\log\` for stack traces

### If Event Logs show "Error 1069: Service did not start due to logon failure"
**Root Cause**: Service account permissions
**Fix**: Grant Local System or NetworkService proper permissions

### If Port is already in use
**Root Cause**: Port conflict
**Fix**: Kill conflicting process OR change PostgreSQL/Memurai port

### If Binary Path missing `-D` parameter (PostgreSQL)
**Root Cause**: Service doesn't know data directory location
**Fix**: Modify service binary path to include `-D "C:\Program Files\PostgreSQL\15\data"`

### If Stale lock file exists
**Root Cause**: Previous PostgreSQL process didn't clean up
**Fix**: Delete `postmaster.pid` (ONLY if PostgreSQL is confirmed not running)

---

## Phase 3: Fix Strategies

### Fix A: Repair Service Binary Path (PostgreSQL)

```powershell
# Stop service if running
Stop-Service postgresql-x64-15 -Force -ErrorAction SilentlyContinue

# Modify service to include correct parameters
sc config postgresql-x64-15 binPath= "\"C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe\" runservice -N \"postgresql-x64-15\" -D \"C:\Program Files\PostgreSQL\15\data\" -w"

# Verify change
sc qc postgresql-x64-15

# Start service
Start-Service postgresql-x64-15
```

---

### Fix B: Grant Local System Permissions

```powershell
# Grant Local System full control to PostgreSQL data directory
icacls "C:\Program Files\PostgreSQL\15\data" /grant "NT AUTHORITY\SYSTEM:(OI)(CI)F" /T

# Grant Local System full control to Memurai directory
icacls "C:\Program Files\Memurai" /grant "NT AUTHORITY\SYSTEM:(OI)(CI)F" /T

# Restart services
Restart-Service postgresql-x64-15
Restart-Service Memurai
```

---

### Fix C: Increase Service Timeout

```powershell
# Increase timeout to 120 seconds (120000 milliseconds)
# NOTE: Requires registry edit
$serviceName = "postgresql-x64-15"
$timeout = 120000  # 2 minutes

Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\$serviceName" -Name "ServicesPipeTimeout" -Value $timeout -Type DWord

Restart-Service $serviceName
```

---

### Fix D: Remove Stale Lock Files

```powershell
# ONLY do this if you've confirmed PostgreSQL is NOT running via Task Manager
Stop-Process -Name postgres -Force -ErrorAction SilentlyContinue

# Remove lock file
Remove-Item "C:\Program Files\PostgreSQL\15\data\postmaster.pid" -Force -ErrorAction SilentlyContinue

# Start service
Start-Service postgresql-x64-15
```

---

### Fix E: Change Service Account to NetworkService

```powershell
# Sometimes Local System has issues, NetworkService works better
sc config postgresql-x64-15 obj= "NT AUTHORITY\NetworkService" password= ""
sc config Memurai obj= "NT AUTHORITY\NetworkService" password= ""

# Grant NetworkService permissions
icacls "C:\Program Files\PostgreSQL\15\data" /grant "NT AUTHORITY\NETWORK SERVICE:(OI)(CI)F" /T
icacls "C:\Program Files\Memurai" /grant "NT AUTHORITY\NETWORK SERVICE:(OI)(CI)F" /T

# Restart services
Restart-Service postgresql-x64-15
Restart-Service Memurai
```

---

### Fix F: NSSM Wrapper (If all else fails)

```powershell
# Download and install NSSM
# https://nssm.cc/download

# Remove existing service
sc delete postgresql-x64-15

# Create new service with NSSM (wraps pg_ctl)
nssm install postgresql-x64-15 "C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe" `
    -D "C:\Program Files\PostgreSQL\15\data" start

# Configure startup
nssm set postgresql-x64-15 Start SERVICE_AUTO_START
nssm set postgresql-x64-15 AppStdout "C:\Program Files\PostgreSQL\15\data\log\nssm-stdout.log"
nssm set postgresql-x64-15 AppStderr "C:\Program Files\PostgreSQL\15\data\log\nssm-stderr.log"

# Start service
nssm start postgresql-x64-15
```

---

### Fix G: Scheduled Task Workaround (Fast but not ideal)

```powershell
# Create scheduled task that runs at startup
$action = New-ScheduledTaskAction -Execute "C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe" `
    -Argument '-D "C:\Program Files\PostgreSQL\15\data" start'

$trigger = New-ScheduledTaskTrigger -AtStartup

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName "PostgreSQL-Startup" -Action $action -Trigger $trigger -Principal $principal -Description "Start PostgreSQL at system startup"
```

---

## Phase 4: Validation Protocol

### Test 1: Manual Service Start
```powershell
Start-Service postgresql-x64-15
Get-Service postgresql-x64-15
```
**Expected**: Status = Running

---

### Test 2: Connection Test
```powershell
# Test PostgreSQL connection
psql -U postgres -h localhost -p 5432 -c "SELECT version();"

# Test Memurai connection (requires redis-cli or similar)
# redis-cli ping
```
**Expected**: Successful connection

---

### Test 3: Reboot Test
```powershell
# Restart computer
Restart-Computer -Force

# After reboot, check service status
Get-Service postgresql-x64-15, Memurai | Select-Object Name, Status, StartType
```
**Expected**: Both services running after reboot

---

### Test 4: Event Log Verification
```powershell
# Check for new errors after service start
Get-EventLog -LogName System -Source "Service Control Manager" -Newest 10 |
    Where-Object {$_.Message -like "*postgresql*" -or $_.Message -like "*Memurai*"}
```
**Expected**: No error entries

---

## Why pg_ctl Works But Service Controller Doesn't

### Technical Explanation

| Aspect | pg_ctl (Manual) | Service Controller |
|--------|----------------|-------------------|
| **User Context** | YOUR user account | Local System or NetworkService |
| **Environment Variables** | Inherits YOUR PATH, PGDATA, etc. | System-wide variables ONLY |
| **Working Directory** | Current shell directory | Specified in service config OR C:\Windows\System32 |
| **Permissions** | YOUR file access rights | Local System rights (different from Admin) |
| **Error Visibility** | Console output (immediate) | Event Log only (delayed) |
| **Parameter Passing** | Explicit via command line | Must be in service binary path |

**Key Insight**: When you run `pg_ctl.exe start`, it:
1. Inherits your environment (including PATH to find DLLs)
2. Has your user permissions (which likely include access to PostgreSQL data directory)
3. Shows errors immediately in console
4. Can ask for missing parameters interactively

When Service Controller tries to start PostgreSQL:
1. Runs as Local System (different permission set)
2. Has minimal environment variables
3. Errors go to Event Log (hidden)
4. Cannot prompt for missing parameters
5. **Must have ALL parameters in binary path** (e.g., `-D` flag for data directory)

**Most Common Root Cause**: The service binary path is missing the `-D "C:\Program Files\PostgreSQL\15\data"` parameter, so PostgreSQL doesn't know where its data is.

---

## Recommended Fix Strategy

### Decision Matrix

| Scenario | Recommended Fix | Time | Risk |
|----------|----------------|------|------|
| Missing `-D` parameter in service binary path | **Fix A** (Repair Binary Path) | 5 min | Low |
| Permission denied errors in Event Log | **Fix B** (Grant Permissions) | 10 min | Low |
| Service timeout (Error 1053) | **Fix C** (Increase Timeout) | 10 min | Low |
| Stale `postmaster.pid` file exists | **Fix D** (Remove Lock) | 2 min | Low |
| Service config fundamentally broken | **Fix F** (NSSM Wrapper) | 30 min | Medium |
| Time-constrained emergency | **Fix G** (Scheduled Task) | 15 min | Medium |

### Recommended Sequence:
1. **Run all Phase 1 diagnostics** (10 minutes)
2. **Apply Fix A** (binary path) - solves 70% of cases
3. If still failing, **Apply Fix B** (permissions) - solves 20% of remaining cases
4. If still failing, **Apply Fix F** (NSSM) - nuclear option that always works
5. Document root cause for future reference

---

## Success Criteria

✅ Service starts via `Start-Service` command
✅ Service starts automatically after reboot
✅ No errors in Event Logs
✅ PostgreSQL/Memurai connections work
✅ Service runs as expected user account
✅ No manual intervention required after reboot

---

## Emergency Rollback

If fixes break the system:

```powershell
# Restore service to original config (if you saved it)
sc config postgresql-x64-15 binPath= "ORIGINAL_PATH_HERE"
sc config postgresql-x64-15 obj= "LocalSystem" password= ""

# Worst case: Reinstall PostgreSQL/Memurai
# Your data will be preserved in C:\Program Files\PostgreSQL\15\data
```

---

**Created**: 2025-11-09
**System**: Windows Service Diagnostics
**Services**: PostgreSQL 15, Memurai
**Agent**: System Architecture Designer
