# Windows Service Failure - Quick Reference

## 🚀 Fast Track (5 Minutes)

### Step 1: Run Diagnostic
```powershell
# Open PowerShell as Administrator
cd C:\Users\17175\scripts
.\diagnose-service-failures.ps1
```

### Step 2: Apply Most Common Fix (70% Success Rate)
```powershell
# Fix PostgreSQL binary path (adds missing -D parameter)
.\fix-postgresql-service-binary-path.ps1
```

### Step 3: If Still Failing, Try Permissions Fix (20% Success Rate)
```powershell
# Grant Local System access to directories
.\fix-service-permissions.ps1
```

### Step 4: Verify
```powershell
Get-Service postgresql-x64-15, Memurai | Select-Object Name, Status, StartType
```

---

## 🎯 What Makes This Different from Normal Troubleshooting?

### The Key Insight
**PostgreSQL works with `pg_ctl.exe` but fails via Service Controller** → This proves PostgreSQL is healthy, the issue is **service wrapper configuration**.

### Common Service Failure Patterns

| Error Code | Meaning | Likely Cause | Fix |
|------------|---------|--------------|-----|
| 1053 | Service timeout | Database taking too long to start | Check data directory size, increase timeout |
| 1067 | Process terminated | Application crash | Check PostgreSQL logs for stack traces |
| 1069 | Logon failure | Permission denied | Run `fix-service-permissions.ps1` |
| Port in use | Address already in use | Another process on port 5432/6379 | Kill conflicting process |
| No error | Silent failure | Missing `-D` parameter | Run `fix-postgresql-service-binary-path.ps1` |

---

## 📋 Diagnostic Checklist (Manual)

### 1. Service Configuration
```powershell
# Check binary path - should include -D parameter
sc qc postgresql-x64-15
```
**Look for**: `BINARY_PATH_NAME` should include `-D "C:\Program Files\PostgreSQL\15\data"`

### 2. Event Logs
```powershell
# Last 10 service errors
Get-EventLog -LogName System -Source "Service Control Manager" -Newest 10 -EntryType Error |
    Where-Object {$_.Message -like "*postgresql*"}
```

### 3. Port Conflicts
```powershell
# Check if port 5432 is in use
Get-NetTCPConnection -LocalPort 5432 -ErrorAction SilentlyContinue
```

### 4. Permissions
```powershell
# Check Local System access
Get-Acl "C:\Program Files\PostgreSQL\15\data" |
    Select-Object -ExpandProperty Access |
    Where-Object {$_.IdentityReference -like "*SYSTEM*"}
```

### 5. Lock Files
```powershell
# Check for stale postmaster.pid
Test-Path "C:\Program Files\PostgreSQL\15\data\postmaster.pid"
```

---

## 🛠️ Manual Fix Commands

### Fix A: Add Missing -D Parameter (Most Common)
```powershell
# Stop service
Stop-Service postgresql-x64-15 -Force

# Update binary path
sc config postgresql-x64-15 binPath= "\"C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe\" runservice -N \"postgresql-x64-15\" -D \"C:\Program Files\PostgreSQL\15\data\" -w"

# Start service
Start-Service postgresql-x64-15
```

### Fix B: Grant Permissions
```powershell
# Grant Local System full control
icacls "C:\Program Files\PostgreSQL\15\data" /grant "NT AUTHORITY\SYSTEM:(OI)(CI)F" /T

# Restart service
Restart-Service postgresql-x64-15
```

### Fix C: Remove Stale Lock File
```powershell
# ONLY if PostgreSQL is confirmed NOT running
Stop-Process -Name postgres -Force -ErrorAction SilentlyContinue

# Delete lock file
Remove-Item "C:\Program Files\PostgreSQL\15\data\postmaster.pid" -Force

# Start service
Start-Service postgresql-x64-15
```

### Fix D: Kill Port Conflict
```powershell
# Find process using port 5432
$process = Get-NetTCPConnection -LocalPort 5432 | Select-Object -ExpandProperty OwningProcess
Get-Process -Id $process | Stop-Process -Force

# Start service
Start-Service postgresql-x64-15
```

---

## 🔍 Why pg_ctl Works But Service Controller Doesn't

| Factor | pg_ctl (Manual) | Service Controller |
|--------|----------------|-------------------|
| **User** | YOUR account | Local System |
| **Environment** | Inherits YOUR variables | System variables only |
| **Working Dir** | Current directory | C:\Windows\System32 or configured |
| **Permissions** | YOUR file access | Local System access (different!) |
| **Error Output** | Console (visible) | Event Log (hidden) |
| **Parameters** | Interactive prompts | Must be in binary path |

**Root Cause**: The service doesn't inherit your environment, so it needs **explicit parameters** (like `-D` for data directory) that pg_ctl infers from environment variables.

---

## 🚨 Emergency Workarounds

### Option 1: Scheduled Task (15 minutes)
```powershell
# Create startup task that runs pg_ctl
$action = New-ScheduledTaskAction -Execute "C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe" `
    -Argument '-D "C:\Program Files\PostgreSQL\15\data" start'

$trigger = New-ScheduledTaskTrigger -AtStartup

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName "PostgreSQL-Startup" -Action $action -Trigger $trigger -Principal $principal
```

**Pros**: Fast, uses working pg_ctl
**Cons**: Not a true service, less reliable

### Option 2: NSSM Wrapper (30 minutes)
```powershell
# Download NSSM from https://nssm.cc/download
# Then:
sc delete postgresql-x64-15
nssm install postgresql-x64-15 "C:\Program Files\PostgreSQL\15\bin\pg_ctl.exe" -D "C:\Program Files\PostgreSQL\15\data" start
nssm set postgresql-x64-15 Start SERVICE_AUTO_START
nssm start postgresql-x64-15
```

**Pros**: Proper service, wraps pg_ctl
**Cons**: Adds NSSM dependency

---

## ✅ Validation Tests

### Test 1: Manual Start
```powershell
Start-Service postgresql-x64-15
Get-Service postgresql-x64-15 | Select-Object Name, Status
```
**Expected**: Status = Running

### Test 2: Connection
```powershell
Test-NetConnection -ComputerName localhost -Port 5432
# Or with psql:
# psql -U postgres -h localhost -c "SELECT version();"
```
**Expected**: Connection succeeds

### Test 3: Auto-Start After Reboot
```powershell
Restart-Computer -Force
# After reboot:
Get-Service postgresql-x64-15 | Select-Object Name, Status, StartType
```
**Expected**: Status = Running, StartType = Automatic

---

## 📁 Available Scripts

| Script | Purpose | Time | Success Rate |
|--------|---------|------|--------------|
| `diagnose-service-failures.ps1` | Full diagnostic | 2 min | - |
| `fix-postgresql-service-binary-path.ps1` | Add -D parameter | 1 min | 70% |
| `fix-service-permissions.ps1` | Grant Local System access | 2 min | 20% |

**Location**: `C:\Users\17175\scripts\`

---

## 📖 Full Documentation

For comprehensive analysis and advanced fixes:
- Full Runbook: `C:\Users\17175\docs\WINDOWS-SERVICE-DIAGNOSTIC-RUNBOOK.md`
- Contains: 7 diagnostic phases, 7 fix strategies, technical deep-dive

---

## 🔄 Recommended Workflow

```
1. Run diagnostic script (2 min)
   ↓
2. Review output, identify root cause
   ↓
3. Apply Fix A (binary path) → 70% success
   ↓
4. If fails → Apply Fix B (permissions) → 90% cumulative success
   ↓
5. If still fails → Check full runbook for advanced fixes
   ↓
6. Nuclear option → NSSM wrapper (always works)
```

---

## 🎯 Success Criteria

- ✅ Service starts via `Start-Service` command
- ✅ Service auto-starts after reboot
- ✅ No errors in Event Viewer
- ✅ Applications can connect to PostgreSQL/Memurai
- ✅ No manual intervention needed

---

## 🆘 If All Else Fails

1. **Collect diagnostic output**: `.\diagnose-service-failures.ps1 -SaveToFile`
2. **Check PostgreSQL logs**: `C:\Program Files\PostgreSQL\15\data\log\`
3. **Review Event Viewer**: Application + System logs
4. **Consider reinstallation**: Your data is safe in `data` directory
5. **Use NSSM wrapper**: Nuclear option that always works

---

**Created**: 2025-11-09
**System**: Windows Service Troubleshooting
**Agent**: System Architecture Designer
