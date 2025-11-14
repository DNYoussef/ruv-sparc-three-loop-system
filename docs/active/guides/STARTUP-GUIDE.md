# Ruv-Sparc Dashboard - Startup Automation Guide

## Overview

This guide explains how to set up automatic startup for the Ruv-Sparc Dashboard on Windows. The system automatically:

1. Checks and starts Docker Desktop if needed
2. Waits for Docker daemon to be ready
3. Starts all Docker Compose services
4. Performs health checks on the API
5. Opens the dashboard in your browser
6. Logs all operations with error handling

## Quick Start

### 1. Test the Startup Script

Before scheduling, test that the startup script works correctly:

```powershell
# Run the test script
cd C:\Users\17175\scripts\startup
.\test-startup.ps1
```

If successful, you'll see:
- ✓ All pre-flight checks pass
- ✓ Docker Desktop starts (if needed)
- ✓ Services start successfully
- ✓ Health check passes
- ✓ Dashboard opens in browser

### 2. Set Up Task Scheduler

**Run PowerShell as Administrator** and execute:

```powershell
cd C:\Users\17175\scripts\startup
.\setup-task-scheduler.ps1
```

This creates a Windows Task Scheduler task that:
- Triggers at system startup (with 1-minute delay)
- Runs with highest privileges
- Allows running on battery power
- Restarts on failure (up to 3 times)
- Has 1-hour execution timeout

### 3. Verify Installation

1. Open Task Scheduler (`taskschd.msc`)
2. Navigate to **Task Scheduler Library**
3. Find **"Ruv-Sparc Dashboard Auto-Start"**
4. Check that:
   - State is "Ready"
   - Next Run Time shows "At startup"
   - Last Run Result is 0x0 (success)

### 4. Test with Reboot

Restart Windows to verify the dashboard opens automatically:

```powershell
# Restart Windows
Restart-Computer
```

After reboot:
- Docker Desktop should start automatically
- Dashboard should open at http://localhost:3000
- Check logs at `C:\Users\17175\logs\startup\`

## File Structure

```
C:\Users\17175\
├── scripts\startup\
│   ├── startup-master.ps1          # Main startup automation script
│   ├── setup-task-scheduler.ps1    # Task Scheduler installation
│   ├── test-startup.ps1            # Test script (no scheduling)
│   └── uninstall-task.ps1          # Remove scheduled task
└── logs\startup\
    ├── startup-master_*.log        # Main execution logs
    ├── startup-errors_*.log        # Error-only logs
    └── docker-logs_*.log           # Docker service logs
```

## Scripts Reference

### startup-master.ps1

**Main startup automation script**

**Parameters:**
- `ProjectPath` - Path to dashboard project (default: `C:\Users\17175\ruv-sparc-ui-dashboard`)
- `LogPath` - Log directory (default: `C:\Users\17175\logs\startup`)
- `HealthCheckTimeout` - Max wait for health check in seconds (default: 60)
- `HealthCheckInterval` - Health check interval in seconds (default: 5)

**Exit Codes:**
- `0` - Success (all steps completed, health check passed)
- `1` - Warning (services started but health check failed)
- `2` - Failure (startup failed)

**Example Usage:**
```powershell
# Run with defaults
.\startup-master.ps1

# Custom parameters
.\startup-master.ps1 -ProjectPath "D:\projects\dashboard" -HealthCheckTimeout 120
```

### setup-task-scheduler.ps1

**Creates Windows Task Scheduler task**

**Requirements:** Administrator privileges

**Parameters:**
- `TaskName` - Name of scheduled task (default: "Ruv-Sparc Dashboard Auto-Start")
- `ScriptPath` - Path to startup-master.ps1
- `LogPath` - Log directory
- `Force` - Skip confirmation prompts

**Example Usage:**
```powershell
# Interactive setup
.\setup-task-scheduler.ps1

# Force reinstall (no prompts)
.\setup-task-scheduler.ps1 -Force

# Custom task name
.\setup-task-scheduler.ps1 -TaskName "My Custom Dashboard"
```

### test-startup.ps1

**Test script without scheduling**

**Parameters:**
- `Verbose` - Enable verbose output
- `SkipDocker` - Skip Docker checks (for testing)
- `SkipBrowser` - Skip browser launch (for testing)

**Example Usage:**
```powershell
# Basic test
.\test-startup.ps1

# Verbose test
.\test-startup.ps1 -Verbose

# Test without launching browser
.\test-startup.ps1 -SkipBrowser
```

### uninstall-task.ps1

**Remove scheduled task**

**Requirements:** Administrator privileges

**Parameters:**
- `TaskName` - Name of task to remove
- `Force` - Skip confirmation prompt

**Example Usage:**
```powershell
# Interactive removal
.\uninstall-task.ps1

# Force removal (no prompt)
.\uninstall-task.ps1 -Force
```

## Task Scheduler Management

### View Task Status

```powershell
# Get task information
Get-ScheduledTask -TaskName "Ruv-Sparc Dashboard Auto-Start"

# Get last run result
Get-ScheduledTaskInfo -TaskName "Ruv-Sparc Dashboard Auto-Start"
```

### Manual Task Control

```powershell
# Run task manually
Start-ScheduledTask -TaskName "Ruv-Sparc Dashboard Auto-Start"

# Disable task (stop auto-start)
Disable-ScheduledTask -TaskName "Ruv-Sparc Dashboard Auto-Start"

# Enable task (resume auto-start)
Enable-ScheduledTask -TaskName "Ruv-Sparc Dashboard Auto-Start"

# Remove task
Unregister-ScheduledTask -TaskName "Ruv-Sparc Dashboard Auto-Start"
```

## Log Files

### startup-master_*.log

Main execution log with timestamped entries:

```
[2025-11-08 09:00:00] [INFO] Ruv-Sparc Dashboard Auto-Start
[2025-11-08 09:00:01] [INFO] STEP 1: Docker Desktop Check
[2025-11-08 09:00:02] [SUCCESS] Docker Desktop already running (PID: 12345)
[2025-11-08 09:00:03] [SUCCESS] Docker daemon is ready
[2025-11-08 09:00:05] [SUCCESS] Docker services started successfully
[2025-11-08 09:00:10] [SUCCESS] Health check passed! Service is healthy.
[2025-11-08 09:00:11] [SUCCESS] Dashboard opened successfully
```

### startup-errors_*.log

Error-only log for troubleshooting:

```
[2025-11-08 09:00:15] [ERROR] Health check timeout after 60s. Service did not become healthy.
[2025-11-08 09:00:16] [ERROR] Failed to start Docker services: Container failed to start
```

### docker-logs_*.log

Docker service logs (captured on errors):

```
backend-1    | INFO:     Application startup complete.
frontend-1   | Compiled successfully!
database-1   | PostgreSQL init process complete; ready for start up.
```

## Troubleshooting

### Dashboard doesn't open on startup

1. Check Task Scheduler:
   ```powershell
   Get-ScheduledTaskInfo -TaskName "Ruv-Sparc Dashboard Auto-Start"
   ```

2. Check last run result (should be 0x0):
   - Open Task Scheduler GUI (`taskschd.msc`)
   - Find the task
   - Check "Last Run Result" column

3. Review logs:
   ```powershell
   # View latest log
   Get-Content C:\Users\17175\logs\startup\startup-master_*.log -Tail 50

   # View errors
   Get-Content C:\Users\17175\logs\startup\startup-errors_*.log
   ```

### Docker Desktop doesn't start

**Symptoms:** "Docker Desktop not running" in logs

**Solutions:**

1. Verify Docker Desktop installation:
   ```powershell
   Test-Path "C:\Program Files\Docker\Docker\Docker Desktop.exe"
   ```

2. Check Docker service:
   ```powershell
   Get-Service com.docker.service
   ```

3. Manually start Docker Desktop and wait 30 seconds

4. Check Windows Event Viewer for Docker errors

### Health check timeout

**Symptoms:** Exit code 1, "Health check timeout" in logs

**Solutions:**

1. Increase timeout in startup-master.ps1:
   ```powershell
   .\startup-master.ps1 -HealthCheckTimeout 120
   ```

2. Check service logs:
   ```powershell
   cd C:\Users\17175\ruv-sparc-ui-dashboard
   docker-compose -f docker-compose.prod.yml logs
   ```

3. Verify API endpoint:
   ```powershell
   Invoke-WebRequest -Uri "http://localhost:8000/api/v1/health"
   ```

4. Check for port conflicts:
   ```powershell
   netstat -ano | findstr :8000
   netstat -ano | findstr :3000
   ```

### Permission denied errors

**Symptoms:** "Access is denied" in error logs

**Solutions:**

1. Run setup-task-scheduler.ps1 as Administrator

2. Check script execution policy:
   ```powershell
   Get-ExecutionPolicy
   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. Verify task runs with highest privileges:
   - Open Task Scheduler
   - Right-click task → Properties
   - General tab → "Run with highest privileges" should be checked

### Task doesn't trigger on startup

**Symptoms:** Task exists but doesn't run

**Solutions:**

1. Check trigger configuration:
   ```powershell
   (Get-ScheduledTask -TaskName "Ruv-Sparc Dashboard Auto-Start").Triggers
   ```

2. Verify task is enabled:
   ```powershell
   Enable-ScheduledTask -TaskName "Ruv-Sparc Dashboard Auto-Start"
   ```

3. Check if "Start when available" is enabled:
   - Task Scheduler → Task Properties
   - Settings tab → "Run task as soon as possible after a scheduled start is missed"

4. Increase startup delay (may need more time):
   - Edit task trigger to add 2-minute delay instead of 1 minute

## Advanced Configuration

### Custom Project Path

If your project is in a different location:

```powershell
# Edit startup-master.ps1 parameter or pass at runtime
.\startup-master.ps1 -ProjectPath "D:\my-projects\dashboard"

# Update Task Scheduler task
.\setup-task-scheduler.ps1 -Force -ScriptPath "D:\my-scripts\startup-master.ps1"
```

### Different Health Check Endpoint

Edit `startup-master.ps1` and modify the health check URL:

```powershell
# Line ~260 in startup-master.ps1
Test-ServiceHealth -Url "http://localhost:8000/api/v1/custom-health"
```

### Disable Browser Auto-Open

Edit `startup-master.ps1` and comment out Step 5:

```powershell
# Step 5: Open dashboard
# Write-Log "STEP 5: Open Dashboard" -Level "INFO"
# Open-Dashboard -Url "http://localhost:3000"
```

### Multiple Dashboards

Create separate task for each dashboard:

```powershell
# Dashboard 1
.\setup-task-scheduler.ps1 `
  -TaskName "Dashboard 1 Auto-Start" `
  -ScriptPath "C:\scripts\dashboard1-startup.ps1"

# Dashboard 2
.\setup-task-scheduler.ps1 `
  -TaskName "Dashboard 2 Auto-Start" `
  -ScriptPath "C:\scripts\dashboard2-startup.ps1"
```

## Performance Tuning

### Reduce Startup Delay

Default: 1-minute delay after system startup

To reduce (may cause issues if system not ready):

```powershell
# Edit task trigger manually in Task Scheduler
# Or modify setup-task-scheduler.ps1:
$trigger.Delay = "PT30S"  # 30 seconds
```

### Increase Health Check Timeout

For slower systems:

```powershell
.\startup-master.ps1 -HealthCheckTimeout 180 -HealthCheckInterval 10
```

### Parallel Service Startup

Docker Compose already starts services in parallel. To optimize further, reduce service dependencies in `docker-compose.prod.yml`.

## Security Considerations

### Script Execution Policy

The setup uses `-ExecutionPolicy Bypass` to run scripts. This is safe for this use case but:

- Scripts are run with highest privileges
- Only run trusted scripts
- Review scripts before scheduling

### Credential Management

Startup scripts do NOT store credentials. Docker and services use:
- Environment variables (`.env` files)
- Docker secrets
- Windows Credential Manager

Never hardcode credentials in startup scripts!

### Network Security

The startup script requires network access for:
- Docker daemon communication
- Health check HTTP requests
- Docker image pulling (if needed)

Ensure firewall allows:
- Docker Desktop processes
- Localhost connections (127.0.0.1)
- Health check endpoint (port 8000)

## Backup and Recovery

### Backup Task Configuration

```powershell
# Export task to XML
Export-ScheduledTask -TaskName "Ruv-Sparc Dashboard Auto-Start" -TaskPath "\" `
  | Out-File "C:\backup\dashboard-task.xml"

# Import task from XML
Register-ScheduledTask -Xml (Get-Content "C:\backup\dashboard-task.xml" | Out-String) `
  -TaskName "Ruv-Sparc Dashboard Auto-Start"
```

### Backup Logs

Logs are timestamped and accumulate over time. To archive:

```powershell
# Archive logs older than 30 days
$archivePath = "C:\Users\17175\logs\startup\archive"
New-Item -ItemType Directory -Path $archivePath -Force

Get-ChildItem "C:\Users\17175\logs\startup\*.log" |
  Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
  Move-Item -Destination $archivePath
```

### Disaster Recovery

If startup automation breaks:

1. **Remove scheduled task:**
   ```powershell
   .\uninstall-task.ps1 -Force
   ```

2. **Start services manually:**
   ```powershell
   cd C:\Users\17175\ruv-sparc-ui-dashboard
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Review logs for root cause:**
   ```powershell
   Get-Content C:\Users\17175\logs\startup\startup-errors_*.log
   ```

4. **Fix issues and reinstall:**
   ```powershell
   .\test-startup.ps1  # Verify fix
   .\setup-task-scheduler.ps1  # Reinstall
   ```

## Monitoring and Alerts

### Email Notifications

To add email alerts on failure, edit `startup-master.ps1` and add:

```powershell
function Send-EmailAlert {
    param([string]$Subject, [string]$Body)

    Send-MailMessage `
        -From "dashboard@example.com" `
        -To "admin@example.com" `
        -Subject $Subject `
        -Body $Body `
        -SmtpServer "smtp.gmail.com" `
        -Port 587 `
        -UseSsl `
        -Credential (Get-Credential)
}

# Call in catch block:
Send-EmailAlert -Subject "Dashboard Startup Failed" -Body $_
```

### Windows Event Log

To log to Windows Event Log:

```powershell
# Create event source (run as Admin, one-time)
New-EventLog -LogName Application -Source "RuvSparcDashboard"

# Write to event log
Write-EventLog -LogName Application -Source "RuvSparcDashboard" `
  -EventId 1000 -EntryType Information -Message "Dashboard started"
```

### Health Check Monitoring

To monitor health continuously (not just at startup):

```powershell
# Create scheduled task that runs every 5 minutes
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-File C:\scripts\health-monitor.ps1"

Register-ScheduledTask -TaskName "Dashboard Health Monitor" `
  -Trigger $trigger -Action $action
```

## FAQ

### Q: Can I delay the browser opening?

**A:** Yes, add a sleep before `Open-Dashboard`:

```powershell
Start-Sleep -Seconds 30
Open-Dashboard -Url "http://localhost:3000"
```

### Q: How do I run multiple instances?

**A:** Modify `docker-compose.prod.yml` to use different ports, then create separate startup scripts for each instance.

### Q: Can this work on other Windows versions?

**A:** Yes, supports:
- Windows 10 (all editions)
- Windows 11
- Windows Server 2016+

### Q: Does this work with WSL2?

**A:** Yes, Docker Desktop on Windows with WSL2 backend is fully supported.

### Q: How do I change the health check URL?

**A:** Pass parameter to `Test-ServiceHealth` in `startup-master.ps1`:

```powershell
Test-ServiceHealth -Url "http://localhost:8080/health"
```

### Q: Can I disable notifications?

**A:** Comment out `Show-Notification` calls in `startup-master.ps1`.

### Q: What if Docker Desktop crashes?

**A:** The task has auto-restart configured (3 attempts). If Docker keeps crashing, check Docker Desktop logs and Windows Event Viewer.

## Support

For issues or questions:

1. Check logs: `C:\Users\17175\logs\startup\`
2. Review Task Scheduler history
3. Test manually: `.\test-startup.ps1 -Verbose`
4. Check Docker status: `docker info`
5. Verify project files: `docker-compose config`

## Version History

- **v1.0** (2025-11-08) - Initial release with full automation
  - Docker Desktop management
  - Health check validation
  - Browser auto-launch
  - Error logging and notifications
  - Task Scheduler integration

---

**Related Documentation:**
- [Docker Production Setup](../ruv-sparc-ui-dashboard/DOCKER-PRODUCTION.md)
- [Loop 1 Requirements (FR4.1-FR4.6)](../docs/THREE-LOOP-V2-COMPLETION-REPORT.md)
- [Deployment Guide](../ruv-sparc-ui-dashboard/DEPLOYMENT.md)
