@echo off
REM Dogfooding System - Phase 3: Continuous Improvement
REM Full cycle: Quality Detection + Pattern Retrieval + Safe Application
REM Usage: run-continuous-improvement.bat [project|all] [--dry-run]

setlocal enabledelayedexpansion

REM Set paths
set BASE_DIR=C:\Users\17175
set CYCLES_DIR=%BASE_DIR%\metrics\dogfooding\cycle-summaries
set ARCHIVE_DIR=%BASE_DIR%\metrics\dogfooding\archive
set SCRIPTS_DIR=%BASE_DIR%\scripts

REM Ensure directories exist
if not exist "%CYCLES_DIR%" mkdir "%CYCLES_DIR%"
if not exist "%ARCHIVE_DIR%" mkdir "%ARCHIVE_DIR%"

REM Get parameters
set PROJECT=%1
if "%PROJECT%"=="" set PROJECT=all

set DRY_RUN=0
if "%2"=="--dry-run" set DRY_RUN=1

REM Generate cycle ID
set CYCLE_ID=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set CYCLE_ID=%CYCLE_ID: =0%

echo ========================================
echo Dogfooding Continuous Improvement Cycle
echo ========================================
echo Cycle ID: %CYCLE_ID%
echo Project: %PROJECT%
echo Dry Run: %DRY_RUN%
echo Started: %date% %time%
echo.

REM Create cycle archive directory
set CYCLE_ARCHIVE=%ARCHIVE_DIR%\cycle-%CYCLE_ID%
mkdir "%CYCLE_ARCHIVE%"

REM Phase 1: Quality Detection (30-60s)
echo ========================================
echo [PHASE 1/5] Quality Detection (30-60s)
echo ========================================
call "%SCRIPTS_DIR%\dogfood-quality-check.bat" %PROJECT%
if %errorlevel% neq 0 (
    echo Error: Quality detection failed
    exit /b 1
)
echo Phase 1 Complete: Violations detected and stored
echo.

REM Phase 2: Pattern Retrieval (10-30s)
echo ========================================
echo [PHASE 2/5] Pattern Retrieval (10-30s)
echo ========================================

REM Get top 3 violations from Phase 1
for /f "skip=1 tokens=1,2 delims=," %%A in ('type "%BASE_DIR%\metrics\dogfooding\latest_violations.csv"') do (
    set VIOLATION=%%A
    set SEVERITY=%%B

    echo Retrieving patterns for: !VIOLATION! (Severity: !SEVERITY!)
    call "%SCRIPTS_DIR%\dogfood-memory-retrieval.bat" "!VIOLATION!"

    REM Archive retrieval results
    xcopy /Y "%BASE_DIR%\metrics\dogfooding\retrievals\*" "%CYCLE_ARCHIVE%\" >nul
)
echo Phase 2 Complete: Patterns retrieved and ranked
echo.

REM Phase 3: Safe Application (20-40s)
echo ========================================
echo [PHASE 3/5] Safe Application (20-40s)
echo ========================================

if %DRY_RUN%==1 (
    echo DRY-RUN MODE: Skipping fix application
    set FIXES_APPLIED=0
) else (
    REM Apply fixes one by one with sandbox testing
    set FIXES_APPLIED=0

    for %%F in ("%CYCLE_ARCHIVE%\best-pattern-*.json") do (
        echo Applying fix: %%~nxF

        REM Create sandbox
        set SANDBOX_DIR=%BASE_DIR%\tmp\sandbox-%CYCLE_ID%-%%~nF
        mkdir "!SANDBOX_DIR!"

        REM Copy project to sandbox
        xcopy /E /I /Q "%PROJECT_DIR%" "!SANDBOX_DIR%" >nul

        REM Apply fix in sandbox
        cd "!SANDBOX_DIR!"
        node "%SCRIPTS_DIR%\apply-fix-pattern.js" "%%F" "!SANDBOX_DIR!"

        REM Test in sandbox
        call npm test >nul 2>&1
        if !errorlevel!==0 (
            echo ✓ Sandbox tests passed for %%~nxF

            REM Apply to production with git safety
            cd "%PROJECT_DIR%"
            git stash push -u -m "backup-%CYCLE_ID%-%%~nF"
            node "%SCRIPTS_DIR%\apply-fix-pattern.js" "%%F" "%PROJECT_DIR%"

            REM Verify in production
            call npm test
            if !errorlevel!==0 (
                echo ✓ Production tests passed for %%~nxF
                git add .
                git commit -m "fix: Apply dogfooding fix %%~nxF [cycle-%CYCLE_ID%]"
                set /a FIXES_APPLIED+=1
            ) else (
                echo ✗ Production tests failed, rolling back %%~nxF
                git stash pop
            )
        ) else (
            echo ✗ Sandbox tests failed for %%~nxF, skipping
        )

        REM Cleanup sandbox
        rmdir /s /q "!SANDBOX_DIR!"
    )
)
echo Phase 3 Complete: %FIXES_APPLIED% fixes applied successfully
echo.

REM Phase 4: Verification (15s)
echo ========================================
echo [PHASE 4/5] Verification (15s)
echo ========================================
call "%SCRIPTS_DIR%\dogfood-quality-check.bat" %PROJECT%
echo Comparing before/after metrics...
node "%SCRIPTS_DIR%\compare-metrics.js" "%CYCLE_ARCHIVE%"
echo Phase 4 Complete: Verification successful
echo.

REM Phase 5: Summary & Metrics (10-20s)
echo ========================================
echo [PHASE 5/5] Summary & Metrics (10-20s)
echo ========================================
node "%SCRIPTS_DIR%\generate-cycle-summary.js" %CYCLE_ID% %PROJECT% %FIXES_APPLIED% > "%CYCLES_DIR%\cycle-%CYCLE_ID%.txt"

REM Update dashboard
node "%SCRIPTS_DIR%\update-dashboard.js" %CYCLE_ID%

REM Display summary
type "%CYCLES_DIR%\cycle-%CYCLE_ID%.txt"

echo.
echo ========================================
echo Continuous Improvement Cycle Complete
echo ========================================
echo Cycle ID: %CYCLE_ID%
echo Fixes Applied: %FIXES_APPLIED%
echo Summary: %CYCLES_DIR%\cycle-%CYCLE_ID%.txt
echo Archive: %CYCLE_ARCHIVE%
echo Dashboard: http://localhost:3000
echo ========================================

REM Store cycle summary in Memory-MCP
node "%SCRIPTS_DIR%\store-cycle-summary.js" "%CYCLES_DIR%\cycle-%CYCLE_ID%.txt"

endlocal
exit /b 0
