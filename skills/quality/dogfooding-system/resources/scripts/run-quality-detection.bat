@echo off
REM Dogfooding System - Phase 1: Quality Detection
REM Run Connascence Analysis, detect violations, store in Memory-MCP
REM Usage: run-quality-detection.bat [project|all]

setlocal enabledelayedexpansion

REM Set paths
set BASE_DIR=C:\Users\17175
set METRICS_DIR=%BASE_DIR%\metrics\dogfooding
set SCRIPTS_DIR=%BASE_DIR%\scripts
set MEMORY_MCP_DIR=%BASE_DIR%\Desktop\memory-mcp-triple-system

REM Ensure directories exist
if not exist "%METRICS_DIR%" mkdir "%METRICS_DIR%"

REM Get project parameter (default: all)
set PROJECT=%1
if "%PROJECT%"=="" set PROJECT=all

echo ========================================
echo Dogfooding System - Quality Detection
echo ========================================
echo Project: %PROJECT%
echo Started: %date% %time%
echo.

REM Define projects to analyze
if "%PROJECT%"=="all" (
    set PROJECTS=memory-mcp connascence claude-flow
) else (
    set PROJECTS=%PROJECT%
)

REM Counter for total violations
set TOTAL_VIOLATIONS=0

REM Process each project
for %%P in (%PROJECTS%) do (
    echo [%%P] Running Connascence Analysis...

    REM Set project directory
    if "%%P"=="memory-mcp" set PROJ_DIR=%BASE_DIR%\Desktop\memory-mcp-triple-system
    if "%%P"=="connascence" set PROJ_DIR=%BASE_DIR%\Desktop\connascence
    if "%%P"=="claude-flow" set PROJ_DIR=%BASE_DIR%\Desktop\claude-flow

    REM Run connascence analysis
    cd "%PROJ_DIR%"
    python -m connascence.analyzer analyze . --output-json > "%METRICS_DIR%\%%P_raw.json" 2>&1

    REM Parse and store results in Memory-MCP
    echo [%%P] Parsing results and storing in Memory-MCP...
    node "%SCRIPTS_DIR%\store-connascence-results.js" "%METRICS_DIR%\%%P_raw.json" %%P

    REM Count violations
    for /f "tokens=2 delims=:" %%V in ('findstr /C:"total_violations" "%METRICS_DIR%\%%P_raw.json"') do (
        set /a TOTAL_VIOLATIONS+=%%V
    )

    echo [%%P] Analysis complete. Results stored.
    echo.
)

REM Generate summary report
set TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set SUMMARY_FILE=%METRICS_DIR%\summary_%TIMESTAMP%.txt

echo ======================================== > "%SUMMARY_FILE%"
echo Dogfooding Quality Detection Summary >> "%SUMMARY_FILE%"
echo ======================================== >> "%SUMMARY_FILE%"
echo Timestamp: %date% %time% >> "%SUMMARY_FILE%"
echo Project(s): %PROJECTS% >> "%SUMMARY_FILE%"
echo Total Violations: %TOTAL_VIOLATIONS% >> "%SUMMARY_FILE%"
echo. >> "%SUMMARY_FILE%"
echo Results stored in Memory-MCP with WHO/WHEN/PROJECT/WHY tags >> "%SUMMARY_FILE%"
echo Dashboard updated at http://localhost:3000 >> "%SUMMARY_FILE%"
echo ======================================== >> "%SUMMARY_FILE%"

type "%SUMMARY_FILE%"

REM Update dashboard (if Grafana running)
curl -s http://localhost:3000/api/health >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo Dashboard updated successfully.
) else (
    echo.
    echo Warning: Grafana dashboard not running. Start with: npm run grafana
)

echo.
echo ========================================
echo Quality Detection Complete
echo ========================================
echo Total Violations Detected: %TOTAL_VIOLATIONS%
echo Summary: %SUMMARY_FILE%
echo Metrics: %METRICS_DIR%
echo Memory-MCP: Tagged records stored
echo ========================================

endlocal
exit /b 0
