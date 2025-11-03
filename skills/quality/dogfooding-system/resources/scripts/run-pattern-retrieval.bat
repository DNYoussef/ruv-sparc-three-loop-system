@echo off
REM Dogfooding System - Phase 2: Pattern Retrieval
REM Query Memory-MCP for similar past fixes using vector search
REM Usage: run-pattern-retrieval.bat "violation description" [--apply]

setlocal enabledelayedexpansion

REM Set paths
set BASE_DIR=C:\Users\17175
set RETRIEVALS_DIR=%BASE_DIR%\metrics\dogfooding\retrievals
set SCRIPTS_DIR=%BASE_DIR%\scripts
set MEMORY_MCP_DIR=%BASE_DIR%\Desktop\memory-mcp-triple-system

REM Ensure directories exist
if not exist "%RETRIEVALS_DIR%" mkdir "%RETRIEVALS_DIR%"

REM Get violation query (required)
set QUERY=%~1
if "%QUERY%"=="" (
    echo Error: Violation query required.
    echo Usage: run-pattern-retrieval.bat "violation description" [--apply]
    exit /b 1
)

REM Check for --apply flag
set APPLY_FIX=0
if "%2"=="--apply" set APPLY_FIX=1

echo ========================================
echo Dogfooding System - Pattern Retrieval
echo ========================================
echo Query: %QUERY%
echo Apply Fix: %APPLY_FIX%
echo Started: %date% %time%
echo.

REM Generate timestamp for output files
set TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%

REM Run vector search via Memory-MCP
echo [1/3] Executing vector search in Memory-MCP...
node "%SCRIPTS_DIR%\query-memory-mcp.js" "%QUERY%" > "%RETRIEVALS_DIR%\query-%TIMESTAMP%.json"

if %errorlevel% neq 0 (
    echo Error: Vector search failed. Check Memory-MCP server status.
    exit /b 1
)

REM Parse and rank results
echo [2/3] Ranking patterns by similarity + success rate...
node "%SCRIPTS_DIR%\rank-patterns.js" "%RETRIEVALS_DIR%\query-%TIMESTAMP%.json" > "%RETRIEVALS_DIR%\best-pattern-%TIMESTAMP%.json"

REM Display top 5 results
echo.
echo Top 5 Similar Patterns:
echo ========================================
type "%RETRIEVALS_DIR%\best-pattern-%TIMESTAMP%.json"
echo ========================================

REM Optionally apply best pattern
if %APPLY_FIX%==1 (
    echo.
    echo [3/3] Applying best pattern with sandbox testing...

    REM Create sandbox environment
    set SANDBOX_DIR=%BASE_DIR%\tmp\dogfood-sandbox-%TIMESTAMP%
    mkdir "%SANDBOX_DIR%"

    REM Apply transformation
    node "%SCRIPTS_DIR%\apply-fix-pattern.js" "%RETRIEVALS_DIR%\best-pattern-%TIMESTAMP%.json" "%SANDBOX_DIR%"

    if %errorlevel%==0 (
        echo.
        echo ✓ Pattern applied successfully in sandbox
        echo Sandbox: %SANDBOX_DIR%
        echo Run 'npm test' in sandbox to verify before production
    ) else (
        echo.
        echo ✗ Pattern application failed
        rmdir /s /q "%SANDBOX_DIR%"
    )
) else (
    echo.
    echo [3/3] Skipping application (use --apply flag to auto-apply)
)

echo.
echo ========================================
echo Pattern Retrieval Complete
echo ========================================
echo Query: %QUERY%
echo Results: %RETRIEVALS_DIR%\query-%TIMESTAMP%.json
echo Best Pattern: %RETRIEVALS_DIR%\best-pattern-%TIMESTAMP%.json
echo ========================================

endlocal
exit /b 0
