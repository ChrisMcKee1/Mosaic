# Cosmos DB Cleanup PowerShell Script
# Wrapper for cosmos_cleanup.py with Windows-friendly execution

param(
    [switch]$Confirm,
    [switch]$Force,
    [switch]$Backup,
    [switch]$NoBackup,
    [switch]$Test,
    [switch]$Help
)

# Script configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonScript = Join-Path $ScriptDir "cosmos_cleanup.py"
$TestScript = Join-Path $ScriptDir "tests\test_cosmos_cleanup.py"

function Show-Help {
    Write-Host "Cosmos DB Cleanup Script" -ForegroundColor Cyan
    Write-Host "========================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\cleanup_cosmos.ps1 -Confirm [-Backup]     # Safe interactive mode"
    Write-Host "  .\cleanup_cosmos.ps1 -Force [-NoBackup]     # Automated mode"
    Write-Host "  .\cleanup_cosmos.ps1 -Test                  # Run tests"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "  -Confirm    Run with confirmation prompts (safe mode)"
    Write-Host "  -Force      Skip all confirmations (for CI/CD)"
    Write-Host "  -Backup     Create backup before cleanup (default)"
    Write-Host "  -NoBackup   Skip backup creation"
    Write-Host "  -Test       Run unit tests"
    Write-Host "  -Help       Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Green
    Write-Host "  .\cleanup_cosmos.ps1 -Confirm -Backup"
    Write-Host "  .\cleanup_cosmos.ps1 -Force -NoBackup"
    Write-Host "  .\cleanup_cosmos.ps1 -Test"
    Write-Host ""
    Write-Host "TASK: CRUD-000 - Clean Cosmos DB for Fresh Testing Environment" -ForegroundColor Magenta
}

# Main execution logic
if ($Help) {
    Show-Help
    exit 0
}

if ($Test) {
    Write-Host "🧪 Running validation tests..." -ForegroundColor Cyan
    python "validate_cleanup.py"
    exit $LASTEXITCODE
}

# Validate parameters for cleanup
if (-not $Confirm -and -not $Force) {
    Write-Host "❌ Error: Must specify either -Confirm or -Force" -ForegroundColor Red
    Write-Host "Use -Help for usage information" -ForegroundColor Yellow
    exit 1
}

if ($Backup -and $NoBackup) {
    Write-Host "❌ Error: Cannot specify both -Backup and -NoBackup" -ForegroundColor Red
    exit 1
}

# Check if Python script exists
if (-not (Test-Path $PythonScript)) {
    Write-Host "❌ Cleanup script not found: $PythonScript" -ForegroundColor Red
    exit 1
}

# Build arguments for Python script
$pythonArgs = @()

if ($Confirm) {
    $pythonArgs += "--confirm"
}

if ($Force) {
    $pythonArgs += "--force"
}

if ($Backup) {
    $pythonArgs += "--backup"
}

if ($NoBackup) {
    $pythonArgs += "--no-backup"
}

# Execute cleanup
Write-Host "🚀 Starting Cosmos DB cleanup..." -ForegroundColor Cyan
Write-Host "Script: $PythonScript" -ForegroundColor Gray
Write-Host "Arguments: $($pythonArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

try {
    & python $PythonScript @pythonArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Cleanup completed successfully!" -ForegroundColor Green
        Write-Host "📊 Cosmos DB is ready for fresh CRUD testing" -ForegroundColor Green
    }
    else {
        Write-Host ""
        Write-Host "❌ Cleanup failed with exit code: $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
catch {
    Write-Host "❌ Error executing cleanup: $_" -ForegroundColor Red
    exit 1
}
