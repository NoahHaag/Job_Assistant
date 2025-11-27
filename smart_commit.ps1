# Smart Commit Script
# Automatically pulls before committing tracker files

param(
    [string]$Message = "Update tracker data"
)

$ErrorActionPreference = "Stop"

Write-Host "[INFO] Checking for tracker file changes..." -ForegroundColor Cyan

# Check if tracker files are modified
$trackerFiles = git status --porcelain | Select-String -Pattern '(cold_emails\.json|job_applications\.json)'

if ($trackerFiles) {
    Write-Host "[PULL] Tracker files modified. Pulling latest changes first..." -ForegroundColor Yellow
    
    # Stash any uncommitted changes first
    $stashNeeded = $false
    git diff --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[STASH] Stashing uncommitted changes..." -ForegroundColor Cyan
        git stash push -m "Auto-stash before sync"
        $stashNeeded = $true
    }
    
    # Pull latest
    git pull --rebase origin main
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Pull failed. Please resolve conflicts manually." -ForegroundColor Red
        exit 1
    }
    
    # Restore stashed changes
    if ($stashNeeded) {
        Write-Host "[STASH] Restoring stashed changes..." -ForegroundColor Cyan
        git stash pop
    }
    
    Write-Host "[SUCCESS] Synced with remote." -ForegroundColor Green
}

# Stage all changes
git add .

# Check if there are any changes to commit
git diff --staged --quiet

if ($LASTEXITCODE -ne 0) {
    # There are changes - commit and push
    Write-Host "[COMMIT] Committing changes..." -ForegroundColor Cyan
    git commit -m $Message
    
    Write-Host "[PUSH] Pushing to GitHub..." -ForegroundColor Cyan
    git push origin main
    
    Write-Host "[SUCCESS] All done!" -ForegroundColor Green
}
else {
    Write-Host "[INFO] No changes to commit" -ForegroundColor Yellow
}
