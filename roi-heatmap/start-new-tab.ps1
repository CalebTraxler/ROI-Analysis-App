Write-Host "Starting React app and opening in new tab..." -ForegroundColor Green

# Set environment variables to force new tab
$env:BROWSER = "chrome"
$env:BROWSER_ARGS = "--new-window"

# Start the React app
npm start
