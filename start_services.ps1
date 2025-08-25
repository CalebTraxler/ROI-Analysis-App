Write-Host "Starting ROI Analysis Services..." -ForegroundColor Green
Write-Host ""

Write-Host "[1/3] Installing Flask API dependencies..." -ForegroundColor Yellow
pip install -r api_requirements.txt

Write-Host ""
Write-Host "[2/3] Starting Flask Coordinate API..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python coordinate_api.py" -WindowStyle Normal

Write-Host ""
Write-Host "[3/3] Starting React App..." -ForegroundColor Yellow
Set-Location roi-heatmap
Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm start" -WindowStyle Normal

Write-Host ""
Write-Host "Services are starting up..." -ForegroundColor Green
Write-Host "- Flask API: http://localhost:5000" -ForegroundColor Cyan
Write-Host "- React App: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit this script..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
