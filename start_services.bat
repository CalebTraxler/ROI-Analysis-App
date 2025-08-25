@echo off
echo Starting ROI Analysis Services...
echo.

echo [1/3] Installing Flask API dependencies...
pip install -r api_requirements.txt

echo.
echo [2/3] Starting Flask Coordinate API...
start "Flask API" cmd /k "python coordinate_api.py"

echo.
echo [3/3] Starting React App...
cd roi-heatmap
start "React App" cmd /k "npm start"

echo.
echo Services are starting up...
echo - Flask API: http://localhost:5000
echo - React App: http://localhost:3000
echo.
echo Press any key to exit this script...
pause > nul
