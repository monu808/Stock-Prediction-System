@echo off
REM Windows batch script to setup and run the trading system

echo ========================================
echo Stock Market Prediction System Setup
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

echo Python found, checking version...
python -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)"
if %errorlevel% neq 0 (
    echo ERROR: Python 3.9+ is required
    pause
    exit /b 1
)

echo Python version OK

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Docker is not installed
    echo Some features may not work without Docker
    echo You can continue without Docker for basic functionality
    pause
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo Installing Python dependencies...
pip install -r requirements.txt

REM Create config file if it doesn't exist
if not exist "config\config.yaml" (
    echo Creating configuration file...
    copy "config\config.example.yaml" "config\config.yaml"
    echo Configuration file created. You may want to edit config\config.yaml
)

REM Create logs directory
if not exist "logs" (
    mkdir logs
    echo Created logs directory
)

REM Test the system
echo Testing system...
python test_system.py
if %errorlevel% neq 0 (
    echo System tests failed. Please check the error messages above.
    pause
    exit /b 1
)

echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To start the system:
echo   1. Make sure Docker services are running: docker-compose up -d
echo   2. Run the main application: python main.py
echo   3. Open dashboard: http://localhost:8000/dashboard
echo.
echo Press any key to start Docker services now, or Ctrl+C to exit
pause >nul

REM Start Docker services
echo Starting Docker services...
docker-compose up -d

if %errorlevel% neq 0 (
    echo WARNING: Failed to start Docker services
    echo You can try starting them manually with: docker-compose up -d
    echo Or run the system without Docker (limited functionality)
)

echo.
echo Docker services started. You can now run the main application:
echo   python main.py
echo.
pause