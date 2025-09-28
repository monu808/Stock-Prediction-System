#!/bin/bash
# Linux/macOS setup script for the trading system

set -e  # Exit on any error

echo "========================================"
echo "Stock Market Prediction System Setup"
echo "========================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.9+ and try again"
    exit 1
fi

echo "Python found, checking version..."
python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" || {
    echo "ERROR: Python 3.9+ is required"
    exit 1
}

echo "Python version OK"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "WARNING: Docker is not installed"
    echo "Some features may not work without Docker"
    echo "You can continue without Docker for basic functionality"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create config file if it doesn't exist
if [ ! -f "config/config.yaml" ]; then
    echo "Creating configuration file..."
    cp config/config.example.yaml config/config.yaml
    echo "Configuration file created. You may want to edit config/config.yaml"
fi

# Create logs directory
if [ ! -d "logs" ]; then
    mkdir -p logs
    echo "Created logs directory"
fi

# Test the system
echo "Testing system..."
python test_system.py || {
    echo "System tests failed. Please check the error messages above."
    exit 1
}

echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "To start the system:"
echo "  1. Make sure Docker services are running: docker-compose up -d"
echo "  2. Activate virtual environment: source venv/bin/activate"
echo "  3. Run the main application: python main.py"
echo "  4. Open dashboard: http://localhost:8000/dashboard"
echo

# Ask if user wants to start Docker services
read -p "Start Docker services now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting Docker services..."
    docker-compose up -d || {
        echo "WARNING: Failed to start Docker services"
        echo "You can try starting them manually with: docker-compose up -d"
        echo "Or run the system without Docker (limited functionality)"
    }
    
    echo
    echo "Docker services started. You can now run the main application:"
    echo "  source venv/bin/activate"
    echo "  python main.py"
fi

echo "Setup complete!"