#!/bin/bash

# Setup script for 911 Emergency Operator project

echo "Setting up 911 Emergency Operator environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data
mkdir -p outputs
mkdir -p logs

# Create sample dataset if it doesn't exist
if [ ! -f "data/911_calls.jsonl" ]; then
    echo "Creating sample dataset..."
    python data_preparation.py
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  python train.py"
echo ""
echo "For inference, run:"
echo "  python inference.py --interactive"
