#!/bin/bash

echo "Setting up Fake News Detector..."
echo

echo "Step 1: Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment. Please ensure Python 3.8+ is installed."
    exit 1
fi

echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo "Step 3: Upgrading pip..."
pip install --upgrade pip

echo "Step 4: Installing dependencies..."
echo "This may take 5-10 minutes depending on your internet connection..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Please check your internet connection."
    exit 1
fi

echo
echo "Setup complete!"
echo
echo "To start the application, run one of these commands:"
echo "  streamlit run app/simple_modern_app.py    (Recommended for beginners)"
echo "  streamlit run app/modern_app.py           (Advanced features)"
echo "  streamlit run app/app.py                  (Original version)"
echo
echo "The application will open in your web browser at http://localhost:8501"
echo
