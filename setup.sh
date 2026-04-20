#!/bin/bash
# ============================================================
# setup.sh — Setup script for Face Recognition Attendance
# ============================================================

echo "=================================================="
echo "Face Recognition Attendance System Setup"
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create data directory
mkdir -p data/snapshots
echo "✓ Created data directories"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed."
    exit 1
fi

# Install requirements
echo ""
echo "Installing dependencies (this may take a few minutes)..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ Setup complete!"
    echo "=================================================="
    echo ""
    echo "To start the attendance system, run:"
    echo "  python3 face_attendance_system.py"
    echo ""
    echo "Keyboard controls:"
    echo "  Q — Quit the program"
    echo "  E — Enroll a new student"
    echo "  S — Save database"
    echo ""
else
    echo "❌ Failed to install dependencies."
    exit 1
fi
