#!/bin/bash
# ============================================================
# run.sh — Quick launcher for Face Recognition Attendance
# ============================================================

echo "=================================================="
echo "Face Recognition Attendance System"
echo "=================================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    exit 1
fi

# Change to script directory
cd "$(dirname "$0")" || exit

# Check if required packages are installed
python3 -c "import cv2, face_recognition, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not installed. Running setup..."
    ./setup.sh
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

echo "Starting attendance system..."
python3 face_attendance_system.py
