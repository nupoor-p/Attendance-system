#!/bin/bash
# ============================================================
# startup.sh - Start all components of FaceAttend Pro
# ============================================================

echo "🚀 Starting FaceAttend Pro..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down FaceAttend Pro..."
    pkill -P $$ npm
    pkill -P $$ python3.11
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if Python 3.11 is installed
if ! command -v /opt/homebrew/bin/python3.11 &> /dev/null; then
    echo "❌ Python 3.11 is not installed. Please install Python 3.11 first."
    exit 1
fi

# Start Node.js UI server
echo "[1/3] Starting Node.js UI Server..."
npm start > /tmp/nodejs_ui.log 2>&1 &
NODE_PID=$!
sleep 2

# Check if Node.js server started successfully
if ! kill -0 $NODE_PID 2>/dev/null; then
    echo "❌ Failed to start Node.js UI server"
    echo "Check /tmp/nodejs_ui.log for details"
    exit 1
fi
echo "✓ Node.js UI Server running (PID: $NODE_PID)"

# Start Flask API server
echo "[2/3] Starting Flask API Server..."
/opt/homebrew/bin/python3.11 flask_server.py > /tmp/flask_server.log 2>&1 &
FLASK_PID=$!
sleep 2

# Check if Flask server started successfully
if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo "❌ Failed to start Flask server"
    echo "Check /tmp/flask_server.log for details"
    kill $NODE_PID
    exit 1
fi
echo "✓ Flask API Server running (PID: $FLASK_PID)"

# Start Python face recognition engine
echo "[3/3] Starting Face Recognition Engine..."
/opt/homebrew/bin/python3.11 main.py > /tmp/face_recognition.log 2>&1 &
PYTHON_PID=$!
sleep 3

# Check if Python process started successfully
if ! kill -0 $PYTHON_PID 2>/dev/null; then
    echo "❌ Failed to start face recognition engine"
    echo "Check /tmp/face_recognition.log for details"
    kill $NODE_PID
    kill $FLASK_PID
    exit 1
fi
echo "✓ Face Recognition Engine running (PID: $PYTHON_PID)"

echo ""
echo "================================"
echo "✅ FaceAttend Pro is running!"
echo "================================"
echo ""
echo "🌐 Open your browser and go to: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for all processes
wait $NODE_PID $FLASK_PID $PYTHON_PID
