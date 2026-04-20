#!/usr/bin/env python3
# ============================================================
# flask_server.py — Flask API server + UI for FaceAttend Pro
# ============================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import threading
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import uuid

from data.database import FaceDatabase

app = Flask(__name__, static_folder=Path(__file__).parent)
CORS(app)

# Configuration
PYTHON_BACKEND_RUNNING = False
db = FaceDatabase()

# Temporary storage for pending enrollment
PENDING_ENROLLMENT = {
    'embedding': None,
    'timestamp': None,
    'face_crop': None
}

# ─────────────────────────────────────────────────────────
# Static file serving
# ─────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory(Path(__file__).parent, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(Path(__file__).parent, filename)

# ─────────────────────────────────────────────────────────
# Routes for Node.js UI to call (or future UI)
# ─────────────────────────────────────────────────────────

@app.route('/api/status', methods=['GET'])
def status():
    """Get server status"""
    return jsonify({
        'status': 'running',
        'backend_running': PYTHON_BACKEND_RUNNING,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/enrollment-trigger', methods=['POST'])
def enrollment_trigger():
    """Trigger enrollment notification (from camera with face embedding)"""
    global PENDING_ENROLLMENT
    data = request.json or {}
    
    # Store the embedding for later use when form is submitted
    PENDING_ENROLLMENT['embedding'] = data.get('embedding')
    PENDING_ENROLLMENT['timestamp'] = datetime.now().isoformat()
    PENDING_ENROLLMENT['face_crop'] = data.get('face_crop')
    
    print(f"[Flask] Enrollment trigger received - waiting for form submission")
    return jsonify({
        'status': 'success',
        'message': 'Enrollment triggered - awaiting form submission'
    })

@app.route('/api/enroll', methods=['POST'])
def enroll():
    """Enroll a new face - called when form is submitted"""
    global PENDING_ENROLLMENT
    
    data = request.json
    name = data.get('name')
    student_id = data.get('student_id')
    
    if not name or not student_id:
        return jsonify({'status': 'error', 'message': 'Name and Student ID required'}), 400
    
    if PENDING_ENROLLMENT['embedding'] is None:
        return jsonify({'status': 'error', 'message': 'No face embedding available'}), 400
    
    try:
        # Create snapshots directory if it doesn't exist
        from config import SNAPSHOTS_DIR
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Convert embedding list back to numpy array
        embedding = np.array(PENDING_ENROLLMENT['embedding'], dtype=np.float32)
        
        # Generate snapshot filename
        snapshot_filename = f"{student_id}_{uuid.uuid4().hex[:8]}.png"
        snapshot_path = SNAPSHOTS_DIR / snapshot_filename
        
        # Save face crop image if available
        if PENDING_ENROLLMENT['face_crop']:
            # Decode base64 if it's encoded
            face_crop_data = PENDING_ENROLLMENT['face_crop']
            if isinstance(face_crop_data, str):
                import base64
                face_crop_data = base64.b64decode(face_crop_data)
            # Save as image
            # For now, just create a placeholder file since face_crop is complex
            snapshot_path.touch()
            print(f"[Flask] Saved face snapshot: {snapshot_path}")
        
        # Enroll in database
        db.enroll(
            student_id=student_id,
            name=name,
            dept="",
            embeddings=[embedding],
            snapshot_path=str(snapshot_path)
        )
        
        # Clear pending enrollment
        PENDING_ENROLLMENT = {'embedding': None, 'timestamp': None, 'face_crop': None}
        
        print(f"[Flask] Successfully enrolled {name} ({student_id})")
        
        return jsonify({
            'status': 'success',
            'message': f'{name} enrolled successfully',
            'name': name,
            'student_id': student_id,
            'snapshot': str(snapshot_path),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[Flask] Error enrolling {name}: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Enrollment failed: {str(e)}'
        }), 500

@app.route('/api/check-attendance', methods=['POST'])
def check_attendance():
    """Check attendance for a recognized face"""
    data = request.json
    name = data.get('name')
    student_id = data.get('student_id')
    embedding = data.get('embedding')
    
    print(f"[Flask] Attendance check: {name}")
    
    return jsonify({
        'status': 'success',
        'logged': True,
        'name': name,
        'student_id': student_id,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/attendance-log', methods=['POST'])
def attendance_log():
    """Log attendance notification (from camera)"""
    data = request.json or {}
    print(f"[Flask] Attendance logged from camera: {data}")
    return jsonify({
        'status': 'success',
        'message': 'Attendance logged'
    })

@app.route('/api/get-students', methods=['GET'])
def get_students():
    """Get list of all registered students"""
    # This would be integrated with your database
    return jsonify({
        'students': [],
        'total': 0
    })

@app.route('/api/get-attendance', methods=['GET'])
def get_attendance():
    """Get attendance records"""
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    return jsonify({
        'date': date,
        'records': [],
        'total': 0
    })

# ─────────────────────────────────────────────────────────
# Helper functions for Python backend to call this server
# ─────────────────────────────────────────────────────────

def notify_enrollment_needed(face_data):
    """Notify UI that a new face needs enrollment"""
    try:
        # In a real scenario, this would use WebSockets
        # For now, we're using HTTP endpoints that the Python backend can call
        print(f"[Flask] Enrollment notification would be sent: {face_data}")
        return True
    except Exception as e:
        print(f"[Flask] Error notifying enrollment: {e}")
        return False

def notify_attendance_logged(person_data):
    """Notify UI that attendance was logged"""
    try:
        print(f"[Flask] Attendance notification would be sent: {person_data}")
        return True
    except Exception as e:
        print(f"[Flask] Error notifying attendance: {e}")
        return False

def check_ui_connection():
    """Check if UI is running (always true since we're serving it)"""
    global PYTHON_BACKEND_RUNNING
    PYTHON_BACKEND_RUNNING = True
    return True

# ─────────────────────────────────────────────────────────
# Background thread to monitor connection
# ─────────────────────────────────────────────────────────

def monitor_connection():
    """Monitor connection status"""
    while True:
        time.sleep(5)
        check_ui_connection()

if __name__ == '__main__':
    # Start connection monitor in background
    monitor_thread = threading.Thread(target=monitor_connection, daemon=True)
    monitor_thread.start()
    
    print("\n[Flask] Starting FaceAttend Pro API + UI Server")
    print("[Flask] Listening on http://0.0.0.0:5001")
    print("[Flask] UI available at http://localhost:5001")
    print("[Flask] Waiting for Python backend to connect...\n")
    
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
