#!/usr/bin/env python3
# ============================================================
# face_attendance_system.py  —  Face Recognition Attendance
# 
# A MacBook-optimized face recognition attendance system with
# Supabase integration for cloud-based attendance logging.
#
# SETUP:
#   1. Create .env file from .env.example
#   2. Add your Supabase URL and API key
#   3. pip install -r requirements.txt
#   4. python face_attendance_system.py
#
# KEYBOARD CONTROLS:
#   Q  — quit the program
#   E  — enroll a new student
#   S  — save database
#
# ============================================================

import cv2
import numpy as np
import face_recognition
import time
import csv
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ── SUPABASE CREDENTIALS (Load from .env file) ──────────────────────────
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
USE_SUPABASE = SUPABASE_URL and SUPABASE_KEY

if USE_SUPABASE:
    try:
        from supabase import create_client, Client
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[SUPABASE] ✓ Connected to Supabase successfully")
    except Exception as e:
        print(f"[SUPABASE] ✗ Failed to connect: {e}")
        print("[SUPABASE] Falling back to local CSV logging")
        USE_SUPABASE = False
else:
    print("[SUPABASE] No credentials found. Using local CSV logging.")
    print("[SUPABASE] To enable cloud logging, create .env file with credentials.")

# ── LOCAL CONFIG ─────────────────────────────────────────────────────────
from config import (
    STUDENT_RECORDS_CSV, ATTENDANCE_CSV, SNAPSHOTS_DIR,
    FRAME_SCALE, HOG_UPSAMPLE, TOLERANCE,
    ENROLLMENT_SAMPLE_FRAMES, ATTENDANCE_COOLDOWN_MIN,
    OPENCV_THREADS, RECOGNITION_EVERY_N_FRAMES
)

# Optimize OpenCV for MacBook M1
cv2.setNumThreads(OPENCV_THREADS)

# Display constants
COLOR_RECOGNIZED = (0, 200, 80)    # Green   — recognized student
COLOR_UNKNOWN = (0, 80, 255)       # Orange  — unknown face
COLOR_ENROLLING = (255, 200, 0)    # Cyan    — enrollment mode
FONT = cv2.FONT_HERSHEY_SIMPLEX


class StudentDatabase:
    """Manages student records and face encodings in CSV format."""
    
    def __init__(self):
        ATTENDANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize student_records.csv if needed
        if not STUDENT_RECORDS_CSV.exists():
            with open(STUDENT_RECORDS_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'roll_no', 'section', 'encoding', 'enrolled_at', 'snapshot_path'])
            print(f"[DB] Created {STUDENT_RECORDS_CSV}")
        
        # Initialize attendance.csv if needed
        if not ATTENDANCE_CSV.exists():
            with open(ATTENDANCE_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'roll_no', 'timestamp'])
            print(f"[DB] Created {ATTENDANCE_CSV}")
        
        # Load known student encodings into memory
        self.students = {}  # {student_id: {'name': ..., 'encodings': [...]}}
        self._load_students()
    
    def _load_students(self):
        """Load all student encodings from CSV into memory."""
        if not STUDENT_RECORDS_CSV.exists():
            return
        
        with open(STUDENT_RECORDS_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get('encoding'):
                    continue
                
                student_id = f"{row['roll_no']}_{row['name']}"
                encoding_str = row['encoding']
                
                try:
                    # Parse encoding from CSV (comma-separated 128 values)
                    encoding = np.array([float(x) for x in encoding_str.split(',')])
                    
                    if student_id not in self.students:
                        self.students[student_id] = {
                            'name': row['name'],
                            'roll_no': row['roll_no'],
                            'section': row['section'],
                            'encodings': []
                        }
                    
                    self.students[student_id]['encodings'].append(encoding)
                except ValueError:
                    print(f"[DB] Warning: Could not parse encoding for {row['name']}")
        
        print(f"[DB] Loaded {len(self.students)} students with encodings")
    
    def enroll_student(self, name, roll_no, section, encoding_list, snapshot_path):
        """Add or update a student's encoding."""
        student_id = f"{roll_no}_{name}"
        
        # Average encodings for robustness
        avg_encoding = np.mean(encoding_list, axis=0)
        encoding_str = ','.join([str(float(x)) for x in avg_encoding])
        
        # Append to CSV
        with open(STUDENT_RECORDS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, roll_no, section, encoding_str, datetime.now().isoformat(), snapshot_path])
        
        # Update in-memory database
        if student_id not in self.students:
            self.students[student_id] = {
                'name': name,
                'roll_no': roll_no,
                'section': section,
                'encodings': []
            }
        
        self.students[student_id]['encodings'].extend(encoding_list)
        print(f"[DB] Enrolled {name} ({roll_no}) from {section}")
    
    def recognize_face(self, unknown_encoding):
        """
        Find the closest match for an unknown face encoding.
        Returns: (name, roll_no, distance) or (None, None, float('inf'))
        """
        if not self.students:
            return None, None, float('inf')
        
        best_name = None
        best_roll_no = None
        best_distance = float('inf')
        
        for student_id, student_data in self.students.items():
            encodings = student_data['encodings']
            
            # Compute distances to all known encodings for this student
            distances = face_recognition.face_distance(encodings, unknown_encoding)
            min_distance = float(distances.min())
            
            if min_distance < best_distance:
                best_distance = min_distance
                best_name = student_data['name']
                best_roll_no = student_data['roll_no']
        
        if best_distance < TOLERANCE:
            return best_name, best_roll_no, best_distance
        else:
            return None, None, best_distance


class AttendanceLogger:
    """Logs attendance to Supabase (or CSV as fallback)."""
    
    def __init__(self):
        self._last_seen = {}  # {student_id: timestamp}
        self._cooldown_secs = ATTENDANCE_COOLDOWN_MIN * 60
        
        # Initialize CSV if using local fallback
        if not USE_SUPABASE:
            ATTENDANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
            if not ATTENDANCE_CSV.exists():
                with open(ATTENDANCE_CSV, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['name', 'roll_no', 'department', 'timestamp'])
                print(f"[ATTENDANCE] Created {ATTENDANCE_CSV}")
    
    def log_attendance(self, name, roll_no, department=""):
        """
        Log attendance to Supabase or CSV with cooldown protection.
        Returns: True if logged, False if in cooldown period.
        """
        student_id = f"{roll_no}_{name}"
        now = time.time()
        last_time = self._last_seen.get(student_id, 0)
        
        # Check cooldown period
        if now - last_time < self._cooldown_secs:
            return False  # Still in cooldown
        
        self._last_seen[student_id] = now
        timestamp = datetime.now()
        timestamp_str = timestamp.isoformat()
        
        # Try Supabase first
        if USE_SUPABASE:
            try:
                response = supabase_client.table("attendance").insert({
                    "name": name,
                    "roll_no": roll_no,
                    "department": department,
                    "timestamp": timestamp_str
                }).execute()
                
                print(f"[ATTENDANCE] ✓ {name} ({roll_no}) logged to Supabase at {timestamp.strftime('%H:%M:%S')}")
                return True
                
            except Exception as e:
                print(f"[ATTENDANCE] ✗ Supabase error: {e}")
                print(f"[ATTENDANCE] Falling back to CSV for {name}")
                # Fall back to CSV
                self._log_to_csv(name, roll_no, department, timestamp_str)
                return True
        
        # Use CSV (fallback or if Supabase disabled)
        else:
            self._log_to_csv(name, roll_no, department, timestamp_str)
            return True
    
    def _log_to_csv(self, name, roll_no, department, timestamp_str):
        """Write attendance record to local CSV file."""
        try:
            with open(ATTENDANCE_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([name, roll_no, department, timestamp_str])
            
            timestamp = datetime.fromisoformat(timestamp_str)
            print(f"[ATTENDANCE] ✓ {name} ({roll_no}) logged to CSV at {timestamp.strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"[ATTENDANCE] ✗ CSV write error: {e}")


class FaceRecognitionEngine:
    """Detects and recognizes faces in video frames."""
    
    def __init__(self, db, attendance_logger):
        self.db = db
        self.attendance_logger = attendance_logger
        self._last_results = []
        self.enrollment_mode = False
        self.enrollment_frames = []
        self.enrollment_crops = []  # Store face crops for snapshot
        self.last_frame = None  # Store last processed frame
    
    def process_frame(self, frame_bgr):
        """
        Detect faces and recognize them.
        Returns list of results with bounding boxes and identities.
        """
        # Downscale for faster processing
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Store original frame for enrollment snapshots
        self.last_frame = frame_bgr.copy()
        
        # Detect faces using HOG
        face_locations = face_recognition.face_locations(
            rgb_small,
            number_of_times_to_upsample=HOG_UPSAMPLE,
            model="hog"
        )
        
        if not face_locations:
            self._last_results = []
            return []
        
        # Compute face encodings
        face_encodings = face_recognition.face_encodings(
            rgb_small,
            known_face_locations=face_locations,
            num_jitters=1,
            model="large"
        )
        
        # Scale back to original frame coordinates
        scale_factor = 1.0 / FRAME_SCALE
        face_locations_full = [
            (int(top * scale_factor), int(right * scale_factor),
             int(bottom * scale_factor), int(left * scale_factor))
            for (top, right, bottom, left) in face_locations
        ]
        
        results = []
        
        for location, encoding in zip(face_locations_full, face_encodings):
            top, right, bottom, left = location
            face_crop = frame_bgr[top:bottom, left:right]
            
            if self.enrollment_mode:
                # Capture frames for enrollment
                self.enrollment_frames.append(encoding)
                self.enrollment_crops.append(face_crop.copy())
                name = "ENROLLING..."
                roll_no = None
                distance = 0.0
                is_recognized = False
            else:
                # Recognize face
                name, roll_no, distance = self.db.recognize_face(encoding)
                is_recognized = name is not None
                
                # Log attendance if recognized
                if is_recognized:
                    # Get department from database
                    student_id = f"{roll_no}_{name}"
                    department = self.db.students.get(student_id, {}).get('section', '')
                    self.attendance_logger.log_attendance(name, roll_no, department)
            
            results.append({
                'location': location,
                'name': name if name else 'Unknown',
                'roll_no': roll_no,
                'distance': distance,
                'is_recognized': is_recognized,
                'face_crop': face_crop
            })
        
        self._last_results = results
        return results
    
    def get_last_results(self):
        """Return cached results for display on skipped frames."""
        return self._last_results
    
    def start_enrollment(self):
        """Start enrollment mode."""
        self.enrollment_mode = True
        self.enrollment_frames = []
        print("[ENROLLMENT] Mode activated. Show your face clearly to the camera.")
    
    def finish_enrollment(self):
        """Complete enrollment and save to database."""
        if len(self.enrollment_frames) < 2:
            print("[ENROLLMENT] Not enough frames captured. Try again.")
            self.enrollment_mode = False
            self.enrollment_frames = []
            self.enrollment_crops = []
            return False
        
        # Get student info from terminal
        print("\n" + "="*60)
        print("ENROLLMENT FORM")
        print("="*60)
        name = input("Enter student name: ").strip()
        roll_no = input("Enter roll number: ").strip()
        section = input("Enter section (e.g., A, B, C): ").strip()
        
        if not name or not roll_no or not section:
            print("[ENROLLMENT] Missing information. Enrollment cancelled.")
            self.enrollment_mode = False
            self.enrollment_frames = []
            self.enrollment_crops = []
            return False
        
        # Save best snapshot (middle frame of captured frames)
        snapshot_path = ""
        if self.enrollment_crops:
            best_crop_idx = len(self.enrollment_crops) // 2
            best_crop = self.enrollment_crops[best_crop_idx]
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            snapshot_filename = f"{roll_no}_{name}_{timestamp}.jpg"
            snapshot_path = SNAPSHOTS_DIR / snapshot_filename
            
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snapshot_path), best_crop)
            print(f"[ENROLLMENT] Snapshot saved to {snapshot_path}")
        
        # Enroll student
        self.db.enroll_student(name, roll_no, section, self.enrollment_frames, str(snapshot_path))
        
        self.enrollment_mode = False
        self.enrollment_frames = []
        self.enrollment_crops = []
        print(f"[ENROLLMENT] ✓ Successfully enrolled {name}\n")
        return True


class VideoRenderer:
    """Renders faces with bounding boxes and labels on video frames."""
    
    @staticmethod
    def render(frame, results, fps):
        """Draw bounding boxes, names, and confidence on frame."""
        h, w = frame.shape[:2]
        
        for r in results:
            top, right, bottom, left = r['location']
            
            # Choose color based on recognition status
            if r['is_recognized']:
                color = COLOR_RECOGNIZED
                conf = max(0.0, 1.0 - r['distance'] / TOLERANCE)
                label = f"{r['name']} ({r['roll_no']}) - {conf*100:.0f}%"
            else:
                color = COLOR_UNKNOWN
                label = "Unknown"
            
            # Draw bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw name badge
            (text_w, text_h), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.rectangle(frame, (left, bottom), (left + text_w + 10, bottom + text_h + 8),
                          color, cv2.FILLED)
            cv2.putText(frame, label, (left + 5, bottom + text_h + 4),
                        FONT, 0.55, (10, 10, 10), 1, cv2.LINE_AA)
            
            # Draw confidence bar for recognized faces
            if r['is_recognized']:
                conf = max(0.0, 1.0 - r['distance'] / TOLERANCE)
                bar_width = right - left
                filled_width = int(bar_width * conf)
                bar_y = top - 6
                cv2.rectangle(frame, (left, bar_y - 4), (right, bar_y), (40, 40, 40), -1)
                cv2.rectangle(frame, (left, bar_y - 4), (left + filled_width, bar_y), color, -1)
        
        # Status bar at bottom
        recognized_count = sum(1 for r in results if r['is_recognized'])
        total_count = len(results)
        status_text = f"FPS: {fps:.1f} | Faces: {total_count} | Recognized: {recognized_count} | Q=Quit E=Enroll S=Save"
        
        cv2.rectangle(frame, (0, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, status_text, (10, h - 8), FONT, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
        
        return frame


def main():
    """Main program loop."""
    print("\n" + "="*60)
    print("FACE RECOGNITION ATTENDANCE SYSTEM")
    print("="*60)
    print("Loading database...")
    
    # Initialize components
    db = StudentDatabase()
    attendance_logger = AttendanceLogger()
    engine = FaceRecognitionEngine(db, attendance_logger)
    renderer = VideoRenderer()
    
    # Initialize webcam
    print("[SYSTEM] Initializing camera (this may take a moment)...")
    cap = cv2.VideoCapture(0)
    
    # Try alternative camera indices if default doesn't work
    if not cap.isOpened():
        print("[SYSTEM] Camera 0 failed, trying camera 1...")
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print("[ERROR] On macOS, you may need to:")
        print("  1. Grant camera permission to Terminal:")
        print("     System Settings → Privacy & Security → Camera → Terminal ✓")
        print("  2. Try running in a different terminal app (iTerm2, etc.)")
        print("  3. Restart your Mac")
        return
    
    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("[SYSTEM] ✓ Camera initialized successfully")
    print("[SYSTEM] Starting recognition...")
    print("[SYSTEM] Press 'Q' to quit, 'E' to enroll, 'S' to save")
    print("[SYSTEM] Window should appear on screen shortly...\n")
    
    frame_count = 0
    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()
            
            # Process frame
            if frame_count % RECOGNITION_EVERY_N_FRAMES == 0:
                results = engine.process_frame(frame)
            else:
                results = engine.get_last_results()
            
            # Render
            display_frame = renderer.render(frame.copy(), results, fps_display)
            
            # Create window and display frame
            window_name = "Face Recognition Attendance System"
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_name, display_frame)
            
            # Force window to front on macOS
            import sys
            if sys.platform == 'darwin':  # macOS
                import subprocess
                try:
                    subprocess.run(['osascript', '-e', 'tell app "Python" to activate'], 
                                   capture_output=True, timeout=0.1)
                except:
                    pass
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n[SYSTEM] Shutting down...")
                break
            elif key == ord('e') or key == ord('E'):
                if not engine.enrollment_mode:
                    engine.start_enrollment()
                else:
                    engine.finish_enrollment()
            elif key == ord('s') or key == ord('S'):
                print("[SYSTEM] Database saved.")
    
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[SYSTEM] Session ended. Attendance recorded in", ATTENDANCE_CSV)


if __name__ == "__main__":
    main()
