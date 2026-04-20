# ============================================================
# main.py  —  FaceAttend Pro  |  Smart Self-Evolving Attendance
# MacBook M1 optimized  |  HOG detection + ResNet-128d embeddings
# Author: GEHU BCA Project  |  Inspired by Prashant Chauhan's FaceAttend Pro
#
# HOW TO RUN:
#   conda activate attendance
#   python main.py
#
# KEYBOARD SHORTCUTS (while the camera window is active):
#   Q  — quit
#   E  — force enrollment for next detected face (for manual enroll sessions)
#   S  — save database snapshot now
# ============================================================

import cv2
import numpy as np
import face_recognition   # pip install face-recognition
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path

from config import (
    FRAME_SCALE, HOG_UPSAMPLE, RECOGNITION_THRESHOLD,
    OPENCV_THREADS, RECOGNITION_EVERY_N_FRAMES,
    ENROLL_SAMPLE_FRAMES, ENROLL_TRIGGER_SECONDS,
    SNAPSHOTS_DIR
)
from data.database import FaceDatabase
from attendance import AttendanceLogger
import requests

# ── HTTP client for communicating with Node.js UI via Flask server ────────
UI_SERVER_URL = 'http://localhost:5001'

def notify_ui_enrollment_needed(face_data):
    """Notify UI that a new face needs enrollment"""
    try:
        response = requests.post(
            f'{UI_SERVER_URL}/api/enrollment-trigger',
            json=face_data,
            timeout=5
        )
        print(f"[Main] Enrollment notification sent to UI")
        return True
    except Exception as e:
        print(f"[Main] Error notifying UI about enrollment: {e}")
        return False

def notify_ui_attendance_logged(person_data):
    """Notify UI that attendance was logged"""
    try:
        response = requests.post(
            f'{UI_SERVER_URL}/api/attendance-log',
            json=person_data,
            timeout=5
        )
        print(f"[Main] Attendance logged notified to UI")
        return True
    except Exception as e:
        print(f"[Main] Error notifying UI about attendance: {e}")
        return False

# ── M1 Optimisation: tune OpenCV thread pool ──────────────────────────────
# M1 has 4 performance cores.  More threads = more memory-bus contention
# on the unified memory architecture = slower, not faster.
cv2.setNumThreads(OPENCV_THREADS)

# ── Display constants ─────────────────────────────────────────────────────
COLOR_KNOWN    = (0,  200,  80)    # Green  — recognised student
COLOR_UNKNOWN  = (0,   80, 255)    # Orange — stranger detected
COLOR_ENROLLING = (255, 200, 0)   # Cyan   — capturing enrollment frames
FONT           = cv2.FONT_HERSHEY_SIMPLEX


# ═══════════════════════════════════════════════════════════════════════════
# RECOGNITION ENGINE
# Runs inside the main video loop.  Stateful: caches last-known identities
# so we can render boxes on skipped frames too.
# ═══════════════════════════════════════════════════════════════════════════

class RecognitionEngine:
    """
    Encapsulates the per-frame face detection + recognition pipeline.

    PIPELINE (runs every RECOGNITION_EVERY_N_FRAMES frames):
      1. Downscale frame  →  fast HOG detection
      2. Compute 128-d ResNet embedding for each detected face
      3. Compare each embedding against all known embeddings in DB
         using vectorised L2 distance (numpy + Apple Accelerate BLAS)
      4. Classify:  distance < threshold  →  known student
                    distance ≥ threshold  →  stranger
      5. Track strangers: if same stranger visible for > ENROLL_TRIGGER_SECONDS
         → trigger auto-enrollment GUI
    """

    def __init__(self, db: FaceDatabase, attendance: AttendanceLogger, session: str):
        self.db         = db
        self.attendance = attendance
        self.session    = session

        # Cache: last recognised faces (so overlays stay on skipped frames)
        self._last_results: list[dict] = []

        # Stranger tracking:  face_position_key → (first_seen_time, embedding)
        # We key by a rounded bounding-box region so we don't re-trigger
        # if the face moves a few pixels between frames.
        self._stranger_tracker: dict[str, dict] = {}

        # Flag set when GUI is open — prevents duplicate windows
        self._enrollment_active = False

        # Force-enroll flag (set by pressing 'E')
        self.force_enroll = False

    # ── Core recognition step ─────────────────────────────────────────────

    def process_frame(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Detect + identify all faces in one frame.

        Returns:
            List of result dicts, one per face:
            {
                "location":    (top, right, bottom, left),  # in original coords
                "student_id":  str | None,
                "name":        str,
                "distance":    float,
                "is_stranger": bool,
                "enrolling":   bool,
                "face_crop":   np.ndarray (BGR),
            }
        """
        # ── Step 1: Downscale frame for speed ─────────────────────────────
        # WHY: HOG scans a 64×128 window across the image with 8px strides.
        # Scaling to 0.25 reduces pixel count by 16× → ~16× fewer window positions.
        # The detected bounding boxes are scaled back up after detection.
        small = cv2.resize(frame_bgr, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

        # face_recognition expects RGB; OpenCV gives BGR
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ── Step 2: HOG Face Detection ─────────────────────────────────────
        # dlib HOG detector uses:
        #   - Divide image into 8×8 pixel cells
        #   - Compute gradient magnitude + direction in each cell
        #   - Group cells into 16×16 blocks, normalise → 36 values/block
        #   - Concatenate all block descriptors → feature vector
        #   - Slide a linear SVM classifier over this feature map
        # On M1 with Accelerate BLAS: ~8 ms per 480×270 frame (at 0.25× scale)
        locations_small = face_recognition.face_locations(
            rgb_small,
            number_of_times_to_upsample=HOG_UPSAMPLE,
            model="hog"
        )

        if not locations_small:
            self._last_results = []
            return []

        # ── Step 3: Compute 128-d ResNet embeddings ────────────────────────
        # dlib's ResNet-v1 backbone:
        #   - Input: aligned face chip (150×150 RGB)
        #   - Architecture: 29 conv layers with residual shortcuts
        #   - Output: 128-d L2-normalised embedding vector
        #
        # "num_jitters=1" means each face is passed through the network once.
        # num_jitters=10 averages 10 random perturbations → more accurate but
        # 10× slower.  jitters=1 is fine for real-time.
        encodings = face_recognition.face_encodings(
            rgb_small,
            known_face_locations=locations_small,
            num_jitters=1,
            model="large"    # "large" uses the full 5-point landmark aligner
        )

        # ── Scale bounding boxes back to original frame coordinates ────────
        inv = 1.0 / FRAME_SCALE
        locations_full = [
            (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
            for (t, r, b, l) in locations_small
        ]

        results = []

        for (location, encoding) in zip(locations_full, encodings):
            top, right, bottom, left = location

            # Extract face crop from original (full-res) frame
            face_crop = frame_bgr[top:bottom, left:right]

            # ── Step 4: Vector similarity matching ────────────────────────
            # db.find_match runs the vectorised L2 computation described
            # in database.py.  Returns best match or (None, None, inf).
            student_id, name, dist = self.db.find_match(
                encoding, RECOGNITION_THRESHOLD
            )

            is_stranger = student_id is None

            # ── Step 5: Stranger tracking → auto-enrollment trigger ────────
            enrolling = False
            if is_stranger or self.force_enroll:
                # Build a coarse position key (nearest 40px grid)
                pos_key = f"{(left//40)*40}_{(top//40)*40}"
                now = time.time()

                if pos_key not in self._stranger_tracker:
                    self._stranger_tracker[pos_key] = {
                        "first_seen": now,
                        "embedding":  encoding,
                        "crop":       face_crop.copy(),
                    }

                tracked = self._stranger_tracker[pos_key]
                dwell_time = now - tracked["first_seen"]

                if dwell_time >= ENROLL_TRIGGER_SECONDS and not self._enrollment_active:
                    # Face has been visible long enough — trigger UI enrollment
                    enrolling = True
                    notify_ui_enrollment_needed({
                        'embedding': tracked["embedding"].tolist()
                    })
                    self._stranger_tracker.pop(pos_key, None)
                    self.force_enroll = False
                elif dwell_time < ENROLL_TRIGGER_SECONDS:
                    enrolling = True   # Still counting down

            else:
                # Recognised — log attendance and clear any tracker entry
                self.attendance.log(student_id, name, "", self.session)
                notify_ui_attendance_logged({
                    'name': name,
                    'student_id': student_id,
                    'timestamp': datetime.now().isoformat()
                })
                # Clean up tracker (face was unknown before enrollment)
                pos_key = f"{(left//40)*40}_{(top//40)*40}"
                self._stranger_tracker.pop(pos_key, None)

            results.append({
                "location":    location,
                "student_id":  student_id,
                "name":        name if name else "Unknown",
                "distance":    round(dist, 3),
                "is_stranger": is_stranger,
                "enrolling":   enrolling,
                "face_crop":   face_crop,
            })

        # Prune stale tracker entries (face left frame)
        current_keys = {
            f"{(l//40)*40}_{(t//40)*40}"
            for (t, r, b, l) in locations_full
        }
        stale = [k for k in self._stranger_tracker if k not in current_keys]
        for k in stale:
            self._stranger_tracker.pop(k, None)

        self._last_results = results
        return results

    def get_last_results(self) -> list[dict]:
        """Return cached results for use on skipped frames."""
        return self._last_results


# ═══════════════════════════════════════════════════════════════════════════
# FRAME RENDERER
# Draws bounding boxes, name badges, confidence bars, and FPS onto the frame.
# ═══════════════════════════════════════════════════════════════════════════

class FrameRenderer:

    @staticmethod
    def render(frame: np.ndarray, results: list[dict], fps: float) -> np.ndarray:
        h, w = frame.shape[:2]

        for r in results:
            top, right, bottom, left = r["location"]

            # ── Choose colour ──────────────────────────────────────────────
            if r["enrolling"]:
                color = COLOR_ENROLLING
                label = "Registering..."
            elif not r["is_stranger"]:
                color = COLOR_KNOWN
                # Confidence = 1 - (distance / threshold)
                # A distance of 0.0 → 100% confidence
                # A distance of 0.6 → 0% confidence
                conf  = max(0.0, 1.0 - r["distance"] / RECOGNITION_THRESHOLD)
                label = f"{r['name']}  {conf*100:.0f}%"
            else:
                color = COLOR_UNKNOWN
                label = f"Unknown  d={r['distance']:.3f}"

            # ── Bounding box ──────────────────────────────────────────────
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # ── Name badge ────────────────────────────────────────────────
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.rectangle(frame, (left, bottom), (left + tw + 12, bottom + th + 10),
                          color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom + th + 4),
                        FONT, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

            # ── Confidence bar (known faces only) ─────────────────────────
            if not r["is_stranger"] and not r["enrolling"]:
                conf = max(0.0, 1.0 - r["distance"] / RECOGNITION_THRESHOLD)
                bar_w = right - left
                filled = int(bar_w * conf)
                bar_y  = top - 6
                cv2.rectangle(frame, (left, bar_y - 4), (right, bar_y), (40, 40, 40), -1)
                cv2.rectangle(frame, (left, bar_y - 4), (left + filled, bar_y), color, -1)

        # ── Status bar ────────────────────────────────────────────────────
        known = sum(1 for r in results if not r["is_stranger"])
        total = len(results)
        status = f"FPS {fps:.1f}  |  Faces {total}  |  Recognised {known}  |  Q=quit E=enroll"
        cv2.rectangle(frame, (0, h - 32), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, status, (10, h - 10), FONT, 0.48,
                    (180, 180, 180), 1, cv2.LINE_AA)

        return frame


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Video Stream Loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Unique session ID for grouping attendance records
    session = datetime.now().strftime("%Y%m%d_%H%M") + "_" + str(uuid.uuid4())[:4].upper()
    print(f"[MAIN] Session: {session}")

    db         = FaceDatabase()
    attendance = AttendanceLogger()
    engine     = RecognitionEngine(db, attendance, session)
    renderer   = FrameRenderer()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # Reduce capture latency on M1

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera 0")

    frame_count = 0
    fps_timer   = time.time()
    fps_display = 0.0
    fps_counter = 0

    print("[MAIN] Stream started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        fps_counter += 1

        # ── FPS calculation ───────────────────────────────────────────────
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.time()

        # ── Recognition every N frames, render every frame ────────────────
        # This decouples the display FPS from the recognition FPS.
        # The overlay from the last recognition pass is re-drawn every frame,
        # keeping the UI smooth even though recognition runs at ~10 FPS.
        if frame_count % RECOGNITION_EVERY_N_FRAMES == 0:
            results = engine.process_frame(frame)
        else:
            results = engine.get_last_results()

        # ── Render ────────────────────────────────────────────────────────
        display = renderer.render(frame.copy(), results, fps_display)
        cv2.imshow("FaceAttend Pro — GEHU", display)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            engine.force_enroll = True
            print("[MAIN] Force-enroll mode activated for next face.")
        elif key == ord('s'):
            db.save()
            print("[MAIN] Database saved.")

    cap.release()
    cv2.destroyAllWindows()
    db.save()
    print("[MAIN] Session ended.")


if __name__ == "__main__":
    main()