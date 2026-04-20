# ============================================================
# config.py  —  Face Recognition Attendance System Configuration
# All tunable parameters live here so you never hunt through
# main code to change a threshold or a path.
# ============================================================

import pathlib

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR            = pathlib.Path(__file__).parent
STUDENT_RECORDS_CSV = BASE_DIR / "data" / "student_records.csv"    # Face encodings + student info
ATTENDANCE_CSV      = BASE_DIR / "data" / "attendance.csv"         # Attendance log
SNAPSHOTS_DIR       = BASE_DIR / "data" / "snapshots"              # Enrollment photos
DB_PATH             = BASE_DIR / "data" / "face_db.pkl"            # Legacy pickle store (optional)

# ── Detection ─────────────────────────────────────────────────────────────
# Scale factor: 0.25 means the frame is shrunk to 25% before detection runs.
# WHY: Face detection scans the image. Fewer pixels = faster scan.
# 0.25 gives ~16× fewer pixels to process.
FRAME_SCALE         = 0.25

# Number of HOG pyramid levels.  0 = fastest (may miss small/far faces).
HOG_UPSAMPLE        = 1

# ── Recognition ───────────────────────────────────────────────────────────
# EUCLIDEAN THRESHOLD — the core metric for face matching.
# 
# dlib's ResNet maps every face to a 128-dimensional vector.
# Two photos of the SAME person produce vectors that are CLOSE together.
# Two DIFFERENT people produce vectors that are FAR apart.
#
# We measure "how far" with the Euclidean distance:
#   d(A, B) = sqrt( (A1-B1)² + (A2-B2)² + … + (A128-B128)² )
#
# Default thresholds:
#   d < 0.6  →  probably the same person
#   d ≥ 0.6  →  probably a different person
TOLERANCE               = 0.6

# ── Enrollment ───────────────────────────────────────────────────────────
# Number of frames to capture and average for robust enrollment
ENROLLMENT_SAMPLE_FRAMES = 5

# ── Attendance Logging ───────────────────────────────────────────────────
# Cooldown period (minutes): prevent logging the same student twice within this window
ATTENDANCE_COOLDOWN_MIN = 5

# ── Performance (MacBook M1 Optimizations) ──────────────────────────────
# M1 uses unified memory.  4 matches M1's performance-core count.
OPENCV_THREADS          = 4

# Process every Nth frame for recognition (detection runs every frame).
# Frame 1: detect + recognize.  Frame 2, 3: reuse last result (overlay only).
RECOGNITION_EVERY_N_FRAMES  = 3