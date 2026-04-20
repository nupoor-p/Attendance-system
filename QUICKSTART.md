# QUICK START GUIDE
# Face Recognition Attendance System

## Installation (First Time Only)

### Option 1: Using the setup script
```bash
cd /Users/nupoor/Desktop/mini\ project\ hoja\ yr
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual installation
```bash
pip3 install -r requirements.txt
```

### Verify Installation
```bash
python3 test_installation.py
```

---

## Running the System

### Quick Start
```bash
python3 face_attendance_system.py
```

Or use the launcher script:
```bash
chmod +x run.sh
./run.sh
```

---

## How to Use (During Runtime)

### Main Display
- **Video feed** with real-time face detection
- **Status bar** shows FPS, number of faces, recognized count
- **Green boxes** = recognized students
- **Orange boxes** = unknown faces
- **Confidence bars** above recognized faces

### Enrolling New Students

1. **Press 'E'** — Start enrollment mode
2. **Position your face** in front of camera
   - Keep still and clearly visible
   - Let the system capture ~5 frames
3. **Press 'E' again** — Trigger enrollment form
4. **Enter student details**:
   - Name: Arjun Sharma
   - Roll No: CSE001
   - Section: A
5. **Press Enter** — Student is enrolled!

### Recognition & Attendance

Once enrolled:
- Student appears on camera → automatically detected
- If distance < 0.6 → **Recognized** (green box)
- If recognized → **Attendance logged** automatically
- **Cooldown**: Student won't be logged again within 5 minutes

### Keyboard Controls (While Camera is Active)

| Key | Action |
|-----|--------|
| **Q** | Quit program |
| **E** | Enroll new student |
| **S** | Save database |

---

## Output Files

### attendance.csv
```
name,roll_no,timestamp
Arjun Sharma,CSE001,2024-04-19T10:35:42.123456
Priya Singh,CSE002,2024-04-19T10:36:15.654321
```

### student_records.csv
```
name,roll_no,section,encoding,...
Arjun Sharma,CSE001,A,"0.123,0.456,...,0.789",...
```

### Snapshots
- Stored in `data/snapshots/`
- Filename: `{roll_no}_{name}_{timestamp}.jpg`

---

## Troubleshooting

### Camera not opening?
- Check camera is connected
- Grant camera permissions in macOS:
  - System Settings → Privacy & Security → Camera

### Poor recognition?
- Ensure good lighting
- Enroll multiple times from different angles
- Adjust `TOLERANCE` in config.py (default: 0.6)

### Slow performance?
- Reduce video resolution in config.py
- Increase `FRAME_SCALE` to 0.5
- Increase `RECOGNITION_EVERY_N_FRAMES` to 5

---

## Configuration (config.py)

```python
TOLERANCE = 0.6                    # Recognition threshold (lower = stricter)
FRAME_SCALE = 0.25                # Frame downscale (0.25 = 25%)
ENROLLMENT_SAMPLE_FRAMES = 5      # Frames to capture per enrollment
ATTENDANCE_COOLDOWN_MIN = 5        # Minutes between duplicate logs
```

---

## Technical Details

### Face Encoding
- Uses ResNet-128d (128-dimensional face vector)
- Computed from face landmarks (5-point alignment)
- Euclidean distance measures similarity

### Detection
- HOG (Histogram of Oriented Gradients)
- Fast on CPU, ~8ms per frame at 0.25× scale

### Matching
- Distance < 0.6 → same person (recognized)
- Distance ≥ 0.6 → different person (unknown)

---

## Files Structure

```
.
├── face_attendance_system.py    # Main application
├── config.py                    # Configuration
├── requirements.txt             # Dependencies
├── setup.sh                     # Installation script
├── run.sh                       # Quick launcher
├── test_installation.py         # Verify setup
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
└── data/
    ├── student_records.csv      # Student database
    ├── attendance.csv           # Attendance log
    └── snapshots/               # Enrollment photos
```

---

## Tips & Best Practices

✅ **Good Enrollment**
- Front-facing camera angle
- Even lighting (avoid shadows)
- Natural expression
- 5-7 capture frames

✅ **Best Recognition**
- Similar lighting to enrollment
- Similar distance from camera
- Clear, unobstructed face

❌ **Avoid**
- Sunglasses, masks (during enrollment)
- Poor lighting
- Side angles (front-facing works best)
- Enrolling too quickly

---

## Example Session

```
$ python3 face_attendance_system.py

============================================================
FACE RECOGNITION ATTENDANCE SYSTEM
============================================================
Loading database...
[DB] Loaded 2 students with encodings
[ATTENDANCE] Logger initialized
[SYSTEM] Camera initialized. Starting recognition...
[SYSTEM] Press 'Q' to quit, 'E' to enroll, 'S' to save

Press 'E' to enroll a new student...

[ENROLLMENT] Mode activated. Show your face to camera.
[ENROLLMENT] Mode activated. Show your face clearly...

[ENROLLMENT] Snapshot saved to data/snapshots/CSE001_...
[DB] Enrolled Arjun Sharma (CSE001) from A
[ENROLLMENT] ✓ Successfully enrolled Arjun Sharma

Arjun Sharma appears on camera:
[ATTENDANCE] ✓ Arjun Sharma (CSE001) marked present at 10:35:42

Press 'Q' to quit...
[SYSTEM] Shutting down...
[SYSTEM] Session ended.
```

---

Need help? Check README.md for detailed documentation.
