# Face Recognition Attendance System

A MacBook-optimized Python face recognition attendance system that captures student identities, recognizes them in real-time, and logs attendance automatically.

## Features

✅ **Real-time Face Recognition** — Detects and recognizes multiple faces simultaneously using 128-dimensional face embeddings  
✅ **Student Enrollment** — Terminal-based enrollment capturing Name, Roll No, and Section  
✅ **Automatic Attendance Logging** — Logs attendance with timestamps, preventing duplicate entries within a cooldown period  
✅ **CSV-Based Storage** — Simple CSV files for student records and attendance logs  
✅ **Video Display** — Real-time video feed with bounding boxes, names, and confidence scores  
✅ **MacBook M1 Optimized** — Uses frame scaling and smart frame skipping for high FPS  

## System Architecture

### Key Components

1. **StudentDatabase** — Manages student records and face encodings stored in `data/student_records.csv`
2. **AttendanceLogger** — Logs attendance to `data/attendance.csv` with cooldown to prevent duplicates
3. **FaceRecognitionEngine** — Detects faces using HOG and recognizes them using 128-d ResNet embeddings
4. **VideoRenderer** — Displays video with bounding boxes and confidence scores

### Recognition Process

```
Frame Input
    ↓
Downscale to 0.25× (16× faster processing)
    ↓
HOG Face Detection
    ↓
ResNet-128d Encoding (compute embedding for each face)
    ↓
Euclidean Distance Matching (compare against all known students)
    ↓
Classify as RECOGNIZED or UNKNOWN
    ↓
Log attendance if recognized & not in cooldown
    ↓
Display with bounding boxes & confidence
```

## Installation

### 1. Clone and navigate to project
```bash
cd /Users/nupoor/Desktop/mini\ project\ hoja\ yr
```

### 2. Run setup script
```bash
chmod +x setup.sh
./setup.sh
```

Or install manually:
```bash
pip3 install -r requirements.txt
```

### 3. Ensure data directories exist
```bash
mkdir -p data/snapshots
```

## Usage

### Start the System
```bash
python3 face_attendance_system.py
```

### Keyboard Controls
| Key | Action |
|-----|--------|
| **Q** | Quit the program |
| **E** | Enroll a new student (press once to start, shows enrollment form after capturing frames) |
| **S** | Save database |

### Enrollment Process

1. **Press 'E'** — Enter enrollment mode
2. **Position your face** — Keep your face visible for ~5 frames while the system captures encodings
3. **Press 'E' again** — When you see the enrollment form
4. **Enter Details**:
   - Student name (e.g., "Arjun Sharma")
   - Roll number (e.g., "CSE001")
   - Section (e.g., "A")
5. **Confirm** — Student is enrolled and ready for recognition

### Recognition & Attendance

Once students are enrolled:
1. **Faces detected** — Real-time HOG face detection
2. **Matched** — If distance < 0.6, student is recognized
3. **Logged** — Attendance automatically recorded with timestamp
4. **Cooldown** — Student won't be logged again within 5 minutes (configurable)

## Configuration

Edit `config.py` to customize:

```python
# Recognition confidence threshold
TOLERANCE = 0.6                    # Euclidean distance threshold

# Frame processing
FRAME_SCALE = 0.25                # Downscale factor (25% = 16× faster)
RECOGNITION_EVERY_N_FRAMES = 3    # Run recognition every Nth frame

# Enrollment
ENROLLMENT_SAMPLE_FRAMES = 5      # Number of frames to capture per enrollment

# Attendance logging
ATTENDANCE_COOLDOWN_MIN = 5        # Minutes between duplicate entries
```

## Data Files

### `data/student_records.csv`
Stores student information and face encodings:
```
name,roll_no,section,encoding,enrolled_at,snapshot_path
Arjun Sharma,CSE001,A,"0.123,0.456,...,0.789",2024-04-19T10:30:00,data/snapshots/CSE001_Arjun_Sharma_20240419_103000.jpg
```

**Encoding Format**: 128 comma-separated float values (128-dimensional ResNet embedding)

### `data/attendance.csv`
Logs attendance with timestamps:
```
name,roll_no,timestamp
Arjun Sharma,CSE001,2024-04-19T10:35:42.123456
```

## How Face Recognition Works

### ResNet-128d Embedding
- Each face is converted to a **128-dimensional vector** using dlib's ResNet
- Two photos of the same person → vectors are **close** (low Euclidean distance)
- Two different people → vectors are **far** (high Euclidean distance)

### Euclidean Distance
```
distance(A, B) = √[(A₁-B₁)² + (A₂-B₂)² + ... + (A₁₂₈-B₁₂₈)²]
```

### Matching
- If distance **< 0.6** → **Same person** (recognized)
- If distance **≥ 0.6** → **Different person** (unknown)

### Why It's Fast on MacBook
1. **HOG Detection** — Histogram of Oriented Gradients (not CNN-based)
2. **Frame Downscaling** — 0.25× = 16× fewer pixels to process
3. **Vectorized Operations** — NumPy + Apple Accelerate BLAS
4. **Frame Skipping** — Recognition every 3rd frame, display every frame
5. **Unified Memory** — M1/M2/M3 benefit from shared CPU/GPU memory

## Performance

### Expected FPS
- **Video Display**: 20-30 FPS (every frame rendered)
- **Face Recognition**: 7-10 FPS (every 3rd frame)
- **Frame Downscaling**: 0.25× reduces processing by ~16×

### Processing Time (per frame)
- Downscaling: ~1 ms
- HOG Detection: ~5 ms (0.25× scale)
- ResNet Encoding: ~15 ms (per face)
- Distance Matching: <1 ms (vectorized)

## Troubleshooting

### Camera Not Found
```
[ERROR] Cannot open camera. Please check if it's connected.
```
- Ensure webcam is connected
- Try: `ls /dev/video*` to list available cameras
- May need to grant camera permissions in macOS settings

### Enrollment Not Working
- Ensure face is clearly visible and well-lit
- Keep still for ~5 frames during enrollment capture
- Try pressing 'E' again if form doesn't appear

### Low Recognition Accuracy
- Increase lighting in the room
- Enroll multiple times from different angles
- Adjust `TOLERANCE` in config.py (lower = stricter, higher = more lenient)

### Performance Issues
- Increase `FRAME_SCALE` to further reduce resolution
- Increase `RECOGNITION_EVERY_N_FRAMES` to skip more frames
- Reduce `ENROLLMENT_SAMPLE_FRAMES` for faster enrollment

## Project Structure

```
mini project hoja yr/
├── face_attendance_system.py    # Main application
├── config.py                    # Configuration parameters
├── requirements.txt             # Python dependencies
├── setup.sh                     # Installation script
├── README.md                    # This file
└── data/
    ├── student_records.csv      # Student encodings (auto-created)
    ├── attendance.csv           # Attendance log (auto-created)
    └── snapshots/               # Enrollment photos
```

## Technical Details

### Dependencies
- **face-recognition** — Wraps dlib's ResNet face encoder
- **opencv-python** — Video capture and rendering
- **dlib** — HOG face detector, ResNet embedding model
- **numpy** — Vectorized distance computation

### Why CSV Over Database
- **500 students × 512 bytes/encoding = 256 KB** — Fits entirely in RAM
- **Euclidean search** — Vectorized numpy operations in microseconds
- **Simple, portable** — No database server needed
- **Human-readable** — Easy to export, inspect, backup

## License

Educational project for GEHU BCA

## Author

Built for face recognition attendance with MacBook optimization
