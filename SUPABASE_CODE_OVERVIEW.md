# Supabase Integration - Code Overview

## What Changed

Your face recognition system now has **cloud-based attendance logging** with Supabase while maintaining **local face enrollment** for privacy.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Face Recognition System                         │
│                                                              │
│  Camera Input                                                │
│     ↓                                                        │
│  Face Detection (HOG)                                       │
│     ↓                                                        │
│  Face Encoding (ResNet-128d)                               │
│     ↓                                                        │
│  Face Recognition (Local Database)                         │
│     ↓                                                        │
│  ┌─────────────────────────────────────────────┐           │
│  │ Attendance Logging (Choose One or Both)     │           │
│  ├─────────────────────────────────────────────┤           │
│  │ ✓ Supabase Cloud (if .env configured)      │           │
│  │ ✓ Local CSV (always as fallback)           │           │
│  └─────────────────────────────────────────────┘           │
│     ↓                                                        │
│  ✓ Attendance Recorded                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Setup

### 1. Install Supabase Library

```bash
pip install -r requirements.txt
```

This installs:
```
supabase==2.3.5
python-dotenv==1.0.0
```

### 2. Create .env File

Create a file named `.env` in your project directory:

```bash
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-public-key-here
```

**Where to find these values:**
1. Open https://supabase.com
2. Go to your project
3. Click Settings → API
4. Copy the "Project URL" and "anon public" key

### 3. Create Database Table

Go to your Supabase dashboard:
1. Click SQL Editor
2. Click New Query
3. Copy the contents of `schema.sql` and paste it
4. Click Run

This creates an `attendance` table with columns:
- `id` — Unique record ID
- `name` — Student name
- `roll_no` — Student roll number
- `department` — Student section/department
- `timestamp` — When attendance was logged
- `created_at` — Server creation time

### 4. Run the System

```bash
python3 face_attendance_system.py
```

The system will:
- Load credentials from `.env`
- Connect to Supabase
- Log attendance to both Supabase AND local CSV
- Show status in console

## Code Changes

### Credentials Loading

At the top of `face_attendance_system.py`:

```python
from dotenv import load_dotenv

# Load .env file
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
USE_SUPABASE = SUPABASE_URL and SUPABASE_KEY

if USE_SUPABASE:
    from supabase import create_client, Client
    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[SUPABASE] ✓ Connected successfully")
```

### AttendanceLogger Class (Updated)

```python
class AttendanceLogger:
    """Logs attendance to Supabase (or CSV as fallback)."""
    
    def log_attendance(self, name, roll_no, department=""):
        """
        Log to Supabase with fallback to CSV.
        Returns: True if logged, False if in cooldown.
        """
        # Check 5-minute cooldown
        student_id = f"{roll_no}_{name}"
        now = time.time()
        if now - self._last_seen.get(student_id, 0) < self._cooldown_secs:
            return False
        
        self._last_seen[student_id] = now
        timestamp = datetime.now().isoformat()
        
        # Try Supabase first
        if USE_SUPABASE:
            try:
                supabase_client.table("attendance").insert({
                    "name": name,
                    "roll_no": roll_no,
                    "department": department,
                    "timestamp": timestamp
                }).execute()
                
                print(f"[ATTENDANCE] ✓ {name} logged to Supabase")
                return True
            except Exception as e:
                print(f"[ATTENDANCE] ✗ Supabase error: {e}")
                # Fall back to CSV
        
        # Log to CSV
        self._log_to_csv(name, roll_no, department, timestamp)
        return True
```

## Data Flow

### When a Student is Recognized:

```
1. Face detected and encoded
2. Matched against local database
3. If distance < 0.6 → RECOGNIZED
4. Attendance logging triggered:
   ├─ Check 5-minute cooldown (local memory)
   ├─ If cooldown expired:
   │  ├─ Insert into Supabase attendance table
   │  ├─ If Supabase fails → Fall back to CSV
   │  └─ Print success message
   └─ If in cooldown → Skip (no duplicate)
```

## Viewing Attendance Data

### In Supabase Dashboard:

1. Click **Table Editor**
2. Select `attendance` table
3. View all logged records

### In Local CSV:

Open `data/attendance.csv` with any spreadsheet app

### Querying with SQL:

```sql
-- View today's attendance
SELECT name, roll_no, department, timestamp
FROM attendance
WHERE DATE(timestamp) = CURRENT_DATE
ORDER BY timestamp DESC;

-- Count attendance
SELECT COUNT(DISTINCT roll_no) as students_present
FROM attendance
WHERE DATE(timestamp) = CURRENT_DATE;
```

## Configuration

### Attendance Cooldown

Edit `config.py`:
```python
ATTENDANCE_COOLDOWN_MIN = 5  # 5 minutes (300 seconds)
```

### Toggle Supabase

To use **only CSV** (no Supabase):
- Don't create `.env` file
- System will auto-detect and use CSV only

To use **both Supabase and CSV**:
- Create `.env` with credentials
- System logs to both automatically

## Error Handling

The system handles errors gracefully:

| Scenario | Behavior |
|----------|----------|
| Supabase unavailable | Falls back to CSV |
| Network error | Logs to CSV immediately |
| Invalid credentials | Prints error, uses CSV |
| Table doesn't exist | Error message, uses CSV |
| .env file missing | Prints warning, uses CSV only |

## Monitoring

Watch the console output:

```
[SUPABASE] ✓ Connected to Supabase successfully
[ATTENDANCE] ✓ Nupoor pandey (53) logged to Supabase at 13:19:11
[ATTENDANCE] ✓ Bhumika Bora (22) logged to Supabase at 13:19:12
```

Or if Supabase is unavailable:

```
[SUPABASE] ✗ Failed to connect: Connection timeout
[SUPABASE] Falling back to local CSV logging
[ATTENDANCE] ✓ Nupoor pandey (53) logged to CSV at 13:19:11
```

## Files Modified/Created

### New Files:
- `.env.example` — Template for credentials
- `SUPABASE_SETUP.md` — Detailed setup guide
- `schema.sql` — SQL to create the attendance table

### Updated Files:
- `face_attendance_system.py` — Added Supabase integration
- `requirements.txt` — Added supabase and python-dotenv

### Unchanged:
- `config.py` — Configuration (still works as before)
- `data/student_records.csv` — Local enrollment storage
- `data/attendance.csv` — Local attendance backup

## Troubleshooting

### "ERROR: Could not find a version that satisfies the requirement supabase"

Run: `pip install --upgrade pip`

Then: `pip install supabase==2.3.5`

### "AttributeError: module 'supabase' has no attribute 'create_client'"

Update Supabase library:
```bash
pip install --upgrade supabase
```

### Attendance logged to CSV but not Supabase

Check:
1. Is `.env` file created?
2. Are SUPABASE_URL and SUPABASE_KEY correct?
3. Does the `attendance` table exist in Supabase?
4. Check console for error messages

### "No module named 'dotenv'"

Install: `pip install python-dotenv`

## Security Best Practices

1. **Never commit .env to Git:**
   ```bash
   echo ".env" >> .gitignore
   ```

2. **Keep API keys secret** — don't share `.env` file

3. **Use Row Level Security** in Supabase for production

4. **Rotate keys periodically** in Supabase dashboard

## Example: Complete Workflow

```bash
# 1. Setup (one time)
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Supabase credentials
# Create attendance table in Supabase using schema.sql

# 2. Run the system
python3 face_attendance_system.py

# 3. System output
[SUPABASE] ✓ Connected to Supabase successfully
[DB] Loaded 2 students with encodings
[SYSTEM] Camera initialized...

# 4. Recognition happens
[ATTENDANCE] ✓ Nupoor pandey (53) logged to Supabase at 13:19:11
[ATTENDANCE] ✓ Bhumika Bora (22) logged to Supabase at 13:19:12

# 5. View data
# Option A: Supabase dashboard → Table Editor
# Option B: data/attendance.csv
```

## Next Steps

1. ✓ Install dependencies
2. ✓ Create `.env` file
3. ✓ Create Supabase account
4. ✓ Set up attendance table
5. ✓ Run the system
6. ✓ View attendance in Supabase dashboard

You're all set! 🚀
