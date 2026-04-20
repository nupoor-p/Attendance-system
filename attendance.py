# ============================================================
# attendance.py  —  Attendance Logger
# Writes to CSV with a per-student cooldown to prevent
# duplicate entries in the same class period.
# ============================================================

import csv
import time
from datetime import datetime
from pathlib import Path
from config import ATTENDANCE_CSV, ATTENDANCE_COOLDOWN_MIN


class AttendanceLogger:
    def __init__(self):
        ATTENDANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
        self._last_seen: dict[str, float] = {}   # student_id → unix timestamp

        # Write header if file is new
        if not ATTENDANCE_CSV.exists():
            with open(ATTENDANCE_CSV, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["student_id", "name", "dept", "date", "time", "session"])

    def log(self, student_id: str, name: str, dept: str, session: str) -> bool:
        """
        Write one attendance record.  Returns True if written, False if
        within cooldown window (already marked this period).
        """
        now = time.time()
        last = self._last_seen.get(student_id, 0)
        cooldown_secs = ATTENDANCE_COOLDOWN_MIN * 60

        if now - last < cooldown_secs:
            return False    # Still in cooldown — skip

        self._last_seen[student_id] = now
        dt = datetime.now()
        with open(ATTENDANCE_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                student_id, name, dept,
                dt.strftime("%Y-%m-%d"),
                dt.strftime("%H:%M:%S"),
                session
            ])
        print(f"[ATTENDANCE] Marked: {name} ({student_id}) at {dt.strftime('%H:%M:%S')}")
        return True