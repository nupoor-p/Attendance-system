#!/usr/bin/env python3
# ============================================================
# test_installation.py — Verify all dependencies are installed
# ============================================================

import sys
import subprocess

print("=" * 60)
print("TESTING INSTALLATION")
print("=" * 60)

packages = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "face_recognition": "face-recognition",
    "dlib": "dlib",
    "PIL": "Pillow"
}

all_ok = True

for module_name, package_name in packages.items():
    try:
        __import__(module_name)
        print(f"✓ {package_name:25} OK")
    except ImportError:
        print(f"✗ {package_name:25} NOT INSTALLED")
        all_ok = False

print()

# Test camera access
print("Testing camera access...")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ Camera access            OK")
        cap.release()
    else:
        print("✗ Camera not accessible")
        all_ok = False
except Exception as e:
    print(f"✗ Camera test failed: {e}")
    all_ok = False

print()
print("=" * 60)

if all_ok:
    print("✓ All tests passed! Ready to run:")
    print("  python3 face_attendance_system.py")
else:
    print("✗ Some packages are missing. Run:")
    print("  ./setup.sh")
    sys.exit(1)
