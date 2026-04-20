#!/usr/bin/env python3
# ============================================================
# camera_test.py — Test camera access and OpenCV display
# ============================================================

import cv2
import sys

print("=" * 60)
print("CAMERA & OPENCV DIAGNOSTIC TEST")
print("=" * 60)
print()

# Check OpenCV
print("[TEST] OpenCV version:", cv2.__version__)
print("[TEST] Python version:", sys.version)
print()

# Test camera access
print("[TEST] Attempting to open camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[FAIL] ✗ Camera 0 not accessible")
    print()
    print("SOLUTIONS FOR MACOS:")
    print("1. Grant camera permission to your terminal:")
    print("   System Settings → Privacy & Security → Camera")
    print("   Make sure your terminal app (Terminal, iTerm2, etc.) has ✓")
    print()
    print("2. Try restarting the terminal or using iTerm2")
    print()
    print("3. Check if camera is in use by another app (FaceTime, Photo Booth, etc.)")
    sys.exit(1)

print("[PASS] ✓ Camera opened successfully")

# Try to grab a frame
ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    print(f"[PASS] ✓ Frame captured: {w}×{h}")
else:
    print("[FAIL] ✗ Could not capture frame")
    cap.release()
    sys.exit(1)

# Test OpenCV display
print()
print("[TEST] Attempting to display frame...")
cv2.namedWindow("Camera Test", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Camera Test", frame)

print("[PASS] ✓ Window created and frame displayed")
print("[INFO] A camera window should be visible on your screen")
print("[INFO] Press any key in the window to close...")

key = cv2.waitKey(5000)  # Wait 5 seconds
cv2.destroyAllWindows()

print()
print("=" * 60)
print("✓ ALL TESTS PASSED")
print("=" * 60)
print()
print("Your camera and OpenCV are working correctly!")
print("Run: python3 face_attendance_system.py")
print()

cap.release()
