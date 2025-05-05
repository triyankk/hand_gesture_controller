# Gesture Control System

This application enables control of your computer through hand gestures, with face detection for added security.

## Gesture Controls

### Activation
- System only accepts gestures when a face is detected
- Green border appears around the screen when system is ready to accept gestures

### Basic Controls
1. **Mouse Movement**
   - Open palm with spread fingers to move cursor
   - Hand must be visible to the camera

2. **Click Actions**
   - Close all fingers into a fist for left click
   - Hold fist closed to maintain click
   - Open hand to release click

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- PyAutoGUI
- Win32GUI

## Usage
1. Run the application
2. Position yourself in front of the camera
3. Wait for green border to appear (face detection)
4. Use gestures to control the cursor

## Calibration
1. Run the calibrator first: `python gesture_calibrator.py`
2. Follow on-screen instructions to calibrate:
   - Open Palm: Spread all fingers wide
   - Closed Fist: Close all fingers tightly
3. Hold each pose steady for 3 seconds during calibration