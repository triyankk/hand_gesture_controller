import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import win32gui
import win32con
import win32api
import json
from pathlib import Path
import time

class GestureController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.screen_width, self.screen_height = pyautogui.size()
        self.cap = cv2.VideoCapture(0)
        # Set camera resolution to 1280x720 (16:9)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Verify if camera accepts the resolution
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera resolution: {actual_width}x{actual_height}")
        
        # Smoothing factor
        self.smoothing = 0.5  # Smoothing factor between 0 and 1
        self.cursor_x, self.cursor_y = pyautogui.position()
        self.is_clicking = False
        self.border_hwnd = None
        self.create_border_window()
        self.gestures_dir = Path("gestures")
        self.loaded_gestures = self.load_gestures()
        self.debug_mode = True  # Add debug mode for gesture recognition
        self.current_gesture = None
        self.gesture_confidence = 0.0
        self.hover_start_time = 0
        self.hover_position = None
        self.hover_threshold = 2.0  # seconds
        self.movement_tolerance = 20  # pixels
        self.hover_feedback_color = (0, 255, 255)  # Yellow color for hover feedback

    def create_border_window(self):
        # Register window class
        wc = win32gui.WNDCLASS()
        wc.lpszClassName = "GestureControlBorder"
        wc.hbrBackground = win32gui.CreateSolidBrush(win32api.RGB(0, 255, 0))
        try:
            win32gui.RegisterClass(wc)
        except:
            pass  # Class already registered

        # Create border windows (4 sides)
        border_thickness = 4
        styles = win32con.WS_POPUP | win32con.WS_VISIBLE
        ex_styles = win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT

        # Store all border handles
        self.borders = []
        # Top border
        hwnd = win32gui.CreateWindowEx(
            ex_styles, "GestureControlBorder", None, styles,
            0, 0, self.screen_width, border_thickness,
            None, None, None, None
        )
        self.borders.append(hwnd)
        
        # Bottom border
        hwnd = win32gui.CreateWindowEx(
            ex_styles, "GestureControlBorder", None, styles,
            0, self.screen_height - border_thickness, self.screen_width, border_thickness,
            None, None, None, None
        )
        self.borders.append(hwnd)
        
        # Left border
        hwnd = win32gui.CreateWindowEx(
            ex_styles, "GestureControlBorder", None, styles,
            0, 0, border_thickness, self.screen_height,
            None, None, None, None
        )
        self.borders.append(hwnd)
        
        # Right border
        hwnd = win32gui.CreateWindowEx(
            ex_styles, "GestureControlBorder", None, styles,
            self.screen_width - border_thickness, 0, border_thickness, self.screen_height,
            None, None, None, None
        )
        self.borders.append(hwnd)

    def show_active_border(self, show=True):
        for hwnd in self.borders:
            if show:
                win32gui.SetLayeredWindowAttributes(hwnd, 0, 180, win32con.LWA_ALPHA)
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            else:
                win32gui.ShowWindow(hwnd, win32con.SW_HIDE)

    def start(self):
        while True:
            success, image = self.cap.read()
            image = cv2.flip(image, 1)  # Mirror image
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect face first
            face_results = self.face_detection.process(rgb_image)
            face_detected = face_results.detections is not None and len(face_results.detections) > 0
            
            self.show_active_border(face_detected)
            
            if face_detected:
                results = self.hands.process(rgb_image)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_draw.draw_landmarks(
                            image, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Process hand landmarks with image parameter
                        self.process_hand_landmarks(hand_landmarks, image, image.shape[0], image.shape[1])
            
            cv2.imshow("Gesture Control", image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def load_gestures(self):
        gestures = {}
        if self.gestures_dir.exists():
            for gesture_file in self.gestures_dir.glob("*.json"):
                with open(gesture_file, 'r') as f:
                    data = json.load(f)
                    # Handle both old and new format
                    if isinstance(data, dict):
                        gestures[gesture_file.stem] = data['landmarks']
                    else:
                        gestures[gesture_file.stem] = data
                print(f"Loaded gesture: {gesture_file.stem}")
        return gestures

    def compare_gesture(self, current_landmarks):
        if not self.loaded_gestures:
            if self.debug_mode:
                print("No calibrated gestures found. Please run calibrator first.")
            return None
            
        landmarks_data = [[lm.x, lm.y, lm.z] for lm in current_landmarks.landmark]
        
        best_match = None
        min_distance = float('inf')
        
        for gesture_name, gesture_data in self.loaded_gestures.items():
            try:
                # Calculate similarity score
                distances = [np.linalg.norm(np.array(a) - np.array(b)) 
                           for a, b in zip(landmarks_data, gesture_data)]
                avg_distance = np.mean(distances)
                max_distance = np.max(distances)
                
                if self.debug_mode:
                    print(f"Gesture {gesture_name}: avg_dist={avg_distance:.3f}, max_dist={max_distance:.3f}")
                
                if avg_distance < min_distance and avg_distance < 0.3:  # Adjusted threshold
                    min_distance = avg_distance
                    best_match = gesture_name
                    self.gesture_confidence = 1.0 - min(avg_distance * 3, 1.0)
            except Exception as e:
                if self.debug_mode:
                    print(f"Error comparing with {gesture_name}: {e}")
                continue
        
        if self.debug_mode and best_match:
            print(f"Detected: {best_match} (confidence: {self.gesture_confidence:.2f})")
            
        return best_match if self.gesture_confidence > 0.5 else None

    def is_hand_open(self, hand_landmarks):
        # Check if all fingers are extended
        fingers = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        try:
            return all(self.is_finger_open(hand_landmarks, finger) for finger in fingers)
        except Exception:
            return False

    def is_fist_closed(self, hand_landmarks):
        # Check if all fingers are closed
        fingers = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        try:
            return not any(self.is_finger_open(hand_landmarks, finger) for finger in fingers)
        except Exception:
            return False

    def is_index_pointing(self, hand_landmarks):
        try:
            # Get finger states
            index_extended = (hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y < 
                            hand_landmarks.landmark[6].y)
            
            other_fingers_closed = all(
                hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
                for tip, pip in [(12,11), (16,15), (20,19)]
            )
            
            # Check position relative to calibrated pointing gesture
            if "pointing" in self.loaded_gestures:
                landmarks_data = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                distance = np.mean([np.linalg.norm(np.array(a) - np.array(b)) 
                                  for a, b in zip(landmarks_data, self.loaded_gestures["pointing"])])
                return distance < 0.3 and index_extended and other_fingers_closed
            
            return index_extended and other_fingers_closed
        except:
            return False

    def is_within_tolerance(self, pos1, pos2):
        if not pos1 or not pos2:
            return False
        x1, y1 = pos1
        x2, y2 = pos2
        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance <= self.movement_tolerance

    def process_hand_landmarks(self, hand_landmarks, image, frame_height, frame_width):
        if self.is_index_pointing(hand_landmarks):
            # Get current cursor position
            index_tip = hand_landmarks.landmark[8]
            current_x = int(index_tip.x * self.screen_width)
            current_y = int(index_tip.y * self.screen_height)
            current_pos = (current_x, current_y)

            # Smooth cursor movement
            self.cursor_x = self.cursor_x + (current_x - self.cursor_x) * self.smoothing
            self.cursor_y = self.cursor_y + (current_y - self.cursor_y) * self.smoothing
            
            # Move cursor
            pyautogui.moveTo(int(self.cursor_x), int(self.cursor_y), _pause=False)

            # Handle hover detection
            current_time = time.time()
            
            if self.hover_position:
                if self.is_within_tolerance(current_pos, self.hover_position):
                    # Still within tolerance of hover position
                    hover_duration = current_time - self.hover_start_time
                    if hover_duration >= self.hover_threshold:
                        # Perform click
                        pyautogui.click()
                        # Reset hover state
                        self.hover_position = None
                        self.hover_start_time = 0
                    else:
                        # Draw hover feedback
                        screen_x = int(self.cursor_x)
                        screen_y = int(self.cursor_y)
                        progress = hover_duration / self.hover_threshold
                        radius = int(self.movement_tolerance * (1 - progress))
                        cv2.circle(image, (screen_x, screen_y), radius, 
                                 self.hover_feedback_color, 2)
                else:
                    # Moved outside tolerance
                    self.hover_position = current_pos
                    self.hover_start_time = current_time
            else:
                # Start new hover
                self.hover_position = current_pos
                self.hover_start_time = current_time

            if self.debug_mode:
                cv2.putText(image, f"Hover: {time.time() - self.hover_start_time:.1f}s", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        else:
            # Reset hover state when not pointing
            self.hover_position = None
            self.hover_start_time = 0

        pointing = self.is_index_pointing(hand_landmarks)
        gesture = None

        if pointing:
            # Use index finger tip for cursor control
            index_tip = hand_landmarks.landmark[8]
            target_x = int(index_tip.x * self.screen_width)
            target_y = int(index_tip.y * self.screen_height)
            
            self.cursor_x = self.cursor_x + (target_x - self.cursor_x) * self.smoothing
            self.cursor_y = self.cursor_y + (target_y - self.cursor_y) * self.smoothing
            
            pyautogui.moveTo(int(self.cursor_x), int(self.cursor_y), _pause=False)
        else:
            # Check for fist gesture
            gesture = self.compare_gesture(hand_landmarks)

        if self.debug_mode:
            status = []
            if pointing:
                status.append("Pointing")
            if gesture:
                status.append(gesture)
            cv2.putText(image, " | ".join(status) or "No gesture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Handle clicking
        if gesture == "closed_fist" and not self.is_clicking:
            pyautogui.mouseDown(button='left')
            self.is_clicking = True
        elif gesture != "closed_fist" and self.is_clicking:
            pyautogui.mouseUp(button='left')
            self.is_clicking = False

    def is_finger_open(self, hand_landmarks, finger_name):
        try:
            if finger_name == "THUMB":
                tip = self.mp_hands.HandLandmark.THUMB_TIP
                pip = self.mp_hands.HandLandmark.THUMB_IP
            elif finger_name == "INDEX":
                tip = self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                pip = self.mp_hands.HandLandmark.INDEX_FINGER_PIP
            elif finger_name == "MIDDLE":
                tip = self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                pip = self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP
            elif finger_name == "RING":
                tip = self.mp_hands.HandLandmark.RING_FINGER_TIP
                pip = self.mp_hands.HandLandmark.RING_FINGER_PIP
            elif finger_name == "PINKY":
                tip = self.mp_hands.HandLandmark.PINKY_TIP
                pip = self.mp_hands.HandLandmark.PINKY_PIP
            else:
                return False
            
            return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
        except (AttributeError, IndexError):
            return False

if __name__ == "__main__":
    controller = GestureController()
    controller.start()