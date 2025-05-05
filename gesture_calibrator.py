import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
import time

class GestureCalibrator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.gestures_dir = Path("gestures")
        self.gestures_dir.mkdir(exist_ok=True)
        
        # Clear existing gesture files
        for file in self.gestures_dir.glob("*.json"):
            file.unlink()
        
    def validate_hand_position(self, hand_landmarks, gesture_name):
        # Check basic position first
        points = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
        basic_position = 0.1 < points[:, 0].mean() < 0.9 and 0.1 < points[:, 1].mean() < 0.9

        # Get finger states
        fingers = {
            'index': (8, 7, 6),  # tip, pip, mcp
            'middle': (12, 11, 10),
            'ring': (16, 15, 14),
            'pinky': (20, 19, 18)
        }

        def is_finger_extended(finger_points):
            tip, pip, mcp = [hand_landmarks.landmark[i] for i in finger_points]
            return tip.y < pip.y < mcp.y

        if gesture_name == "pointing":
            # Only index finger should be extended
            finger_states = {
                finger: is_finger_extended(points)
                for finger, points in fingers.items()
            }
            return (basic_position and 
                   finger_states['index'] and 
                   not any(finger_states[f] for f in ['middle', 'ring', 'pinky']))

        elif gesture_name == "closed_fist":
            # No fingers should be extended
            return (basic_position and 
                   not any(is_finger_extended(points) for points in fingers.values()))

        return False
    
    def start_calibration(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        gestures = {
            "pointing": "Point with your index finger (other fingers closed)",
            "closed_fist": "Make a tight fist"
        }

        for gesture_name, instruction in gestures.items():
            samples = []
            stage = "preparation"  # preparation, countdown, recording
            countdown = 3
            last_time = time.time()
            recording_duration = 2  # seconds
            recording_start = 0
            
            while True:
                success, image = cap.read()
                if not success:
                    continue
                    
                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_image)
                
                # Draw rectangle guide
                h, w = image.shape[:2]
                margin = int(0.1 * min(h, w))
                cv2.rectangle(image, (margin, margin), (w-margin, h-margin), (0,255,0), 2)

                if stage == "preparation":
                    cv2.putText(image, f"Prepare: {gesture_name}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(image, instruction, (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(image, "Press SPACE when ready", (10, 110),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
                    if cv2.waitKey(1) & 0xFF == 32:  # SPACE
                        stage = "countdown"
                        
                elif stage == "countdown":
                    now = time.time()
                    if now - last_time >= 1:
                        countdown -= 1
                        last_time = now
                        
                    if countdown <= 0:
                        stage = "recording"
                        recording_start = time.time()
                    
                    cv2.putText(image, f"Starting in {countdown}", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                              
                elif stage == "recording":
                    elapsed = time.time() - recording_start
                    progress = int((elapsed / recording_duration) * 100)
                    
                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        self.mp_draw.draw_landmarks(image, hand_landmarks, 
                                                 self.mp_hands.HAND_CONNECTIONS)
                        
                        if self.validate_hand_position(hand_landmarks, gesture_name):
                            landmarks_data = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                            samples.append(landmarks_data)
                            color = (0,255,0)
                        else:
                            color = (0,0,255)
                    else:
                        color = (0,0,255)
                        
                    cv2.putText(image, f"Recording... {progress}%", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    if elapsed >= recording_duration:
                        if len(samples) >= 20:  # Minimum samples required
                            # Calculate median gesture instead of mean for more robustness
                            median_gesture = np.median(samples, axis=0).tolist()
                            with open(self.gestures_dir / f"{gesture_name}.json", 'w') as f:
                                json.dump({
                                    'landmarks': median_gesture,
                                    'timestamp': time.time(),
                                    'samples_count': len(samples)
                                }, f)
                            print(f"Saved {gesture_name} with {len(samples)} samples")
                            break
                        else:
                            # Not enough samples, retry
                            samples = []
                            recording_start = time.time()
                
                cv2.imshow("Calibration", image)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrator = GestureCalibrator()
    calibrator.start_calibration()
