import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path

class GestureTrainer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.gestures_data = {}
        self.current_gesture = None
        self.recording = False
        self.samples = []
        self.gestures_dir = Path("gestures")
        self.gestures_dir.mkdir(exist_ok=True)

    def start_training(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            success, image = cap.read()
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    if self.recording and self.current_gesture:
                        # Record landmark data
                        landmarks_data = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        self.samples.append(landmarks_data)
                        cv2.putText(image, f"Recording: {self.current_gesture}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display controls
            cv2.putText(image, "R: Start Recording", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "S: Stop Recording", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, "ESC: Exit", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Gesture Training", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r') and not self.recording:
                gesture_name = input("Enter gesture name: ")
                self.start_recording(gesture_name)
            elif key == ord('s') and self.recording:
                self.stop_recording()
            elif key == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

    def start_recording(self, gesture_name):
        self.current_gesture = gesture_name
        self.recording = True
        self.samples = []

    def stop_recording(self):
        if self.samples:
            # Calculate average gesture
            avg_gesture = np.mean(self.samples, axis=0).tolist()
            self.gestures_data[self.current_gesture] = avg_gesture
            self.save_gesture(self.current_gesture, avg_gesture)
            print(f"Saved gesture: {self.current_gesture}")
        
        self.recording = False
        self.current_gesture = None
        self.samples = []

    def save_gesture(self, name, data):
        file_path = self.gestures_dir / f"{name}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f)

if __name__ == "__main__":
    trainer = GestureTrainer()
    trainer.start_training()
