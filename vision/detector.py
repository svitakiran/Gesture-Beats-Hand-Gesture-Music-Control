"""
uses mediapipe to detect hand poses from video feed
"""

import mediapipe as mp
import cv2
import numpy as np
from config import Config


class Hand:
    # represents a single detected hand
    
    def __init__(self, landmarks, handedness):
        self.landmarks = landmarks
        self.label = handedness
        self.positions = self._extract_positions(landmarks)
        
        # landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
    
    def _extract_positions(self, landmarks):
        # convert mediapipe landmarks to numpy array
        return np.array([
            [lm.x, lm.y, lm.z] for lm in landmarks.landmark
        ])
    
    def get_center(self):
        return self.positions.mean(axis=0)
    
    def get_wrist(self):
        return self.positions[self.WRIST]
    
    def get_finger_tips(self):
        return self.positions[[
            self.THUMB_TIP,
            self.INDEX_TIP,
            self.MIDDLE_TIP,
            self.RING_TIP,
            self.PINKY_TIP
        ]]


class Detector:
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=Config.MAX_NUM_HANDS,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE,
            model_complexity=1
        )
    
    def detect(self, frame):
        # convert BGR to RGB for mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # run detection
        results = self.hands.process(frame_rgb)
        
        detected_hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                hand = Hand(landmarks, hand_label)
                detected_hands.append(hand)
        
        return detected_hands
    
    def draw_hands(self, frame, hands):
        # draw hand skeletons on the frame
        if not Config.SHOW_HAND_SKELETON:
            return frame
        
        frame_copy = frame.copy()
        for hand in hands:
            self.mp_drawing.draw_landmarks(
                frame_copy,
                hand.landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            )
        
        return frame_copy
    
    def draw_hand_info(self, frame, hands):
        frame_copy = frame.copy()
        for hand in hands:
            center = hand.get_center()
            center_pixel = (
                int(center[0] * frame.shape[1]),
                int(center[1] * frame.shape[0])
            )
            
            cv2.circle(frame_copy, center_pixel, 10, (0, 255, 255), -1)
            
            cv2.putText(
                frame_copy,
                hand.label,
                (center_pixel[0] + 15, center_pixel[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
        
        return frame_copy