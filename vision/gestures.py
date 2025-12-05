"""
recognizing hand gestures
"""

import numpy as np
from dataclasses import dataclass
from config import Config


@dataclass
class Gesture:
    name: str
    confidence: float
    hand_label: str
    
    def __repr__(self):
        return f"{self.name} ({self.confidence:.2f}) - {self.hand_label}"


class GestureClassifier:
    """classifies hand gestures from mediapipe hand poses"""
    
    def __init__(self):
        # initialize gesture classifier with predefined profiles
        self.gesture_profiles = {
            "open_palm": {
                "finger_spread": (0.6, 1.0),
                "thumb_extended": True,
                "fingers_extended": True
            },
            "fist": {
                "finger_spread": (0.0, 0.3),
                "thumb_extended": False,
                "fingers_extended": False
            },
            "peace_sign": {
                "index_extended": True,
                "middle_extended": True,
                "ring_pinky_closed": True,
                "finger_spread": (0.4, 0.8)
            },
            "thumbs_up": {
                "thumb_extended": True,
                "other_fingers_closed": True,
                "thumb_pointing_up": True
            },
            "pointing": {
                "index_extended": True,
                "other_fingers_closed": True,
                "index_pointing": True
            }
        }
    
    def classify(self, hand):
        # classify a hand gesture
        features = self._extract_features(hand)
        best_gesture = None
        best_score = 0
        
        for gesture_name, profile in self.gesture_profiles.items():
            score = self._match_gesture(features, gesture_name, profile)
            if score > best_score:
                best_score = score
                best_gesture = gesture_name
        
        return Gesture(
            name=best_gesture or "unknown",
            confidence=best_score,
            hand_label=hand.label
        )
    
    def _extract_features(self, hand):
        # extract meaningful features from hand landmarks
        positions = hand.positions
        wrist = positions[hand.WRIST]
        
        # finger tip distances from wrist
        thumb_tip = positions[hand.THUMB_TIP]
        index_tip = positions[hand.INDEX_TIP]
        middle_tip = positions[hand.MIDDLE_TIP]
        ring_tip = positions[hand.RING_TIP]
        pinky_tip = positions[hand.PINKY_TIP]
        
        # calculate distances
        # normalize by hand size
        hand_size = np.linalg.norm(middle_tip - wrist)
        
        thumb_dist = np.linalg.norm(thumb_tip - wrist) / hand_size
        index_dist = np.linalg.norm(index_tip - wrist) / hand_size
        middle_dist = np.linalg.norm(middle_tip - wrist) / hand_size
        ring_dist = np.linalg.norm(ring_tip - wrist) / hand_size
        pinky_dist = np.linalg.norm(pinky_tip - wrist) / hand_size
        
        # finger spread
        finger_tips = hand.get_finger_tips()
        finger_spread = self._calculate_spread(finger_tips)
        
        # finger curling
        finger_curl = self._calculate_curl(positions)

        # check finger extension by comparing tip position to base joint - more accurate
        # Landmark indices: base joints are at 1, 5, 9, 13, 17 for thumb through pinky
        thumb_base = positions[2]  # Thumb uses CMC joint
        index_base = positions[5]   # Index MCP joint
        middle_base = positions[9]  # Middle MCP joint
        ring_base = positions[13]   # Ring MCP joint
        pinky_base = positions[17]  # Pinky MCP joint

        # extended fingers - tip should be farther from wrist than base with large tip 
        # to base distance
        thumb_extended = np.linalg.norm(thumb_tip - wrist) > np.linalg.norm(thumb_base - wrist) * 1.3
        index_extended = np.linalg.norm(index_tip - wrist) > np.linalg.norm(index_base - wrist) * 1.4
        middle_extended = np.linalg.norm(middle_tip - wrist) > np.linalg.norm(middle_base - wrist) * 1.4
        ring_extended = np.linalg.norm(ring_tip - wrist) > np.linalg.norm(ring_base - wrist) * 1.4
        pinky_extended = np.linalg.norm(pinky_tip - wrist) > np.linalg.norm(pinky_base - wrist) * 1.4
        
        # composite features for gesture matching
        fingers_extended = sum([index_extended, middle_extended, ring_extended, pinky_extended]) >= 4
        other_fingers_closed = not middle_extended and not ring_extended and not pinky_extended
        ring_pinky_closed = not ring_extended and not pinky_extended
        
        # finger spread for peace sign
        index_middle_spread = np.linalg.norm(index_tip - middle_tip) / hand_size

        # orientation checks
        thumb_pointing_up = (thumb_tip[1] < wrist[1])  # Y decreases upward in image coords
        index_pointing = (index_tip[1] < middle_tip[1])  # Index higher than middle
        

        return {
            "thumb_dist": thumb_dist,
            "index_dist": index_dist,
            "middle_dist": middle_dist,
            "ring_dist": ring_dist,
            "pinky_dist": pinky_dist,
            "finger_spread": finger_spread,
            "finger_curl": finger_curl,
            "thumb_extended": thumb_extended,
            "index_extended": index_extended,
            "middle_extended": middle_extended,
            "ring_extended": ring_extended,
            "pinky_extended": pinky_extended,
            "fingers_extended": fingers_extended,
            "other_fingers_closed": other_fingers_closed,
            "ring_pinky_closed": ring_pinky_closed,
            "index_middle_spread": index_middle_spread,
            "thumb_pointing_up": thumb_pointing_up,
            "index_pointing": index_pointing
        }
    
    def _calculate_spread(self, finger_tips):
        if len(finger_tips) < 2:
            return 0
        
        distances = []
        for i in range(len(finger_tips)):
            for j in range(i + 1, len(finger_tips)):
                distances.append(np.linalg.norm(finger_tips[i] - finger_tips[j]))
        
        spread = np.mean(distances) if distances else 0
        return min(1.0, spread / 0.5)
    
    def _calculate_curl(self, positions):
        # distance from finger tips to palm
        palm_center = positions[:9].mean(axis=0)
        finger_tips = positions[[4, 8, 12, 16, 20]]
        
        distances = np.array([
            np.linalg.norm(tip - palm_center) for tip in finger_tips
        ])
        
        # avg distance normalized
        avg_distance = distances.mean()
        curl = 1.0 - min(1.0, avg_distance / 0.3)
        return curl
    
    def _match_gesture(self, features, gesture_name, profile):
        score = 0
        max_score = 0
        
        for key, expected_value in profile.items():
            max_score += 1
            
            if key not in features:
                continue
            
            actual_value = features[key]
            
            # boolean features
            if isinstance(expected_value, bool):
                if actual_value == expected_value:
                    score += 1
            
            # range tuples (min/max)
            elif isinstance(expected_value, tuple) and len(expected_value) == 2:
                min_val, max_val = expected_value
                if min_val <= actual_value <= max_val:
                    score += 1
                else:
                    # partial credit based on distance
                    if actual_value < min_val:
                        distance = min_val - actual_value
                    else:
                        distance = actual_value - max_val
                    score += max(0, 1.0 - distance * 2)
        
        return score / max_score if max_score > 0 else 0