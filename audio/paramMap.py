"""
maps hand positions and gestures to audio parameters
"""

import numpy as np
from collections import deque
from config import Config


class ParameterMapper:
    # maps hand pose to continuous audio parameters
    
    def __init__(self):
        # initialize parameter mapper with smoothing buffers
        self.position_buffer = deque(maxlen=Config.SMOOTHING_WINDOW)
        self.gesture_buffer = deque(maxlen=5)
        
        # gesture to sound mapping
        self.gesture_to_sound = {
            "open_palm": 0,      # piano
            "fist": 1,           # strings
            "peace_sign": 2,     # synth
            "thumbs_up": 3,      # bells
            "pointing": 4,       # flute
            "unknown": 0         # default to first sound
        }
    
    def map_hand_to_parameters(self, hand, gesture):
        # convert hand position and gesture to audio parameters
        params = {}
                
        hand_center = hand.get_center()
        self.position_buffer.append(hand_center)
        
        # smooth positions
        if len(self.position_buffer) > 0:
            smoothed_pos = np.mean(list(self.position_buffer), axis=0)
        else:
            smoothed_pos = hand_center
        
        # clamp to active region
        x = np.clip(smoothed_pos[0], Config.HAND_X_MIN, Config.HAND_X_MAX)
        y = np.clip(smoothed_pos[1], Config.HAND_Y_MIN, Config.HAND_Y_MAX)
        
        # map hand height to volume
        volume = self._map_range(
            y,
            Config.HAND_Y_MIN,
            Config.HAND_Y_MAX,
            Config.VOLUME_MAX,  # Swap output range to invert mapping
            Config.VOLUME_MIN   # Higher Y (lower hand) = lower volume
        )
        params["volume"] = volume
        
        # map hand x position to pitch/filter frequency
        # feft = lower pitch, right = higher pitch
        pitch = self._map_range(
            x,
            Config.HAND_X_MIN,
            Config.HAND_X_MAX,
            Config.PITCH_MIN,
            Config.PITCH_MAX
        )
        params["pitch"] = pitch
        
        # map hand depth to reverb
        # further away = more reverb
        z = smoothed_pos[2]  # -1 to 1, closer to camera = smaller z
        reverb = self._map_range(
            z,
            -0.5,
            0.5,
            Config.REVERB_MIN,
            Config.REVERB_MAX
        )
        params["reverb"] = reverb
                
        if gesture.confidence > Config.GESTURE_CONFIDENCE_MIN:
            self.gesture_buffer.append(gesture.name)
            
            # map gesture to sound/instrument
            sound_id = self.gesture_to_sound.get(gesture.name, 0)
            params["sound"] = sound_id
                
        # calculate hand movement speed
        if len(self.position_buffer) >= 2:
            prev_pos = list(self.position_buffer)[-2]
            curr_pos = smoothed_pos
            movement_speed = np.linalg.norm(curr_pos - prev_pos)
            
            # map to tempo
            tempo = self._map_range(
                movement_speed,
                0.0,
                0.1,
                Config.TEMPO_MIN,
                Config.TEMPO_MAX
            )
            params["tempo"] = tempo
        else:
            params["tempo"] = (Config.TEMPO_MIN + Config.TEMPO_MAX) / 2
                
        # hand size controls effects intensity
        finger_tips = hand.get_finger_tips()
        hand_size = np.std(finger_tips) if len(finger_tips) > 0 else 0.1
        
        # normalize hand size (0 to 1)
        hand_size_normalized = min(1.0, hand_size / 0.2)
        params["hand_size"] = hand_size_normalized
        
        return params
    
    def _map_range(self, value, in_min, in_max, out_min, out_max):
        # map a value from input range to output range
        # clamp to input range
        value = np.clip(value, in_min, in_max)
        
        # normalize to 0-1
        if in_max == in_min:
            normalized = 0.5
        else:
            normalized = (value - in_min) / (in_max - in_min)
        
        # scale to output range
        return out_min + normalized * (out_max - out_min)
    
    def get_gesture_history(self):
        # get history of recent gestures
        return list(self.gesture_buffer)
    
    def get_most_common_gesture(self):
        # get the most common gesture from recent history
        if not self.gesture_buffer:
            return None
        
        from collections import Counter
        counts = Counter(self.gesture_buffer)
        return counts.most_common(1)[0][0]