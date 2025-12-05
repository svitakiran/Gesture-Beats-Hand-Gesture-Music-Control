"""
configuration file
"""

class Config:
    # camera settings
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    
    # hand detection settings
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    MAX_NUM_HANDS = 2
    
    # gesture smoothing
    SMOOTHING_WINDOW = 5
    
    # audio osc settings
    OSC_HOST = "127.0.0.1"
    OSC_PORT = 4560
    
    # gesture thresholds
    PALM_OPEN_THRESHOLD = 0.7
    FIST_THRESHOLD = 0.3
    GESTURE_CONFIDENCE_MIN = 0.6
    
    # audio parameter ranges
    VOLUME_MIN = 0.0
    VOLUME_MAX = 1.0
    
    PITCH_MIN = 50
    PITCH_MAX = 2000
    
    TEMPO_MIN = 60
    TEMPO_MAX = 200
    
    REVERB_MIN = 0.0
    REVERB_MAX = 1.0
    
    # hand position ranges
    HAND_X_MIN = 0.1
    HAND_X_MAX = 0.9
    HAND_Y_MIN = 0.1
    HAND_Y_MAX = 0.9
    
    # visualization
    SHOW_DEBUG_INFO = True
    SHOW_HAND_SKELETON = True
    SHOW_FPS = True