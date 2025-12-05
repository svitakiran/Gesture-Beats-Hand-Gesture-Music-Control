import cv2
import numpy as np
import time
from vision.detector import Detector
from vision.gestures import GestureClassifier
from audio.paramMap import ParameterMapper
from audio.engine import Engine, DummyAudioEngine
from config import Config


class musicGesture:
    
    def __init__(self, use_audio=True):
        print("Initializing...")
        
        # vision components
        self.hand_detector = Detector()
        self.gesture_classifier = GestureClassifier()
        self.param_mapper = ParameterMapper()
        
        # audio engine
        if use_audio:
            self.audio_engine = Engine(use_osc=True)
        else:
            self.audio_engine = DummyAudioEngine()
        
        # camera setup
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, Config.FPS)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # FPS tracking
        self.fps_clock = time.time()
        self.frame_count = 0
        self.current_fps = 0
        
        print(" Gesture Beats initialized successfully")
        print("  Press 'q' to quit")
        print("  Press 'h' to toggle hand skeleton display")
        print("  Press 'd' to toggle debug info")
        print("  Press 'f' to toggle FPS display")
        print()
    
    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read from camera")
                    break
                
                frame = cv2.flip(frame, 1)
                                
                # detect hands
                hands = self.hand_detector.detect(frame)
                
                # classify gestures
                gestures = [self.gesture_classifier.classify(h) for h in hands]
                
                # map to audio parameters
                all_params = {}
                for hand, gesture in zip(hands, gestures):
                    params = self.param_mapper.map_hand_to_parameters(hand, gesture)
                    all_params = params
                
                if hands:
                    self.audio_engine.update(all_params)
                
                
                # hand skeletons
                frame = self.hand_detector.draw_hands(frame, hands)
                
                # hand info (labels, centers)
                frame = self.hand_detector.draw_hand_info(frame, hands)
                
                # debug info
                if Config.SHOW_DEBUG_INFO:
                    frame = self._draw_debug_info(frame, hands, gestures, all_params)
                
                # draw FPS
                if Config.SHOW_FPS:
                    frame = self._draw_fps(frame)
                
                # Display frame
                cv2.imshow("Music Gestures", frame)
                
                # handling inputs
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('h'):
                    Config.SHOW_HAND_SKELETON = not Config.SHOW_HAND_SKELETON
                    print(f"Hand skeleton: {'ON' if Config.SHOW_HAND_SKELETON else 'OFF'}")
                elif key == ord('d'):
                    Config.SHOW_DEBUG_INFO = not Config.SHOW_DEBUG_INFO
                    print(f"Debug info: {'ON' if Config.SHOW_DEBUG_INFO else 'OFF'}")
                elif key == ord('f'):
                    Config.SHOW_FPS = not Config.SHOW_FPS
                    print(f"FPS display: {'ON' if Config.SHOW_FPS else 'OFF'}")
                
                # update FPS counter
                self.frame_count += 1
                if time.time() - self.fps_clock > 1:
                    self.current_fps = self.frame_count
                    self.frame_count = 0
                    self.fps_clock = time.time()
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            self.cleanup()
    
    def _draw_debug_info(self, frame, hands, gestures, params):
        height, width = frame.shape[:2]
        
        # draw param vals
        y_offset = 30
        info_lines = [
            f"Hands detected: {len(hands)}",
            f"Volume: {params.get('volume', 0):.2f}",
            f"Pitch: {params.get('pitch', 0):.0f} Hz",
            f"Tempo: {params.get('tempo', 0):.0f} BPM",
            f"Reverb: {params.get('reverb', 0):.2f}",
            f"Sound: {params.get('sound', 0)}",
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(
                frame,
                line,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 0),
                1
            )
        
        # draw gesture info
        for i, (hand, gesture) in enumerate(zip(hands, gestures)):
            y_pos = height - 60 + i * 25
            text = f"{hand.label} hand: {gesture.name} ({gesture.confidence:.2f})"
            cv2.putText(
                frame,
                text,
                (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if gesture.confidence > 0.6 else (0, 100, 0),
                2
            )
        
        # draw controls
        cv2.putText(
            frame,
            "q:quit h:skeleton d:debug f:fps",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1
        )
        
        return frame
    
    def _draw_fps(self, frame):
        cv2.putText(
            frame,
            f"FPS: {self.current_fps}",
            (frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        return frame
    
    def cleanup(self):
        print("cleaning up...")
        self.cap.release()
        self.audio_engine.close()
        cv2.destroyAllWindows()
        print("completed cleanup")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Gesture Beats: Hand Gesture Music Control")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio engine (for testing)")
    args = parser.parse_args()
    
    app = musicGesture(use_audio=not args.no_audio)
    app.run()


if __name__ == "__main__":
    main()