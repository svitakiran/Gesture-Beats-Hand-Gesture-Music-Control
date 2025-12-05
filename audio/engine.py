"""
communicates with external audio synthesis (Sonic Pi)
"""

from pythonosc import udp_client
import time
import math


class Engine:
    """
    Sends audio parameters to an external audio engine via OSC.
    Make sure to run Sonic Pi on localhost:4560 before starting this program.
    """
    
    @staticmethod
    def frequency_to_midi(frequency):
        """
        Convert frequency (Hz) to MIDI note number.
        A4 (440 Hz) = MIDI note 69
        """
        if frequency <= 0:
            return 60  # default to middle C
        
        # MIDI note = 69 + 12 * log2(f/440)
        midi_note = 69 + 12 * math.log2(frequency / 440.0)
        return int(round(midi_note))

    def __init__(self, use_osc=True, host="127.0.0.1", port=4560):
        # initialize audio engine
        self.use_osc = use_osc
        self.last_update_time = time.time()
        self.update_rate = 30
        self.min_update_interval = 1.0 / self.update_rate
        
        if use_osc:
            try:
                self.client = udp_client.SimpleUDPClient(host, port)
                print(f"Audio engine connected to OSC {host}:{port}")
            except Exception as e:
                print(f"Failed to connect to OSC server: {e}")
                print(f"  Make sure Sonic Pi is running on {host}:{port}")
                self.client = None
    
    def update(self, parameters):
        """
        Send updated audio parameters to audio engine.
        
        Args:
            parameters: Dict of {parameter_name: value}
        """
        # Rate limit updates to avoid overwhelming OSC
        current_time = time.time()
        if current_time - self.last_update_time < self.min_update_interval:
            return
        
        self.last_update_time = current_time
        
        if not self.use_osc or self.client is None:
            return
        
        # Send each parameter as an OSC message
        try:
            # Volume (0-1)
            if "volume" in parameters:
                self.client.send_message("/volume", float(parameters["volume"]))
            
            # Pitch (Hz)
            if "pitch" in parameters:
                pitch_hz = float(parameters["pitch"])
                self.client.send_message("/pitch", pitch_hz)
                
                # convert pitch to MIDI note
                midi_note = self.frequency_to_midi(pitch_hz)
                # send
                self.client.send_message("/note", midi_note)
            
            # Tempo (BPM)
            if "tempo" in parameters:
                self.client.send_message("/tempo", float(parameters["tempo"]))
            
            # Reverb (0-1)
            if "reverb" in parameters:
                self.client.send_message("/reverb", float(parameters["reverb"]))
            
            # Sound/instrument (int)
            if "sound" in parameters:
                self.client.send_message("/sound", int(parameters["sound"]))
            
            # Hand size (0-1)
            if "hand_size" in parameters:
                self.client.send_message("/hand_size", float(parameters["hand_size"]))
        
        except Exception as e:
            print(f"Error sending OSC message: {e}")
    
    def send_test_signal(self):
        """Send a test signal to verify OSC connection is working."""
        if not self.use_osc or self.client is None:
            return
        
        try:
            self.client.send_message("/test", 1)
            print("✓ Test signal sent to audio engine")
        except Exception as e:
            print(f"✗ Failed to send test signal: {e}")
    
    def close(self):
        """Clean up (OSC doesn't need explicit cleanup)."""
        pass


class DummyAudioEngine:
    """
    Dummy audio engine for testing when no real audio engine is available.
    Just prints parameters instead of sending OSC.
    """
    
    def __init__(self):
        print("Using dummy audio engine (no real sound output)")
        self.last_params = {}
    
    def update(self, parameters):
        """Print parameters instead of sending them."""
        # Only print if parameters have changed significantly
        changed = False
        for key, value in parameters.items():
            if key not in self.last_params or abs(self.last_params[key] - value) > 0.05:
                changed = True
                break
        
        if changed:
            print(f"Audio params: Vol={parameters.get('volume', 0):.2f}, "
                  f"Pitch={parameters.get('pitch', 0):.0f}Hz, "
                  f"Sound={parameters.get('sound', 0)}, "
                  f"Reverb={parameters.get('reverb', 0):.2f}")
            self.last_params = parameters.copy()
    
    def close(self):
        pass