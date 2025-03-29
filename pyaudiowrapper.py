import pyaudio
import numpy as np
from ctypes import windll, c_ubyte, c_size_t, c_float
import win32api
import win32gui

class PyAudioWrapper:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.chunk_size = 1024
        self.format = pyaudio.paFloat32
        self.channels = 2
        self.rate = 44100
        
    def init_wasapi_loopback(self):
        """Initialize WASAPI loopback capture for system audio"""
        try:
            # Find the loopback device
            wasapi_info = None
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if device_info['name'].lower().find('wasapi') != -1:
                    wasapi_info = device_info
                    break
            
            if wasapi_info is None:
                raise Exception("No WASAPI loopback device found")
            
            # Open the stream
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=wasapi_info['index'],
                stream_callback=None
            )
            
            self.stream.start_stream()
            
        except Exception as e:
            raise Exception(f"Failed to initialize WASAPI loopback: {str(e)}")
    
    def read_stream(self):
        """Read audio data from the stream"""
        try:
            if self.stream and self.stream.is_active():
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                return np.frombuffer(data, dtype=np.float32)
            return None
        except Exception as e:
            print(f"Error reading stream: {str(e)}")
            return None
    
    def __del__(self):
        """Cleanup resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate() 