import tkinter as tk
from tkinter import ttk, messagebox, Scale
import sounddevice as sd
import numpy as np
import wave
import threading
import time
import os
from datetime import datetime
import pyaudio

class AudioRecorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Audio Recorder")
        self.root.geometry("500x500")
        
        self.is_recording = False
        self.mic_data = []  # Separate storage for microphone data
        self.sys_data = []  # Separate storage for system audio data
        self.sample_rate = 44100
        self.chunk_size = 1024
        
        # Volume controls
        self.mic_volume = 1.0
        self.sys_volume = 0.3  # Reduced default system volume
        
        # Initialize audio devices
        try:
            # Initialize PyAudio
            self.p = pyaudio.PyAudio()
            
            # Get default microphone
            self.devices = sd.query_devices()
            self.default_mic = sd.default.device[0]
            self.mic_name = self.devices[self.default_mic]['name']
            
            # Find WASAPI loopback device
            self.wasapi_device = None
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if 'Stereo Mix' in device_info['name'] or 'What U Hear' in device_info['name']:
                    self.wasapi_device = i
                    break
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize audio devices: {str(e)}\nPlease check your audio settings.")
            self.root.destroy()
            return
        
        self.setup_gui()
        
    def setup_gui(self):
        style = ttk.Style()
        style.configure("Record.TButton", 
                       padding=10, 
                       font=('Helvetica', 12))
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Device info
        device_frame = ttk.LabelFrame(main_frame, text="Audio Devices", padding="10")
        device_frame.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(device_frame, 
                 text=f"Microphone: {self.mic_name}",
                 font=('Helvetica', 10)).grid(row=0, column=0, sticky=tk.W)
        
        system_audio_status = "System Audio: Ready" if self.wasapi_device is not None else "System Audio: Not available (enable Stereo Mix)"
        ttk.Label(device_frame,
                 text=system_audio_status,
                 font=('Helvetica', 10)).grid(row=1, column=0, sticky=tk.W)
        
        # Volume controls frame
        volume_frame = ttk.LabelFrame(main_frame, text="Volume Controls", padding="10")
        volume_frame.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Microphone volume
        ttk.Label(volume_frame, text="Microphone Volume:").grid(row=0, column=0, sticky=tk.W)
        self.mic_scale = Scale(volume_frame, from_=0, to=1, resolution=0.05,
                             orient=tk.HORIZONTAL, length=200,
                             command=self.update_mic_volume)
        self.mic_scale.set(self.mic_volume)
        self.mic_scale.grid(row=0, column=1, padx=5)
        
        # System audio volume
        ttk.Label(volume_frame, text="System Audio Volume:").grid(row=1, column=0, sticky=tk.W)
        self.sys_scale = Scale(volume_frame, from_=0, to=1, resolution=0.05,
                             orient=tk.HORIZONTAL, length=200,
                             command=self.update_sys_volume)
        self.sys_scale.set(self.sys_volume)
        self.sys_scale.grid(row=1, column=1, padx=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, 
                                    text="Ready to record...",
                                    font=('Helvetica', 12))
        self.status_label.grid(row=2, column=0, pady=20)
        
        # Record button
        self.record_button = ttk.Button(main_frame,
                                      text="Start Recording",
                                      command=self.toggle_recording,
                                      style="Record.TButton")
        self.record_button.grid(row=3, column=0, pady=20)
        
        # Time label
        self.time_label = ttk.Label(main_frame,
                                  text="00:00",
                                  font=('Helvetica', 24))
        self.time_label.grid(row=4, column=0, pady=20)
    
    def update_mic_volume(self, value):
        self.mic_volume = float(value)
    
    def update_sys_volume(self, value):
        self.sys_volume = float(value)
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        try:
            self.is_recording = True
            self.mic_data = []  # Clear microphone data
            self.sys_data = []  # Clear system audio data
            self.record_button.configure(text="Stop Recording")
            self.status_label.configure(text="Recording...")
            
            # Start recording threads
            self.mic_thread = threading.Thread(target=self.record_microphone)
            self.mic_thread.daemon = True
            self.mic_thread.start()
            
            if self.wasapi_device is not None:
                self.system_thread = threading.Thread(target=self.record_system_audio)
                self.system_thread.daemon = True
                self.system_thread.start()
            
            # Start timer thread
            self.timer_thread = threading.Thread(target=self.update_timer)
            self.timer_thread.daemon = True
            self.timer_thread.start()
            
            # Disable volume controls during recording
            self.mic_scale.configure(state=tk.DISABLED)
            self.sys_scale.configure(state=tk.DISABLED)
            
        except Exception as e:
            self.is_recording = False
            messagebox.showerror("Error", f"Failed to start recording: {str(e)}")
            self.status_label.configure(text="Ready to record...")
            
    def stop_recording(self):
        try:
            self.is_recording = False
            self.record_button.configure(text="Start Recording")
            self.status_label.configure(text="Saving recording...")
            
            # Wait for threads to finish with timeout
            if hasattr(self, 'mic_thread'):
                self.mic_thread.join(timeout=2.0)
            if hasattr(self, 'system_thread'):
                self.system_thread.join(timeout=2.0)
                
            # Re-enable volume controls
            self.mic_scale.configure(state=tk.NORMAL)
            self.sys_scale.configure(state=tk.NORMAL)
            
            self.save_recording()
            self.status_label.configure(text="Ready to record...")
            self.time_label.configure(text="00:00")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop recording: {str(e)}")
            self.status_label.configure(text="Ready to record...")
    
    def record_microphone(self):
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Microphone Status: {status}")
                if self.is_recording:
                    # Apply volume and ensure the data is the right shape
                    if len(indata.shape) == 2:
                        data = indata.copy() * self.mic_volume
                    else:
                        data = (indata.reshape(-1, 2)) * self.mic_volume
                    self.mic_data.append(data)

            with sd.InputStream(device=self.default_mic,
                              channels=2,
                              samplerate=self.sample_rate,
                              callback=audio_callback,
                              blocksize=self.chunk_size):
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as e:
            self.is_recording = False
            self.root.after(0, messagebox.showerror, "Error", f"Microphone recording failed: {str(e)}")
            self.root.after(0, self.status_label.configure, {"text": "Ready to record..."})
            self.root.after(0, self.record_button.configure, {"text": "Start Recording"})
    
    def record_system_audio(self):
        try:
            stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=2,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.wasapi_device,
                frames_per_buffer=self.chunk_size
            )
            
            while self.is_recording:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                if self.is_recording:
                    # Convert to numpy array and reshape to (samples, channels)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    audio_data = audio_data.reshape(-1, 2) * self.sys_volume
                    self.sys_data.append(audio_data)
                    
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            self.is_recording = False
            self.root.after(0, messagebox.showerror, "Error", f"System audio recording failed: {str(e)}")
            self.root.after(0, self.status_label.configure, {"text": "Ready to record..."})
            self.root.after(0, self.record_button.configure, {"text": "Start Recording"})
                
    def update_timer(self):
        start_time = time.time()
        while self.is_recording:
            elapsed_time = int(time.time() - start_time)
            minutes = elapsed_time // 60
            seconds = elapsed_time % 60
            self.time_label.configure(text=f"{minutes:02d}:{seconds:02d}")
            time.sleep(1)
            
    def save_recording(self):
        if not (self.mic_data or self.sys_data):
            messagebox.showerror("Error", "No audio data to save!")
            return
            
        try:
            # Create recordings directory if it doesn't exist
            if not os.path.exists("recordings"):
                os.makedirs("recordings")
                
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/recording_{timestamp}.wav"
            
            # Concatenate chunks
            if self.mic_data:
                mic_data = np.concatenate(self.mic_data, axis=0)
            else:
                mic_data = np.zeros((0, 2), dtype=np.float32)
                
            if self.sys_data:
                sys_data = np.concatenate(self.sys_data, axis=0)
            else:
                sys_data = np.zeros((0, 2), dtype=np.float32)
            
            # Ensure both arrays are the same length
            if len(mic_data) > 0 and len(sys_data) > 0:
                min_length = min(len(mic_data), len(sys_data))
                mic_data = mic_data[:min_length]
                sys_data = sys_data[:min_length]
                
                # Apply a small delay to system audio to reduce echo (about 50ms)
                delay_samples = int(0.05 * self.sample_rate)
                sys_data = np.pad(sys_data, ((delay_samples, 0), (0, 0)))[:-delay_samples]
                
                # Mix the streams (volumes already applied during recording)
                mixed_data = mic_data + sys_data
            elif len(mic_data) > 0:
                mixed_data = mic_data
            else:
                mixed_data = sys_data
            
            # Normalize audio
            max_val = np.max(np.abs(mixed_data))
            if max_val > 0:
                mixed_data = mixed_data / max_val
            audio_data = np.int16(mixed_data * 32767)
            
            # Save as WAV file
            with wave.open(filename, 'wb') as wave_file:
                wave_file.setnchannels(2)
                wave_file.setsampwidth(2)
                wave_file.setframerate(self.sample_rate)
                wave_file.writeframes(audio_data.tobytes())
                
            messagebox.showinfo("Success", f"Recording saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save recording: {str(e)}")
    
    def __del__(self):
        if hasattr(self, 'p'):
            self.p.terminate()
        
    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            messagebox.showerror("Error", f"Application error: {str(e)}")

if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.run() 