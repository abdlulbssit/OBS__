import customtkinter as ctk
from PIL import Image
import sounddevice as sd
import numpy as np
import wave
import threading
import time
import os
from datetime import datetime
import pyaudio

class ModernAudioRecorder:
    def __init__(self):
        # Set theme and color
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("Modern Audio Recorder")
        self.root.geometry("800x600")
        
        self.is_recording = False
        self.mic_data = []
        self.sys_data = []
        self.sample_rate = 44100
        self.chunk_size = 1024
        
        # Volume controls
        self.mic_volume = 1.0
        self.sys_volume = 0.3
        
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
            self.show_error("Device Error", f"Failed to initialize audio devices: {str(e)}\nPlease check your audio settings.")
            self.root.destroy()
            return
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        title_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkLabel(
            title_frame,
            text="Audio Recorder",
            font=ctk.CTkFont(size=24, weight="bold")
        ).pack()
        
        # Device Info Panel
        device_frame = ctk.CTkFrame(self.main_frame)
        device_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            device_frame,
            text="Audio Devices",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Microphone info with icon
        mic_info = ctk.CTkFrame(device_frame, fg_color="transparent")
        mic_info.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            mic_info,
            text="🎤",  # Microphone emoji
            font=ctk.CTkFont(size=20)
        ).pack(side="left", padx=5)
        
        ctk.CTkLabel(
            mic_info,
            text=f"Microphone: {self.mic_name}",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        # System audio info with icon
        sys_info = ctk.CTkFrame(device_frame, fg_color="transparent")
        sys_info.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            sys_info,
            text="🔊",  # Speaker emoji
            font=ctk.CTkFont(size=20)
        ).pack(side="left", padx=5)
        
        system_status = "System Audio: Ready" if self.wasapi_device is not None else "System Audio: Not available (enable Stereo Mix)"
        ctk.CTkLabel(
            sys_info,
            text=system_status,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        # Volume Controls Panel
        volume_frame = ctk.CTkFrame(self.main_frame)
        volume_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            volume_frame,
            text="Volume Controls",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Microphone volume slider
        mic_volume = ctk.CTkFrame(volume_frame, fg_color="transparent")
        mic_volume.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            mic_volume,
            text="Microphone",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        self.mic_slider = ctk.CTkSlider(
            mic_volume,
            from_=0,
            to=1,
            number_of_steps=20,
            command=self.update_mic_volume
        )
        self.mic_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.mic_slider.set(self.mic_volume)
        
        self.mic_value_label = ctk.CTkLabel(
            mic_volume,
            text=f"{int(self.mic_volume * 100)}%",
            font=ctk.CTkFont(size=12)
        )
        self.mic_value_label.pack(side="left", padx=5)
        
        # System audio volume slider
        sys_volume = ctk.CTkFrame(volume_frame, fg_color="transparent")
        sys_volume.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            sys_volume,
            text="System Audio",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        self.sys_slider = ctk.CTkSlider(
            sys_volume,
            from_=0,
            to=1,
            number_of_steps=20,
            command=self.update_sys_volume
        )
        self.sys_slider.pack(side="left", fill="x", expand=True, padx=10)
        self.sys_slider.set(self.sys_volume)
        
        self.sys_value_label = ctk.CTkLabel(
            sys_volume,
            text=f"{int(self.sys_volume * 100)}%",
            font=ctk.CTkFont(size=12)
        )
        self.sys_value_label.pack(side="left", padx=5)
        
        # Recording Controls
        control_frame = ctk.CTkFrame(self.main_frame)
        control_frame.pack(fill="x", padx=20, pady=20)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            control_frame,
            text="Ready to record...",
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(pady=10)
        
        # Timer label
        self.time_label = ctk.CTkLabel(
            control_frame,
            text="00:00",
            font=ctk.CTkFont(size=36, weight="bold")
        )
        self.time_label.pack(pady=10)
        
        # Record button
        self.record_button = ctk.CTkButton(
            control_frame,
            text="Start Recording",
            command=self.toggle_recording,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.record_button.pack(pady=20)
        
        # Recent recordings panel
        recordings_frame = ctk.CTkFrame(self.main_frame)
        recordings_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            recordings_frame,
            text="Recent Recordings",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        self.recordings_list = ctk.CTkTextbox(
            recordings_frame,
            height=100,
            font=ctk.CTkFont(size=12)
        )
        self.recordings_list.pack(fill="x", padx=20, pady=10)
        
        # Update recent recordings list
        self.update_recordings_list()
    
    def update_recordings_list(self):
        if not os.path.exists("recordings"):
            return
            
        recordings = sorted(
            [f for f in os.listdir("recordings") if f.endswith('.wav')],
            key=lambda x: os.path.getmtime(os.path.join("recordings", x)),
            reverse=True
        )[:5]  # Show only 5 most recent recordings
        
        self.recordings_list.delete("1.0", "end")
        if recordings:
            for recording in recordings:
                timestamp = datetime.fromtimestamp(
                    os.path.getmtime(os.path.join("recordings", recording))
                ).strftime("%Y-%m-%d %H:%M:%S")
                self.recordings_list.insert("end", f"📝 {recording} - {timestamp}\n")
        else:
            self.recordings_list.insert("end", "No recordings yet...")
    
    def update_mic_volume(self, value):
        self.mic_volume = float(value)
        self.mic_value_label.configure(text=f"{int(self.mic_volume * 100)}%")
    
    def update_sys_volume(self, value):
        self.sys_volume = float(value)
        self.sys_value_label.configure(text=f"{int(self.sys_volume * 100)}%")
    
    def show_error(self, title, message):
        ctk.CTkMessagebox(
            title=title,
            message=message,
            icon="cancel"
        )
    
    def show_info(self, title, message):
        ctk.CTkMessagebox(
            title=title,
            message=message,
            icon="info"
        )
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        try:
            self.is_recording = True
            self.mic_data = []
            self.sys_data = []
            self.record_button.configure(
                text="Stop Recording",
                fg_color="#c42b1c",  # Red color for recording state
                hover_color="#d44942"
            )
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
            
            # Disable volume controls
            self.mic_slider.configure(state="disabled")
            self.sys_slider.configure(state="disabled")
            
        except Exception as e:
            self.is_recording = False
            self.show_error("Recording Error", f"Failed to start recording: {str(e)}")
            self.status_label.configure(text="Ready to record...")
            
    def stop_recording(self):
        try:
            self.is_recording = False
            self.record_button.configure(
                text="Start Recording",
                fg_color=["#3B8ED0", "#1F6AA5"],  # Default blue color
                hover_color=["#36719F", "#144870"]
            )
            self.status_label.configure(text="Saving recording...")
            
            # Wait for threads to finish
            if hasattr(self, 'mic_thread'):
                self.mic_thread.join(timeout=2.0)
            if hasattr(self, 'system_thread'):
                self.system_thread.join(timeout=2.0)
                
            # Re-enable volume controls
            self.mic_slider.configure(state="normal")
            self.sys_slider.configure(state="normal")
            
            self.save_recording()
            self.status_label.configure(text="Ready to record...")
            self.time_label.configure(text="00:00")
            
            # Update recordings list
            self.update_recordings_list()
            
        except Exception as e:
            self.show_error("Recording Error", f"Failed to stop recording: {str(e)}")
            self.status_label.configure(text="Ready to record...")
    
    def record_microphone(self):
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Microphone Status: {status}")
                if self.is_recording:
                    if len(indata.shape) == 2:
                        data = indata.copy() * self.mic_volume
                    else:
                        data = (indata.reshape(-1, 2)) * self.mic_volume
                    self.mic_data.append(data)

            with sd.InputStream(
                device=self.default_mic,
                channels=2,
                samplerate=self.sample_rate,
                callback=audio_callback,
                blocksize=self.chunk_size
            ):
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as e:
            self.is_recording = False
            self.root.after(0, self.show_error, "Recording Error", f"Microphone recording failed: {str(e)}")
            self.root.after(0, self.status_label.configure, text="Ready to record...")
            self.root.after(0, self.record_button.configure, text="Start Recording")
    
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
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    audio_data = audio_data.reshape(-1, 2) * self.sys_volume
                    self.sys_data.append(audio_data)
                    
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            self.is_recording = False
            self.root.after(0, self.show_error, "Recording Error", f"System audio recording failed: {str(e)}")
            self.root.after(0, self.status_label.configure, text="Ready to record...")
            self.root.after(0, self.record_button.configure, text="Start Recording")
                
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
            self.show_error("Recording Error", "No audio data to save!")
            return
            
        try:
            if not os.path.exists("recordings"):
                os.makedirs("recordings")
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/recording_{timestamp}.wav"
            
            if self.mic_data:
                mic_data = np.concatenate(self.mic_data, axis=0)
            else:
                mic_data = np.zeros((0, 2), dtype=np.float32)
                
            if self.sys_data:
                sys_data = np.concatenate(self.sys_data, axis=0)
            else:
                sys_data = np.zeros((0, 2), dtype=np.float32)
            
            if len(mic_data) > 0 and len(sys_data) > 0:
                min_length = min(len(mic_data), len(sys_data))
                mic_data = mic_data[:min_length]
                sys_data = sys_data[:min_length]
                
                delay_samples = int(0.05 * self.sample_rate)
                sys_data = np.pad(sys_data, ((delay_samples, 0), (0, 0)))[:-delay_samples]
                
                mixed_data = mic_data + sys_data
            elif len(mic_data) > 0:
                mixed_data = mic_data
            else:
                mixed_data = sys_data
            
            max_val = np.max(np.abs(mixed_data))
            if max_val > 0:
                mixed_data = mixed_data / max_val
            audio_data = np.int16(mixed_data * 32767)
            
            with wave.open(filename, 'wb') as wave_file:
                wave_file.setnchannels(2)
                wave_file.setsampwidth(2)
                wave_file.setframerate(self.sample_rate)
                wave_file.writeframes(audio_data.tobytes())
                
            self.show_info("Success", f"Recording saved as {filename}")
        except Exception as e:
            self.show_error("Save Error", f"Failed to save recording: {str(e)}")
    
    def __del__(self):
        if hasattr(self, 'p'):
            self.p.terminate()
        
    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            self.show_error("Application Error", f"Application error: {str(e)}")

if __name__ == "__main__":
    recorder = ModernAudioRecorder()
    recorder.run() 