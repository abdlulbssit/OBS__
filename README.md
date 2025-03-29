# Web Audio Recorder

A modern web application for recording both microphone input and system audio simultaneously. Built with Python Flask and WebSocket for real-time communication.

## Features

- 🎤 Record microphone input
- 🔊 Capture system audio (requires Stereo Mix or What U Hear device)
- 🎚️ Adjustable volume controls for both audio sources
- ⏱️ Recording timer
- 📥 Download recordings as WAV files
- 🎯 Real-time device status monitoring
- 📋 List of recent recordings

## Requirements

- Python 3.8 or higher
- Virtual environment (recommended)
- Stereo Mix or What U Hear device enabled for system audio capture

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd web-audio-recorder
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web server:
```bash
python web_audio_recorder.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. The interface will show:
   - Device status (microphone and system audio availability)
   - Volume controls for both audio sources
   - Recording controls with timer
   - List of recent recordings

4. Click "Start Recording" to begin capturing audio. The button will pulse red while recording.

5. Click "Stop Recording" to end the recording. The file will be saved automatically.

6. Your recordings will appear in the "Recent Recordings" section, where you can download them.

## Troubleshooting

### System Audio Not Available

If system audio recording is not available:

1. Open Sound settings in Windows
2. Go to Recording devices
3. Right-click in the list and enable "Show Disabled Devices"
4. Find "Stereo Mix" or "What U Hear"
5. Right-click and enable the device
6. Set it as the default recording device

### Audio Quality Issues

- Adjust the volume sliders to find the optimal balance between microphone and system audio
- For best results, use headphones to prevent microphone feedback
- Default system audio volume is set to 30% to prevent echo

## License

This project is licensed under the MIT License - see the LICENSE file for details. 