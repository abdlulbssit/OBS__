from flask import Flask, render_template, send_file, jsonify, request, make_response, send_from_directory
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

app = Flask(__name__)
CORS(app)

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Web Audio Recorder startup')

# Ensure recordings directory exists
if not os.path.exists('recordings'):
    os.makedirs('recordings')

def save_audio_file(audio_data):
    """Save the audio data to a file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join('recordings', secure_filename(filename))
        
        with open(filepath, 'wb') as f:
            f.write(audio_data)
            
        return filename
    except Exception as e:
        app.logger.error(f"Error saving audio file: {str(e)}")
        return None

def transcribe_audio(filepath):
    """Transcribe an audio file using OpenAI Whisper."""
    try:
        with open(filepath, 'rb') as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
            return transcript
    except Exception as e:
        app.logger.error(f"Transcription failed: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    return jsonify({'status': 'success'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file provided'})
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'})
            
        # Save the audio file
        filename = save_audio_file(audio_file.read())
        if not filename:
            return jsonify({'status': 'error', 'message': 'Failed to save audio file'})
            
        # Generate transcript
        filepath = os.path.join('recordings', filename)
        transcript = transcribe_audio(filepath)
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'transcript': transcript
        })
        
    except Exception as e:
        app.logger.error(f"Error in stop_recording: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory('recordings', filename, as_attachment=True)

@app.route('/recordings')
def list_recordings():
    recordings = []
    if os.path.exists('recordings'):
        recordings = [f for f in os.listdir('recordings') if f.endswith('.wav')]
    return jsonify(recordings)

if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 