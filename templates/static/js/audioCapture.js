class AudioCapture {
    constructor() {
        this.audioContext = null;
        this.microphoneStream = null;
        this.systemAudioStream = null;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.hasSystemAudio = false;
    }

    async initialize(micStream, sysStream) {
        try {
            // Store the streams
            this.microphoneStream = micStream;
            this.systemAudioStream = sysStream;
            this.hasSystemAudio = sysStream !== null;

            // Create audio context
            this.audioContext = new AudioContext();
            
            // Create audio sources
            const micSource = this.audioContext.createMediaStreamSource(this.microphoneStream);
            
            // Create gain nodes for volume control
            this.micGain = this.audioContext.createGain();
            this.sysGain = this.audioContext.createGain();
            
            // Create destination
            this.destination = this.audioContext.createMediaStreamDestination();
            
            // Connect microphone nodes
            micSource.connect(this.micGain);
            this.micGain.connect(this.destination);

            // Connect system audio if available
            if (this.hasSystemAudio) {
                const sysSource = this.audioContext.createMediaStreamSource(this.systemAudioStream);
                sysSource.connect(this.sysGain);
                this.sysGain.connect(this.destination);
            }

            return true;
        } catch (error) {
            console.error('Error initializing audio capture:', error);
            return false;
        }
    }

    setMicVolume(volume) {
        if (this.micGain) {
            this.micGain.gain.value = volume;
        }
    }

    setSystemVolume(volume) {
        if (this.sysGain && this.hasSystemAudio) {
            this.sysGain.gain.value = volume;
        }
    }

    startRecording() {
        if (!this.isRecording && this.destination) {
            this.audioChunks = [];
            this.mediaRecorder = new MediaRecorder(this.destination.stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.start(1000); // Collect data every second
            this.isRecording = true;
            return true;
        }
        return false;
    }

    stopRecording() {
        if (this.isRecording && this.mediaRecorder) {
            return new Promise((resolve) => {
                this.mediaRecorder.onstop = async () => {
                    // Convert audio chunks to WAV format
                    const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                    try {
                        // Convert to WAV using Web Audio API
                        const audioBuffer = await this.convertToWav(audioBlob);
                        const wavBlob = this.audioBufferToWav(audioBuffer);
                        resolve(wavBlob);
                    } catch (error) {
                        console.error('Error converting audio:', error);
                        resolve(null);
                    }
                };
                this.mediaRecorder.stop();
                this.isRecording = false;
            });
        }
        return Promise.resolve(null);
    }

    async convertToWav(blob) {
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
        return audioBuffer;
    }

    audioBufferToWav(audioBuffer) {
        const numOfChannels = audioBuffer.numberOfChannels;
        const sampleRate = audioBuffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;
        const bytesPerSample = bitDepth / 8;
        const blockAlign = numOfChannels * bytesPerSample;
        const byteRate = sampleRate * blockAlign;
        const dataSize = audioBuffer.length * blockAlign;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        // WAV header
        const writeString = (view, offset, string) => {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        };

        writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeString(view, 8, 'WAVE');
        writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, format, true);
        view.setUint16(22, numOfChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitDepth, true);
        writeString(view, 36, 'data');
        view.setUint32(40, dataSize, true);

        // Write audio data
        const offset = 44;
        const channels = [];
        for (let i = 0; i < numOfChannels; i++) {
            channels.push(audioBuffer.getChannelData(i));
        }

        let index = 0;
        const volume = 1;
        for (let i = 0; i < audioBuffer.length; i++) {
            for (let channel = 0; channel < numOfChannels; channel++) {
                const sample = channels[channel][i];
                const scaled = Math.max(-1, Math.min(1, sample)) * volume;
                const val = scaled < 0 ? scaled * 0x8000 : scaled * 0x7FFF;
                view.setInt16(offset + index, val, true);
                index += 2;
            }
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }

    cleanup() {
        if (this.microphoneStream) {
            this.microphoneStream.getTracks().forEach(track => track.stop());
        }
        if (this.systemAudioStream) {
            this.systemAudioStream.getTracks().forEach(track => track.stop());
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
} 