# musicdetector_service.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from inaSpeechSegmenter import Segmenter
import requests
import uuid
import soundfile as sf
import subprocess
import shutil
import numpy as np

# --- (TensorFlow import is the same) ---
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    print("ERROR: TensorFlow is not installed.")
    exit(1)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

@app.route('/segment-audio', methods=['POST'])
def segment_audio():
    print("--- Segment Audio Request Received (True Streaming Mode) ---")
    original_temp_audio_path = None
    
    try:
        # Initialize a fresh Segmenter for each request.
        print("Initializing inaSpeechSegmenter for new request...")
        seg = Segmenter()
        print("Segmenter initialized.")

        # --- (File upload/download logic is the same) ---
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{os.path.splitext(audio_file.filename)[1]}")
            audio_file.save(original_temp_audio_path)
        elif request.is_json and 'audio_url' in request.json:
            audio_url = request.json['audio_url']
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.tmp")
            with open(original_temp_audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        else:
            return jsonify({"error": "No valid audio URL or file provided"}), 400

        with sf.SoundFile(original_temp_audio_path) as f:
            total_duration_seconds = len(f) / f.samplerate
        
        # --- New Streaming Logic ---
        SAMPLE_RATE = 16000  # The rate the model expects
        CHUNK_DURATION_SECONDS = 60 # Process 1 minute of audio at a time
        CHUNK_SIZE = CHUNK_DURATION_SECONDS * SAMPLE_RATE * 2 # 2 bytes per sample (16-bit)

        # This FFmpeg command reads the input file and pipes out raw audio data
        ffmpeg_command = [
            'ffmpeg',
            '-i', original_temp_audio_path,
            '-f', 's16le',          # Format: signed 16-bit little-endian
            '-ac', '1',             # Channels: 1 (mono)
            '-ar', str(SAMPLE_RATE),# Sample rate: 16kHz
            '-'                     # Output to stdout
        ]
        
        print("Starting FFmpeg streaming process...")
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        all_segments = []
        current_offset = 0.0

        while True:
            # Read a chunk of raw audio data from FFmpeg's output
            raw_audio_chunk = ffmpeg_process.stdout.read(CHUNK_SIZE)
            if not raw_audio_chunk:
                break # End of stream

            # Convert the raw bytes to a NumPy array
            audio_array = np.frombuffer(raw_audio_chunk, dtype=np.int16)

            # The segmenter can process a NumPy array directly
            print(f"Processing audio chunk from {current_offset:.2f}s...")
            chunk_segmentation = seg.segment(audio_array, sr=SAMPLE_RATE)
            
            # Adjust timestamps and add to master list
            for label, start, end in chunk_segmentation:
                all_segments.append((label, start + current_offset, end + current_offset))
            
            # Update the offset for the next chunk
            current_offset += CHUNK_DURATION_SECONDS

        # Wait for FFmpeg to finish and check for errors
        ffmpeg_process.wait()
        if ffmpeg_process.returncode != 0:
            error_output = ffmpeg_process.stderr.read().decode()
            raise Exception(f"FFmpeg streaming failed: {error_output}")

        # --- (Post-processing logic for merging segments is the same) ---
        final_music_segments = []
        current_music_block = None
        for label, start, end in all_segments:
            if label == 'music':
                if (end - start) >= 3:
                    if current_music_block is None:
                        current_music_block = {'start': start, 'end': end}
                    else:
                        if start - current_music_block['end'] < 2.0:
                            current_music_block['end'] = end
                        else:
                            final_music_segments.append(current_music_block)
                            current_music_block = {'start': start, 'end': end}
            else:
                if current_music_block is not None:
                    final_music_segments.append(current_music_block)
                    current_music_block = None
        if current_music_block is not None:
            final_music_segments.append(current_music_block)

        print(f"Found {len(final_music_segments)} merged music segments.")
        return jsonify({
            "duration": round(total_duration_seconds, 2),
            "segments": final_music_segments
        }), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500
    finally:
        # --- (Cleanup logic is the same) ---
        if original_temp_audio_path and os.path.exists(original_temp_audio_path):
            os.remove(original_temp_audio_path)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)