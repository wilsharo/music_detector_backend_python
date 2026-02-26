# musicdetector_service.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import requests
import uuid
import soundfile as sf
import subprocess
import shutil
import json
from inaSpeechSegmenter import Segmenter
from pydub import AudioSegment
import assemblyai as aai
import numpy as np

# --- AssemblyAI API Key ---
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    print("WARNING: ASSEMBLYAI_API_KEY environment variable not set.")
aai.settings.api_key = ASSEMBLYAI_API_KEY

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

def generate_peak_data(audio_path, points=1000):
    """Generates simplified peak data for rendering a waveform."""
    print("Generating peak data...")
    try:
        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples())
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)
        samples = samples / (2**15)
        chunk_size = len(samples) // points
        peaks = []
        for i in range(points):
            chunk = samples[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) > 0:
                peak = float(np.max(np.abs(chunk)))
                peaks.append(round(peak, 4))
        return peaks
    except Exception as e:
        print(f"Peak generation failed: {e}")
        return None

def is_segment_in_music(segment_start, segment_end, music_segments):
    """Checks if a given time range overlaps with any music segments."""
    for music in music_segments:
        # Check for any overlap between segments
        if max(segment_start, music['start']) < min(segment_end, music['end']):
            return True
    return False

@app.route('/segment-audio', methods=['POST'])
def segment_audio():
    print("--- Full Analysis Request Received ---")
    original_temp_audio_path = None
    processed_temp_audio_path = None
    
    try:
        audio_source_for_processing = None

        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{os.path.splitext(audio_file.filename)[1]}")
            audio_file.save(original_temp_audio_path)
            audio_source_for_processing = original_temp_audio_path
        elif request.is_json and 'audio_url' in request.json:
            audio_url = request.json['audio_url']
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.tmp")
            with open(original_temp_audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            audio_source_for_processing = original_temp_audio_path
        else:
            return jsonify({"error": "No valid audio URL or file provided"}), 400

        # --- Step 1: Start AssemblyAI Transcription (FIX: Added disfluencies) ---
        print("Submitting to AssemblyAI...")
        config = aai.TranscriptionConfig(
            speaker_labels=True, 
            disfluencies=True  # Crucial for filler words!
        )
        transcriber = aai.Transcriber()
        transcript_future = transcriber.transcribe_async(audio_source_for_processing, config)

        # --- Step 2: Independent Music Segmentation ---
        audio_segment = AudioSegment.from_file(audio_source_for_processing)
        total_duration_seconds = len(audio_segment) / 1000.0
        
        processed_audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        processed_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        processed_audio_segment.export(processed_temp_audio_path, format="wav")
        
        seg_model = Segmenter()
        segmentation_results = seg_model(processed_temp_audio_path)

        final_music_segments = []
        for label, start, end in segmentation_results:
            # We treat any music segment >= 2s as valid music
            if label == 'music' and (end - start) >= 2.0:
                final_music_segments.append({
                    'start': round(start, 2), 
                    'end': round(end, 2)
                })
        
        print(f"Found {len(final_music_segments)} music segments.")

        # --- Step 3: Wait for AssemblyAI ---
        transcript = transcript_future.result()
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"AssemblyAI failed: {transcript.error}")

        # Filter Logic for Filler Words
        filler_targets = ['um', 'uh', 'er', 'ah', 'like', 'basically', 'actually']
        filler_word_segments = [
            {"word": w.text, "start": round(w.start / 1000, 2), "end": round(w.end / 1000, 2)}
            for w in transcript.words
            if w.text.lower().strip(".,?!") in filler_targets
            and not is_segment_in_music(w.start/1000, w.end/1000, final_music_segments)
        ]

        # Speaker Segments
        speaker_segments = [
            {
                "speaker": u.speaker, 
                "text": u.text, 
                "start": round(u.start / 1000, 2), 
                "end": round(u.end / 1000, 2)
            }
            for u in (transcript.utterances or [])
            if not is_segment_in_music(u.start/1000, u.end/1000, final_music_segments)
        ]

        # Calculate Dead Air (Gaps between utterances)
        dead_air_segments = []
        if transcript.utterances:
            last_end = 0
            for u in transcript.utterances:
                start_sec = u.start / 1000
                if (start_sec - last_end) >= 2.0: # 2+ seconds of silence
                    # Only add if this gap isn't actually just music
                    if not is_segment_in_music(last_end, start_sec, final_music_segments):
                        dead_air_segments.append({
                            "start": round(last_end, 2),
                            "end": round(start_sec, 2),
                            "duration": round(start_sec - last_end, 2)
                        })
                last_end = u.end / 1000

        # --- Step 4: Final Response ---
        peak_data = generate_peak_data(processed_temp_audio_path)

        return jsonify({
            "duration": round(total_duration_seconds, 2),
            "music_segments": final_music_segments,
            "filler_word_segments": filler_word_segments,
            "dead_air_segments": dead_air_segments,
            "speaker_segments": speaker_segments,
            "peak_data": peak_data
        }), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        for p in [original_temp_audio_path, processed_temp_audio_path]:
            if p and os.path.exists(p): os.remove(p)

# ... (render_audio remains the same)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)
