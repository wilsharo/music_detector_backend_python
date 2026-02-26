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
    """
    Checks if a segment is buried in music.
    Returns True only if > 50% of the segment is covered by music.
    """
    segment_duration = segment_end - segment_start
    if segment_duration <= 0:
        return False

    for music in music_segments:
        # Calculate intersection
        overlap_start = max(segment_start, music['start'])
        overlap_end = min(segment_end, music['end'])
        
        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
            # If the speaker is mostly covered by music, filter it out
            if (overlap_duration / segment_duration) > 0.5:
                return True
    return False

@app.route('/segment-audio', methods=['POST'])
def segment_audio():
    print("--- Starting Single-Pass Analysis ---")
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
            return jsonify({"error": "No valid audio provided"}), 400

        # --- Step 1: AssemblyAI (REQUEST DISFLUENCIES) ---
        print("Submitting to AssemblyAI...")
        config = aai.TranscriptionConfig(
            speaker_labels=True, 
            disfluencies=True 
        )
        transcriber = aai.Transcriber()
        transcript_future = transcriber.transcribe_async(audio_source_for_processing, config)

        # --- Step 2: Music Segmentation ---
        audio_segment = AudioSegment.from_file(audio_source_for_processing)
        total_duration_seconds = len(audio_segment) / 1000.0
        
        # Standardize audio for inaSpeechSegmenter
        processed_audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        processed_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        processed_audio_segment.export(processed_temp_audio_path, format="wav")
        
        seg_model = Segmenter()
        segmentation_results = seg_model(processed_temp_audio_path)

        final_music_segments = []
        for label, start, end in segmentation_results:
            if label == 'music' and (end - start) >= 2.0:
                final_music_segments.append({'start': round(start, 2), 'end': round(end, 2)})
        
        print(f"Music Detection: {len(final_music_segments)} segments found.")

        # --- Step 3: Processing AI Results ---
        transcript = transcript_future.result()
        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"AI Transcription Error: {transcript.error}")

        # Ensure we have data to iterate over
        all_utterances = transcript.utterances if transcript.utterances is not None else []
        all_words = transcript.words if transcript.words is not None else []

        # Speaker Segments (With Overlap Protection)
        speaker_segments = [
            {
                "speaker": u.speaker, 
                "text": u.text, 
                "start": round(u.start / 1000, 2), 
                "end": round(u.end / 1000, 2)
            }
            for u in all_utterances
            if not is_segment_in_music(u.start/1000, u.end/1000, final_music_segments)
        ]

        # Filler Words
        filler_targets = ['um', 'uh', 'er', 'ah', 'like', 'basically', 'actually']
        filler_word_segments = [
            {"word": w.text, "start": round(w.start / 1000, 2), "end": round(w.end / 1000, 2)}
            for w in all_words
            if w.text.lower().strip(".,?!") in filler_targets
            and not is_segment_in_music(w.start/1000, w.end/1000, final_music_segments)
        ]

        # Dead Air Calculation
        dead_air_segments = []
        last_end = 0
        for u in all_utterances:
            start_sec = u.start / 1000
            if (start_sec - last_end) >= 2.0:
                # Only mark as dead air if it's not a music block
                if not is_segment_in_music(last_end, start_sec, final_music_segments):
                    dead_air_segments.append({
                        "start": round(last_end, 2),
                        "end": round(start_sec, 2),
                        "duration": round(start_sec - last_end, 2)
                    })
            last_end = u.end / 1000

        # --- Step 4: Final Payload ---
        peak_data = generate_peak_data(processed_temp_audio_path)

        print(f"Done. Returning {len(speaker_segments)} speaker segments.")
        return jsonify({
            "duration": round(total_duration_seconds, 2),
            "music_segments": final_music_segments,
            "filler_word_segments": filler_word_segments,
            "dead_air_segments": dead_air_segments,
            "speaker_segments": speaker_segments,
            "peak_data": peak_data
        }), 200

    except Exception as e:
        print(f"FAILED: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        for p in [original_temp_audio_path, processed_temp_audio_path]:
            if p and os.path.exists(p): os.remove(p)

@app.route('/render-audio', methods=['POST'])
def render_audio():
    print("--- Render Audio Request Received ---")
    data = request.get_json()
    audio_source = data.get('audio_url')
    playback_sequence = data.get('playback_sequence')

    if not audio_source or not playback_sequence:
        return jsonify({"error": "Missing parameters"}), 400

    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"exported_{uuid.uuid4()}.mp3")

    try:
        filter_complex_parts = []
        input_mappings = []
        for i, clip in enumerate(playback_sequence):
            start = clip['sourceStart']
            end = clip['sourceEnd']
            filter_complex_parts.append(f"[0:a]atrim={start}:{end},asetpts=PTS-STARTPTS[a{i}]")
            input_mappings.append(f"[a{i}]")

        concat_filter = "".join(input_mappings) + f"concat=n={len(playback_sequence)}:v=0:a=1[outa]"
        filter_graph = ";".join(filter_complex_parts) + ";" + concat_filter

        ffmpeg_command = [
            'ffmpeg', '-i', audio_source, '-filter_complex', filter_graph,
            '-map', '[outa]', output_path
        ]

        subprocess.run(ffmpeg_command, check=True)
        return send_file(output_path, as_attachment=True, mimetype='audio/mpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_dir and os.path.exists(temp_dir): shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)
