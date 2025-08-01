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

# --- NEW: Function to generate waveform peak data ---
def generate_peak_data(audio_path, points=1000):
    """
    Generates a simplified array of peak data for rendering a waveform.
    """
    print("Generating peak data for waveform...")
    try:
        audio = AudioSegment.from_file(audio_path)
        samples = np.array(audio.get_array_of_samples())
        
        # If stereo, convert to mono by averaging channels
        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).mean(axis=1)

        # Normalize to -1.0 to 1.0
        samples = samples / (2**15)

        chunk_size = len(samples) // points
        peaks = []
        for i in range(points):
            chunk = samples[i * chunk_size : (i + 1) * chunk_size]
            if len(chunk) > 0:
                peak = float(np.max(np.abs(chunk)))
                peaks.append(round(peak, 4))
        
        print(f"Peak data generated with {len(peaks)} points.")
        return peaks
    except Exception as e:
        print(f"Could not generate peak data: {e}")
        return None


def is_segment_in_music(segment_start, segment_end, music_segments):
    """Checks if a given time range overlaps with any music segments."""
    for music in music_segments:
        if max(segment_start, music['start']) < min(segment_end, music['end']):
            return True
    return False

@app.route('/segment-audio', methods=['POST'])
def segment_audio():
    print("--- Full Analysis Request Received (Single-Pass Mode) ---")
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
            print(f"Received audio URL to process: {audio_url}")
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.tmp")
            with open(original_temp_audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded remote file to: {original_temp_audio_path}")
            audio_source_for_processing = original_temp_audio_path
        else:
            return jsonify({"error": "No valid audio URL or file provided"}), 400

        # --- Step 1: Start AssemblyAI Transcription (asynchronous) ---
        print("Submitting audio source to AssemblyAI for transcription...")
        config = aai.TranscriptionConfig(speaker_labels=True)
        transcriber = aai.Transcriber()
        transcript_future = transcriber.transcribe_async(audio_source_for_processing, config)

        # --- Step 2: Perform Music Segmentation on the Full File ---
        print("Starting music segmentation on the full audio file...")
        
        audio_segment = AudioSegment.from_file(audio_source_for_processing)
        total_duration_seconds = len(audio_segment) / 1000.0
        
        processed_audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        processed_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        processed_audio_segment.export(processed_temp_audio_path, format="wav")
        
        seg_model = Segmenter()
        segmentation_results = seg_model(processed_temp_audio_path)

        final_music_segments = []
        current_music_block = None
        final_speaking_segments = []
        LONG_SPEECH_THRESHOLD_SECONDS = 120
        i = 0
        while i < len(segmentation_results):
            label, start, end = segmentation_results[i]
            if label == 'music':
                if current_music_block is None:
                    if (end - start) >= 3:
                        current_music_block = {'start': start, 'end': end}
                else:
                    current_music_block['end'] = end
            
            if label in ['male', 'female']:
                speech_duration = end - start
                if speech_duration >= LONG_SPEECH_THRESHOLD_SECONDS:
                    if current_music_block is not None:
                        final_music_segments.append(
                            {'start': round(current_music_block['start'], 2), 'end': round(current_music_block['end'], 2)}
                        )
                        current_music_block = None
                    final_speaking_segments.append(
                            {'label': label, 'start': round(start, 2), 'end': round(end, 2)}
                        )
            i += 1
            
        if current_music_block is not None:
            final_music_segments.append(
                {'start': round(current_music_block['start'], 2), 'end': round(current_music_block['end'], 2)}
            )
        print(f"Music segmentation complete. Found {len(final_music_segments)} segments.")

        # --- Step 3: Wait for AssemblyAI and Process Results ---
        print("Waiting for AssemblyAI transcript to complete...")
        transcript = transcript_future.result()

        if transcript.status == aai.TranscriptStatus.error:
            raise Exception(f"AssemblyAI transcription failed: {transcript.error}")

        filler_words_to_find = [
            'um', 'uh', 'er', 'ah', 'like', 'you know', 
            'i mean', 'so', 'kind of', 'sort of', 'basically', 'actually'
        ]
        
        filler_word_segments = [
            {"word": word.text, "start": round(word.start / 1000, 2), "end": round(word.end / 1000, 2)}
            for word in transcript.words
            if word.text.lower().strip(".,?!") in filler_words_to_find
            if not is_segment_in_music(round(word.start / 1000, 2), round(word.end / 1000, 2), final_music_segments)
        ]
        
        speech_gaps = transcript.speech_gaps if hasattr(transcript, 'speech_gaps') else []
        dead_air_segments = [
            {"start": round(gap.start / 1000, 2), "end": round(gap.end / 1000, 2), "duration": round(gap.duration / 1000, 2)}
            for gap in speech_gaps if gap.duration >= 3000
            if not is_segment_in_music(round(gap.start / 1000, 2), round(gap.end / 1000, 2), final_music_segments)
        ]

        speaker_segments = [
            {"speaker": utterance.speaker, "text": utterance.text, "start": round(utterance.start / 1000, 2), "end": round(utterance.end / 1000, 2)}
            for utterance in transcript.utterances
            if not is_segment_in_music(round(utterance.start / 1000, 2), round(utterance.end / 1000, 2), final_music_segments)
        ]

        # --- Step 4: Generate Peak Data ---
        peak_data = generate_peak_data(processed_temp_audio_path)

        print("All processing complete.")
        return jsonify({
            "duration": round(total_duration_seconds, 2),
            "music_segments": final_music_segments,
            "filler_word_segments": filler_word_segments,
            "dead_air_segments": dead_air_segments,
            "speaker_segments": speaker_segments,
            "peak_data": peak_data # Add peak data to the response
        }), 200

    except Exception as e:
        print(f"An error occurred during segmentation: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500
    finally:
        if original_temp_audio_path and os.path.exists(original_temp_audio_path):
            os.remove(original_temp_audio_path)
        if processed_temp_audio_path and os.path.exists(processed_temp_audio_path):
            os.remove(processed_temp_audio_path)

@app.route('/render-audio', methods=['POST'])
def render_audio():
    """
    Receives an audio source and a playback sequence, and uses FFmpeg
    to render a new, edited audio file.
    """
    print("--- Render Audio Request Received ---")
    data = request.get_json()
    audio_source = data.get('audio_url')
    playback_sequence = data.get('playback_sequence')

    if not audio_source or not playback_sequence:
        return jsonify({"error": "Missing audio_url or playback_sequence"}), 400

    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"exported_{uuid.uuid4()}.mp3")

    try:
        # Build the complex FFmpeg filtergraph to concatenate the clips
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
            'ffmpeg',
            '-i', audio_source,
            '-filter_complex', filter_graph,
            '-map', '[outa]',
            output_path
        ]

        print("Running FFmpeg to render final audio...")
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Exported audio saved to: {output_path}")

        # Return the generated file
        return send_file(
            output_path,
            as_attachment=True,
            download_name='exported_audio.mp3',
            mimetype='audio/mpeg'
        )

    except Exception as e:
        print(f"An error occurred during rendering: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print("FFMPEG STDERR:", e.stderr.decode())
        return jsonify({"error": f"An error occurred during rendering: {e}"}), 500
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)
