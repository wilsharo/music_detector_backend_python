# musicdetector_service.py
from flask import Flask, request, jsonify
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

# --- AssemblyAI API Key ---
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    print("WARNING: ASSEMBLYAI_API_KEY environment variable not set.")
aai.settings.api_key = ASSEMBLYAI_API_KEY

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

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
    processed_temp_audio_path = None # For the standardized WAV file
    
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
            # --- FIX: Download the remote file to a local temporary path ---
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
        # AssemblyAI can handle the local file path directly
        print("Submitting audio source to AssemblyAI for transcription...")
        config = aai.TranscriptionConfig(speaker_labels=True)
        transcriber = aai.Transcriber()
        transcript_future = transcriber.transcribe_async(audio_source_for_processing, config)

        # --- Step 2: Perform Music Segmentation on the Full File ---
        print("Starting music segmentation on the full audio file...")
        
        # Load the entire audio file using pydub from the local path
        audio_segment = AudioSegment.from_file(audio_source_for_processing)
        total_duration_seconds = len(audio_segment) / 1000.0
        
        # Standardize the entire file for the segmenter
        processed_audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # Export the full standardized file to a temporary WAV
        processed_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        processed_audio_segment.export(processed_temp_audio_path, format="wav")
        
        # Run the segmenter once on the complete, standardized file
        seg_model = Segmenter()
        segmentation_results = seg_model(processed_temp_audio_path)

        # Merge the results
        final_music_segments = []
        current_music_block = None
        for label, start, end in segmentation_results:
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

        print("All processing complete.")
        return jsonify({
            "duration": round(total_duration_seconds, 2),
            "music_segments": final_music_segments,
            "filler_word_segments": filler_word_segments,
            "dead_air_segments": dead_air_segments,
            "speaker_segments": speaker_segments
        }), 200

    except Exception as e:
        print(f"An error occurred during segmentation: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500
    finally:
        # Clean up all temporary files
        if original_temp_audio_path and os.path.exists(original_temp_audio_path):
            os.remove(original_temp_audio_path)
        if processed_temp_audio_path and os.path.exists(processed_temp_audio_path):
            os.remove(processed_temp_audio_path)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)
