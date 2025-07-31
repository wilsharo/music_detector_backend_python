# musicdetector_service.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from pydub import AudioSegment
from inaSpeechSegmenter import Segmenter
import requests
import uuid
import soundfile as sf
import numpy as np
import json
import assemblyai as aai
import subprocess

# --- AssemblyAI API Key ---
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    print("WARNING: ASSEMBLYAI_API_KEY environment variable not set.")
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Attempt to import tensorflow
try:
   import tensorflow as tf
   print(f"TensorFlow version: {tf.__version__}")
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
   print("ERROR: TensorFlow is not installed or cannot be imported.")
   exit(1)
except Exception as e:
   print(f"ERROR: An unexpected error occurred during TensorFlow import: {e}")
   exit(1)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize the speech segmenter
try:
   seg = Segmenter()
   print("inaSpeechSegmenter model loaded successfully.")
except Exception as e:
   print(f"ERROR: Failed to load inaSpeechSegmenter model: {e}")
   exit(1)

def is_segment_in_music(segment_start, segment_end, music_segments):
    """Checks if a given time range overlaps with any music segments."""
    for music in music_segments:
        if max(segment_start, music['start']) < min(segment_end, music['end']):
            return True
    return False

@app.route('/segment-audio', methods=['POST'])
def segment_audio():
   """
   Receives an audio file, segments it for music, transcribes for other data,
   and returns a combined analysis.
   """
   print("--- Full Analysis Request Received ---")
   original_temp_audio_path = None
   processed_temp_audio_path = None

   try:
       audio_source_for_processing = None
       if 'audio_file' in request.files:
           audio_file = request.files['audio_file']
           if audio_file.filename == '':
               return jsonify({"error": "No selected file"}), 400
           original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{os.path.splitext(audio_file.filename)[1]}")
           audio_file.save(original_temp_audio_path)
           audio_source_for_processing = original_temp_audio_path
       elif request.is_json and 'audio_url' in request.json:
           audio_url = request.json['audio_url']
           print(f"Received audio URL: {audio_url}")
           response = requests.get(audio_url, stream=True)
           response.raise_for_status()
           original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.tmp")
           with open(original_temp_audio_path, 'wb') as f:
               for chunk in response.iter_content(chunk_size=8192):
                   f.write(chunk)
           print(f"Downloaded audio to temporary path: {original_temp_audio_path}")
           audio_source_for_processing = original_temp_audio_path
       else:
           return jsonify({"error": "No valid audio URL or file provided in request"}), 400

       # --- Step 1: Start AssemblyAI Transcription (asynchronous) ---
       print("Submitting audio to AssemblyAI...")
       config = aai.TranscriptionConfig(speaker_labels=True)
       transcriber = aai.Transcriber()
       transcript_future = transcriber.transcribe_async(audio_source_for_processing, config)

       # --- Step 2: Perform Music Segmentation using provided baseline logic ---
       print("Starting music segmentation...")
       audio_segment = AudioSegment.from_file(audio_source_for_processing)
       total_duration_seconds = len(audio_segment) / 1000.0
       
       processed_audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
       processed_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
       processed_audio_segment.export(processed_temp_audio_path, format="wav")
       
       segmentation_results = seg(processed_temp_audio_path)
       if segmentation_results is None:
           raise Exception("inaSpeechSegmenter returned None.")
       print("inaSpeechSegmenter analysis complete.")

       # --- FIX: Restored the original, correct logic for processing music and speech segments ---
       final_music_segments = []
       current_music_block = None
       final_speaking_segments = [] # This is unused in the final return, but preserved for logic
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
           
           # This logic correctly finalizes a music block if a long speech segment is detected
           if label in ['male', 'female']:
               speech_duration = end - start
               if speech_duration >= LONG_SPEECH_THRESHOLD_SECONDS:
                   if current_music_block is not None:
                       final_music_segments.append(
                           {'start': round(current_music_block['start'], 2), 'end': round(current_music_block['end'], 2)}
                       )
                       current_music_block = None # Reset for a new music block
                   final_speaking_segments.append(
                           {'label': label, 'start': round(start, 2), 'end': round(end, 2)}
                       )
           i += 1
           
       if current_music_block is not None:
           final_music_segments.append(
               {'start': round(current_music_block['start'], 2), 'end': round(current_music_block['end'], 2)}
           )
       print(f"Music segmentation complete. Found {len(final_music_segments)} segments.")
       # --- END OF FIX ---

       # --- Step 3: Wait for AssemblyAI and Process Results ---
       print("Waiting for AssemblyAI transcript to complete...")
       transcript = transcript_future.result()
       if transcript.status == aai.TranscriptStatus.error:
           raise Exception(f"AssemblyAI transcription failed: {transcript.error}")

       filler_words_to_find = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'so', 'kind of', 'sort of', 'basically', 'actually']
       
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
       return jsonify({"error": f"An error occurred during processing: {e}"}), 500
   finally:
       if original_temp_audio_path and os.path.exists(original_temp_audio_path):
           os.remove(original_temp_audio_path)
       if processed_temp_audio_path and os.path.exists(processed_temp_audio_path):
           os.remove(processed_temp_audio_path)

if __name__ == '__main__':
   app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)
