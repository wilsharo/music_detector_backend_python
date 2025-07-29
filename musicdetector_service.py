# inaspeech_service.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from pydub import AudioSegment
from inaSpeechSegmenter import Segmenter
import requests
import uuid
import soundfile as sf
import numpy as np # Keep numpy import, as soundfile uses it


# Attempt to import tensorflow early and catch specific errors if any
try:
   import tensorflow as tf
   print(f"TensorFlow version: {tf.__version__}")
   # Optional: Suppress TensorFlow warnings if it's just about CPU optimization
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress info and warning messages
except ImportError:
   print("ERROR: TensorFlow is not installed or cannot be imported.")
   exit(1)
except Exception as e:
   print(f"ERROR: An unexpected error occurred during TensorFlow import: {e}")
   exit(1)




app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) # Allow all origins for local testing


# Initialize the speech segmenter outside the route for efficiency
# This loads the model once when the service starts
try:
   # Force model download/re-check by setting a different cache directory
   # or by explicitly clearing the cache if possible.
   # inaSpeechSegmenter uses appdirs for cache, so removing that directory might help.
   # Forcing a re-download of models (if they were corrupted)
   # This is a bit aggressive, but aims to ensure model integrity.
   # You might need to manually delete the cache directory for inaSpeechSegmenter models
   # if this doesn't trigger a re-download.
   # The cache location is usually in ~/.cache/inaSpeechSegmenter_models or similar.
  
   seg = Segmenter()
   print("inaSpeechSegmenter model loaded successfully.")
except Exception as e:
   print(f"ERROR: Failed to load inaSpeechSegmenter model: {e}")
   print("This could be due to a corrupted model download, incompatible TensorFlow version, or other deep learning environment issues.")
   print("Please ensure all dependencies for inaSpeechSegmenter are met (e.g., TensorFlow).")
   exit(1) # Exit if model can't be loaded


# --- Global function definition for format_seconds_to_hms ---
def format_seconds_to_hms(seconds):
   """
   Converts a total number of seconds into a string formatted as HH:MM:SS.


   Args:
       seconds (int or float): The total number of seconds.


   Returns:
       str: The time formatted as HH:MM:SS.
   """
   hours = int(seconds // 3600)
   minutes = int((seconds % 3600) // 60)
   remaining_seconds = int(seconds % 60)


   return f"{hours:02}:{minutes:02}:{remaining_seconds:02}"
# --- End of format_seconds_to_hms function definition ---




@app.route('/segment-audio', methods=['POST'])
def segment_audio():
   """
   Receives an audio file (via upload or URL), segments it using inaSpeechSegmenter,
   and returns music segments and total duration.
   """
   print("--- Segment Audio Request Received ---")
   original_temp_audio_path = None
   processed_temp_audio_path = None # Path for the standardized audio
   audio_segment = None


   try:
       if 'audio_file' in request.files: # Check for file upload first
           audio_file = request.files['audio_file']
           if audio_file.filename == '':
               return jsonify({"error": "No selected file"}), 400
          
           # Create a unique temporary file path for uploaded file
           original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{os.path.splitext(audio_file.filename)[1]}")
           audio_file.save(original_temp_audio_path)
           print(f"Saved uploaded file to temporary path: {original_temp_audio_path}")


       elif request.is_json and 'audio_url' in request.json: # Then check for JSON with URL
           audio_url = request.json['audio_url']
           print(f"Received audio URL: {audio_url}")
           # Download audio from URL to a temporary file
           response = requests.get(audio_url, stream=True)
           response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
          
           # Create a unique temporary file path
           original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{os.path.splitext(audio_url)[1] or '.mp3'}")
          
           with open(original_temp_audio_path, 'wb') as f:
               for chunk in response.iter_content(chunk_size=8192):
                   f.write(chunk)
           print(f"Downloaded audio to temporary path: {original_temp_audio_path}")
       else:
           return jsonify({"error": "No valid audio URL or file provided in request"}), 400


       # Load audio using pydub to get duration and ensure valid format for segmenter
       try:
           audio_segment = AudioSegment.from_file(original_temp_audio_path)
           total_duration_seconds = len(audio_segment) / 1000.0 # Duration in seconds
           print(f"Original audio duration detected: {total_duration_seconds:.2f} seconds")


           # Standardize audio format for inaSpeechSegmenter using pydub
           # inaSpeechSegmenter expects 16kHz, mono, 16-bit PCM WAV
           processed_audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
          
           # Save the processed audio to a new temporary WAV file
           processed_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
           processed_audio_segment.export(processed_temp_audio_path, format="wav")
           print(f"Processed audio saved to: {processed_temp_audio_path}")


       except Exception as e:
           raise Exception(f"Could not load or process audio file with pydub. Ensure it's a valid audio format and FFmpeg is installed/in PATH: {e}")


       # Perform segmentation
       print("Starting inaSpeechSegmenter analysis...")
      
       # FIX: Pass the path to the standardized WAV file directly to the segmenter
       # inaSpeechSegmenter handles its own internal loading from the path.
       segmentation_results = seg(processed_temp_audio_path)
      
       # Add a check for NoneType return from inaSpeechSegmenter
       if segmentation_results is None:
           raise Exception("inaSpeechSegmenter returned None. This might indicate an internal processing failure or an unsupported audio format/content.")


       print("inaSpeechSegmenter analysis complete.")


       # Assuming 'segmentation_results' is the list provided by inaSpeechSegmenter, e.g.:
       # [('music', 0.0, 22.48), ('noEnergy', 22.48, 29.08), ...]


       final_music_segments = []
       current_music_block = None # Stores {'start': ..., 'end': ...} of the current merged music block


       final_speaking_segments = [] # Stores {'start': ..., 'end': ...} of speaking segments and identifies if man or woman speaking


       # Define the threshold for long speech segments (120 seconds)
       LONG_SPEECH_THRESHOLD_SECONDS = 120


       # Iterate through the raw segmentation results with an index
       i = 0
       while i < len(segmentation_results):
           label, start, end = segmentation_results[i]


           if label == 'music':
               # If starting a new music block
               if current_music_block is None:
                   # FIX: This line was incorrectly nested. It should be part of the outer 'if' block.
                   # Also, the condition `if end - start >= 3:` was likely intended for the initial segment,
                   # but its placement here was structurally problematic.
                   # Assuming you want to start a block if the music segment itself is at least 3 seconds.
                   if (end - start) >= 3: # This condition was causing indentation issues
                       current_music_block = {'start': start, 'end': end}
               else:
                   # If already in a music block, just extend its end
                   current_music_block['end'] = end


           if label in ['male', 'female']:
               speech_duration = end - start
               if speech_duration >= LONG_SPEECH_THRESHOLD_SECONDS:
                   # Found a long speech segment, finalize the current music block
                   if current_music_block is not None:
                       final_music_segments.append(
                           {'label': 'music', 'start': round(current_music_block['start'], 2), 'end': round(current_music_block['end'], 2)}
                       )
                       current_music_block = None # Reset for a new music block
                   final_speaking_segments.append(
                           {'label': label, 'start': round(start, 2), 'end': round(end, 2)}
                       )


           i += 1 # Increment outer loop index


       # FIX: This append should be conditional and outside the loop's direct flow
       # It should only append if a music block was active at the very end of the audio.
       # This line was also causing an AttributeError if current_music_block was None.
       if current_music_block is not None: # Added check for None
           final_music_segments.append(
               {'label': 'music', 'start': round(current_music_block['start'], 2), 'end': round(current_music_block['end'], 2)}
           )


       return jsonify({
           "duration": round(total_duration_seconds, 2),
           "segments": final_music_segments
       }), 200


   except requests.exceptions.RequestException as e:
       print(f"Error downloading audio from URL: {e}")
       return jsonify({"error": f"Failed to download audio from URL: {e}"}), 400
   except Exception as e:
       print(f"An error occurred during segmentation: {e}")
       return jsonify({"error": f"An error occurred during audio processing: {e}"}), 500
   finally:
       # Clean up the temporary audio files
       if original_temp_audio_path and os.path.exists(original_temp_audio_path):
           try:
               os.remove(original_temp_audio_path)
               print(f"Cleaned up original temporary file: {original_temp_audio_path}")
           except Exception as e:
               print(f"Error cleaning up original temporary file {original_temp_audio_path}: {e}")
       if processed_temp_audio_path and os.path.exists(processed_temp_audio_path):
           try:
               os.remove(processed_temp_audio_path)
               print(f"Cleaned up processed temporary file: {processed_temp_audio_path}")
           except Exception as e:
               print(f"Error cleaning up processed temporary file {processed_temp_audio_path}: {e}")


if __name__ == '__main__':
   app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)