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
import subprocess
import glob
import shutil
import json
from multiprocessing import Process

# This function will be run in a separate process for each chunk
def process_chunk(chunk_file_path, result_file_path):
    try:
        from inaSpeechSegmenter import Segmenter
        import tensorflow as tf
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        print(f"[pid:{os.getpid()}] Initializing Segmenter for chunk...")
        seg = Segmenter()
        print(f"[pid:{os.getpid()}] Segmenter initialized.")

        audio_chunk = AudioSegment.from_file(chunk_file_path)
        processed_chunk = audio_chunk.set_frame_rate(16000).set_channels(1)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            processed_chunk.export(temp_wav.name, format="wav")
            
            print(f"[pid:{os.getpid()}] Segmenting chunk: {os.path.basename(chunk_file_path)}")
            chunk_segmentation = seg(temp_wav.name)
            
            with open(result_file_path, 'w') as f:
                json.dump(chunk_segmentation, f)
            print(f"[pid:{os.getpid()}] Finished processing chunk.")

    except Exception as e:
        print(f"[pid:{os.getpid()}] ERROR in child process: {e}")
        with open(result_file_path, 'w') as f:
            json.dump({"error": str(e)}, f)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

@app.route('/segment-audio', methods=['POST'])
def segment_audio():
    print("--- Segment Audio Request Received (Isolated Subprocess Chunking) ---")
    original_temp_audio_path = None
    temp_chunk_dir = None
    
    try:
        if 'audio_file' in request.files:
            audio_file = request.files['audio_file']
            original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{os.path.splitext(audio_file.filename)[1]}")
            audio_file.save(original_temp_audio_path)
        elif request.is_json and 'audio_url' in request.json:
            audio_url = request.json['audio_url']
            response = requests.get(audio_url, stream=True)
            response.raise_for_status()
            
            # --- FIX: Create a simple, unique temp file name ---
            # Instead of deriving the name from the potentially very long URL.
            original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.tmp")
            
            with open(original_temp_audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
            print(f"Downloaded audio to simple temp path: {original_temp_audio_path}")
        else:
            return jsonify({"error": "No valid audio URL or file provided"}), 400

        with sf.SoundFile(original_temp_audio_path) as f:
            total_duration_seconds = len(f) / f.samplerate
        
        CHUNK_DURATION_SECONDS = 300
        temp_chunk_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_chunk_dir, 'chunk_%03d.wav')
        
        ffmpeg_command = [
            'ffmpeg', '-i', original_temp_audio_path,
            '-f', 'segment', '-segment_time', str(CHUNK_DURATION_SECONDS),
            '-c:a', 'pcm_s16le', output_template
        ]
        
        print("Running FFmpeg to split audio...")
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        chunk_files = sorted(glob.glob(os.path.join(temp_chunk_dir, 'chunk_*.wav')))
        print(f"Found {len(chunk_files)} chunk(s) to process in isolated subprocesses.")
        
        all_segments = []
        
        for i, chunk_file_path in enumerate(chunk_files):
            result_file_path = os.path.join(temp_chunk_dir, f"result_{i}.json")
            
            p = Process(target=process_chunk, args=(chunk_file_path, result_file_path))
            p.start()
            p.join()

            if not os.path.exists(result_file_path):
                raise Exception(f"Processing failed for chunk {i}, result file not found.")
            
            with open(result_file_path, 'r') as f:
                result_data = json.load(f)
                if "error" in result_data:
                    raise Exception(f"Error in chunk {i}: {result_data['error']}")
                
                offset = i * CHUNK_DURATION_SECONDS
                for label, start, end in result_data:
                    all_segments.append((label, start + offset, end + offset))
            
            print(f"Main process successfully processed results for chunk {i}")

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
        print(f"An error occurred during segmentation: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            return jsonify({"error": f"FFmpeg failed: {e.stderr.decode()}"}), 500
        return jsonify({"error": f"An error occurred: {e}"}), 500
    finally:
        if original_temp_audio_path and os.path.exists(original_temp_audio_path):
            os.remove(original_temp_audio_path)
        if temp_chunk_dir and os.path.exists(temp_chunk_dir):
            shutil.rmtree(temp_chunk_dir)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)
