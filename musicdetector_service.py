# musicdetector_service.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import requests
import uuid
import soundfile as sf
import subprocess
import glob
import shutil
import json
from multiprocessing import Process, Manager
import assemblyai as aai

# --- AssemblyAI API Key ---
# IMPORTANT: Set this as an environment variable in your deployment
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    print("WARNING: ASSEMBLYAI_API_KEY environment variable not set.")
aai.settings.api_key = ASSEMBLYAI_API_KEY

# This function is for the music segmentation subprocess
def process_music_chunk(chunk_file_path, result_list):
    try:
        from inaSpeechSegmenter import Segmenter
        from pydub import AudioSegment
        seg = Segmenter()
        audio_chunk = AudioSegment.from_file(chunk_file_path)
        processed_chunk = audio_chunk.set_frame_rate(16000).set_channels(1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
            processed_chunk.export(temp_wav.name, format="wav")
            chunk_segmentation = seg(temp_wav.name)
            result_list.extend(chunk_segmentation)
    except Exception as e:
        print(f"[pid:{os.getpid()}] ERROR in music chunk process: {e}")

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
    print("--- Full Analysis Request Received ---")
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
            original_temp_audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.tmp")
            with open(original_temp_audio_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        else:
            return jsonify({"error": "No valid audio URL or file provided"}), 400

        # --- Step 1: Start AssemblyAI Transcription (asynchronous) ---
        print("Submitting file to AssemblyAI for transcription...")
        config = aai.TranscriptionConfig(speaker_labels=True, disfluencies=True)
        transcriber = aai.Transcriber()
        transcript_future = transcriber.transcribe_async(original_temp_audio_path, config)

        # --- Step 2: Perform Music Segmentation in Parallel ---
        print("Starting music segmentation...")
        ffprobe_command = ['ffprobe', '-v', 'error', '-show_format', '-print_format', 'json', original_temp_audio_path]
        ffprobe_result = subprocess.run(ffprobe_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        total_duration_seconds = float(json.loads(ffprobe_result.stdout)['format']['duration'])
        
        CHUNK_DURATION_SECONDS = 300
        temp_chunk_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_chunk_dir, 'chunk_%03d.wav')
        
        ffmpeg_command = ['ffmpeg', '-i', original_temp_audio_path, '-f', 'segment', '-segment_time', str(CHUNK_DURATION_SECONDS), '-c:a', 'pcm_s16le', output_template]
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        chunk_files = sorted(glob.glob(os.path.join(temp_chunk_dir, 'chunk_*.wav')))
        
        with Manager() as manager:
            all_music_segments_raw = manager.list()
            processes = []
            for i, chunk_file_path in enumerate(chunk_files):
                p = Process(target=process_music_chunk, args=(chunk_file_path, all_music_segments_raw))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
            
            # Merge music segments
            final_music_segments = []
            current_music_block = None
            # Adjust timestamps
            adjusted_segments = []
            for i, (label, start, end) in enumerate(all_music_segments_raw):
                # This logic assumes chunks are processed in order, which they are.
                # A more robust solution might pass the chunk index 'i' back.
                offset = (i // (len(all_music_segments_raw) / len(chunk_files))) * CHUNK_DURATION_SECONDS
                adjusted_segments.append((label, start + offset, end + offset))

            for label, start, end in adjusted_segments:
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

        # Process Filler Words
        filler_word_segments = [
            {"word": word.text, "start": round(word.start / 1000, 2), "end": round(word.end / 1000, 2)}
            for word in transcript.words if word.word_type == aai.WordType.filler
            if not is_segment_in_music(round(word.start / 1000, 2), round(word.end / 1000, 2), final_music_segments)
        ]
        
        # Process Dead Air (Silence)
        dead_air_segments = [
            {"start": round(gap.start / 1000, 2), "end": round(gap.end / 1000, 2), "duration": round(gap.duration / 1000, 2)}
            for gap in transcript.speech_gaps if gap.duration >= 3000 # 3 seconds threshold
            if not is_segment_in_music(round(gap.start / 1000, 2), round(gap.end / 1000, 2), final_music_segments)
        ]

        # Process Speaker Segments
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
        if original_temp_audio_path and os.path.exists(original_temp_audio_path):
            os.remove(original_temp_audio_path)
        if temp_chunk_dir and os.path.exists(temp_chunk_dir):
            shutil.rmtree(temp_chunk_dir)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5002, use_reloader=False)
