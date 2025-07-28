inaSpeechSegmenter Service
This is a Python Flask backend service that wraps the inaSpeechSegmenter library to detect music segments within audio files. It exposes a simple API endpoint that your main Node.js backend (or any other client) can call to get music timestamps.

Features
Receives audio files via URL or direct upload.

Uses inaSpeechSegmenter to classify audio segments as speech, music, or noise.

Returns a JSON response with the total audio duration and timestamps of detected music segments.

Prerequisites
Before running this service, ensure you have the following installed:

Python 3.8+: This service is developed with Python.

pip: Python's package installer (usually comes with Python).

FFmpeg: pydub (used for audio file handling) requires FFmpeg to be installed on your system and accessible in your system's PATH.

macOS (using Homebrew): brew install ffmpeg

Linux (Debian/Ubuntu): sudo apt update && sudo apt install ffmpeg

Windows: Download from https://ffmpeg.org/download.html and add it to your system's PATH.

Setup Instructions
Follow these steps to get the inaSpeechSegmenter service up and running locally:

Clone or Download the Repository:
If this service is part of a larger project, navigate to its directory. Otherwise, create a new directory (e.g., inaspeech_service) and place inaspeech_service.py and requirements.txt inside it.

Create a Python Virtual Environment (Recommended):
It's best practice to use a virtual environment to isolate project dependencies.

cd /path/to/your/inaspeech_service
python3 -m venv venv

Activate the Virtual Environment:

macOS / Linux:

source venv/bin/activate

Windows (Command Prompt):

.\venv\Scripts\activate

Windows (PowerShell):

.\venv\Scripts\Activate.ps1

Your terminal prompt should now show (venv) at the beginning.

Install Python Dependencies:
With the virtual environment activated, install the required packages using the provided requirements.txt file:

pip install -r requirements.txt

This will install Flask, inaSpeechSegmenter, pydub, requests, and their dependencies including TensorFlow.

Running the Service
Once all dependencies are installed, you can start the Flask service:

python inaspeech_service.py

The service will start on http://127.0.0.1:5002. Keep this terminal window open as long as you need the service to be running.

API Endpoint
The service exposes one main endpoint:

POST /segment-audio
Analyzes an audio file for music segments.

Request:

Method: POST

URL: http://localhost:5002/segment-audio

Headers:

Content-Type: application/json (if sending audio_url)

Content-Type: multipart/form-data (if sending audio_file - handled automatically by clients like FormData in JavaScript or files in curl)

Body (JSON for URL):

{
  "audio_url": "https://example.com/path/to/your/audio.mp3"
}

Body (multipart/form-data for file upload):

--<boundary>
Content-Disposition: form-data; name="audio_file"; filename="your_audio.wav"
Content-Type: audio/wav

<binary audio data>
--<boundary>--

Response (JSON):

{
  "duration": 185.67,
  "segments": [
    { "start": 10.25, "end": 45.80 },
    { "start": 90.10, "end": 120.00 }
  ]
}

Testing the Service
You can test the service directly using curl from your terminal:

Test with an Audio URL:
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{ "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" }' \
     http://localhost:5002/segment-audio

Test with a Local Audio File (replace path/to/your/audio.mp3):
curl -X POST \
     -F "audio_file=@/path/to/your/audio.mp3" \
     http://localhost:5002/segment-audio

(Make sure to use @ before the file path for curl to send it as a file).

Troubleshooting
inaSpeechSegmenter model loaded successfully. not appearing: Check your Python environment and TensorFlow installation.

Could not load audio file with pydub: Ensure FFmpeg is correctly installed and in your system's PATH.

Failed to download audio from URL: Check the provided URL and your internet connection.

Connection refused or Failed to fetch from client: Ensure the inaspeech_service.py is running on port 5002 and no firewall is blocking the connection.

500 Internal Server Error: Check the terminal where inaspeech_service.py is running for detailed Python error messages.