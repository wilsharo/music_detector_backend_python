# This script's only purpose is to trigger the download of the
# inaSpeechSegmenter models during the Docker build process.
from inaSpeechSegmenter import Segmenter

print("Downloading inaSpeechSegmenter models...")
try:
    # Initializing the Segmenter will automatically download
    # and cache the necessary model files.
    seg = Segmenter()
    print("Models downloaded and cached successfully.")
except Exception as e:
    print(f"An error occurred during model download: {e}")
    # Exit with a non-zero code to fail the Docker build if download fails
    exit(1)