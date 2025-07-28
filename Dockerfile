# Use a slim Python 3.9 image based on Debian Bullseye (Debian 11)
FROM python:3.9-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Ensure DEBIAN_FRONTEND is noninteractive for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# --- FIX: Install 'procps' for the 'ps' command, needed by some TensorFlow/TensorRT setup scripts ---
# This ensures the 'ps' command is available during pip install.
RUN apt-get update -yqq && \
    apt-get install -y --no-install-recommends procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# This will now install the standard 'tensorflow' package which supports GPU if available.
RUN pip install --no-cache-dir -r requirements.txt

# Install FFmpeg (required by pydub for audio processing)
RUN apt-get update -yqq && \
    apt-get install -y --no-install-recommends ca-certificates ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the model download script into the container
COPY download_models.py .

# Run the script to download and cache the models.
# This happens only once, during the 'docker build' step.
RUN python download_models.py

# Copy the Flask application code into the container
COPY musicdetector_service.py .

# Create a directory for temporary file uploads (used by the service)
RUN mkdir -p /tmp/uploads

# Expose the port that the Flask application will listen on
EXPOSE 5002

# Define the command to run the Flask application when the container starts
CMD ["python", "musicdetector_service.py", "--host", "0.0.0.0", "--port", "5002"]
