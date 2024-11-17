# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install FFmpeg (required for ffprobe to work, else will get an error message: "Error processing audio: [Errno 2] No such file or directory: 'ffprobe'")
# FFmpeg is widely used for multimedia processing, and installing it ensures the container can handle various audio formats (including MP3) and fetch metadata like file duration.
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the container
COPY . /app/

# Expose the port the app will run on
EXPOSE 8001

# Command to run the application
CMD ["python", "asr_api.py"]