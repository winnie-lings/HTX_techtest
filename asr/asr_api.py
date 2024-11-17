import os
import io
import json
import torch
import soundfile as sf
from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pydub import AudioSegment

app = Flask(__name__)

# Load the pre-trained Wav2Vec2 model and processor from Hugging Face
# Wav2Vec2Processor handles both feature extraction from audio and tokenization of the transcriptions.
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Check if a GPU is available and move the model to GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


@app.route('/ping', methods=['GET'])
def ping():
    return "pong", 200


@app.route('/asr', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Log the filename to verify the file is received correctly
    print(f"Received file: {file.filename}")

    # Create a temporary file to save the uploaded file
    temp_file_path = f"temp_{file.filename}"

    try:
        # Save the uploaded file temporarily
        file.save(temp_file_path)

        # Convert the audio file to the correct format (16kHz WAV)
        audio = AudioSegment.from_file(temp_file_path)
        audio = audio.set_frame_rate(16000)  # Set the sampling rate to 16kHz
        audio = audio.set_channels(1)  # Mono channel is needed
        audio = audio.set_sample_width(2)  # 16-bit depth

        # Save to a temporary buffer as WAV format
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)

        # Read the audio file using SoundFile to get the correct array
        speech, sample_rate = sf.read(buffer)

        # Ensure the sample rate is correct (16kHz)
        if sample_rate != 16000:
            return jsonify({"error": "Sample rate should be 16kHz"}), 400

        # Perform ASR (Automatic Speech Recognition)
        input_values = processor(speech, return_tensors="pt", padding=True).input_values
        input_values = input_values.to(device)

        # Get the transcription from the model
        with torch.no_grad():
            logits = model(input_values).logits

        # Decode the logits to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        # Get the duration of the audio file
        duration = len(speech) / sample_rate  # Duration in seconds

        # Return the transcription and duration
        return jsonify({
            "transcription": transcription,
            "duration": str(duration)
        })

    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

    finally:
        # Delete the temporary file after processing (in case of success or failure)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Deleted temporary file: {temp_file_path}")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001)
