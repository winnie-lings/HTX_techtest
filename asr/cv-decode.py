import os
import csv
import requests

# Path to the folder of the downloaded data "Common Voice"
cv_folder = 'C:/Users/lingy/OneDrive/Desktop/HTX/HTX_techtest/common_voice/'

# Path to the folder containing the common-voice mp3 files under cv-valid-dev folder
audio_folder = os.path.join(cv_folder, 'cv-valid-dev/')

# Path to the CSV file that contains the metadata
csv_file = os.path.join(cv_folder, 'cv-valid-dev.csv')

# Define the API URL
api_url = 'http://localhost:8001/asr'

# Read the existing CSV file
with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Prepare a list of files to transcribe
for row in rows:
    # Get the file path from the CSV row
    file_path = os.path.join(audio_folder, row['filename'])

    # Make a request to the /asr API
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'audio/mp3')}
        response = requests.post(api_url, files=files)

        if response.status_code == 200:
            transcription = response.json()['transcription']
            duration = response.json()['duration']
            print(f"Transcription for {file_path}: {transcription} (Duration: {duration} seconds)")

            # Add the transcription to the row
            row['generated_text'] = transcription
        else:
            print(f"Error processing {file_path}: {response.text}")
            row['generated_text'] = 'ERROR'

# Write the updated CSV back to disk
with open(csv_file, 'w', newline='') as file:
    fieldnames = reader.fieldnames + ['generated_text']
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(rows)

print("Transcription completed and saved to CSV.")
