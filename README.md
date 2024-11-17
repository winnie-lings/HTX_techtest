## HTX xData Technical Test: Automatic Speech Recognition (ASR) and Hotword Detection

### Description
This repository contains the code and resources for a series of tasks involving Automatic Speech Recognition (ASR) and hotword detection. The tasks include setting up and deploying a speech-to-text microservice, training a fine-tuned ASR model, and performing hotword detection on transcribed audio data.


### File Directories:

The task is divided into the following directories and files in the main branch:

- (Task 2) asr/: Contains codes for testing and setting up the ASR API using Flask, transcribing Common Voice audio data and containerising the ASR API using Docker.
- (Task 3) asr-train/: Contains code for fine-tuning the ASR model on the Common Voice dataset.
- (Task 4) training-report.pdf: Report to compare the finetuned model (Task 3) against the results from Task 2 for the cv-valid-dev mp3 dataset. Includes observations and proposed series of steps (including datasets and experiments) to improve the accuracy.
- (Task 5) hotword-detection/: Contains code for detecting hotwords in transcribed audio data and finding similar phrases using text embedding.
- (Task 6) essay-ssl.pdf: Essay question (read https://arxiv.org/pdf/2205.08598.pdf), with proposed self-supervised learning pipeline to cater dysarthric speech and continuous learning in 500 words.
- requirements.txt: Lists the Python dependencies for the tasks.
- .gitignore: Specifies files and directories to ignore in the Git repository.
- README.md: Documentation for setting up and running the tasks

 ### Prerequisites

The following needs to be installed before running the above tasks:

- Python 3.8+
- Conda. You may install it by installing Anaconda: https://docs.anaconda.com/anaconda/install/. Check the installation by running

```bash
conda --version
```

- Docker (for containerizing the ASR Flask API). You may refer to https://docs.docker.com/engine/install/ for the installation of Docker Desktop based on your system requirements.
- Git
- Jupyter notebook

```bash
 conda install jupyter
```

- FFmpeg (Widely used for multimedia processing, and installing it ensures your container can handle various audio formats (including MP3) and fetch metadata like file duration)
- (optional, if you prefer using a graphical interface for testing the ASR API) Postman. You may download from https://www.postman.com/downloads/

### Steps to set up conda environment, installing dependencies and running codes within it for Tasks 2, 3 and 5

#### 1. Set up a new conda environment and activate it to run the following codes

a) Create a new Conda environment. You can specify a Python version (e.g., Python 3.8) to match the dependencies required for the tasks. Here, python version used is 3.8.20. You may wish to replace myenv with your own name for the environment. 

```bash
conda create --name myenv python=3.8
```

b) Activate the environment:

```bash
conda activate myenv
```
Once activated, your prompt should change to indicate the active environment, for example:

```bash
(myenv) user@hostname:~/task$
```

#### 2. In the new conda environment, clone the repository and install the dependencies necessary for all the tasks

Ensure you are in the task directory where requirements.txt is located.

a) If Git is not installed in your system, you can download from https://git-scm.com/downloads and verify the installation by running:

```bash
git --version
```

This will show the installed version of Git, e.g., git version 2.x.x.

b) Navigate to the directory where you want to clone the repository and clone it

```bash
cd /path/to/your/task

git clone https://github.com/winnie-lings/HTX_techtest.git
```

c) Install the dependencies

```bash
pip install -r requirements.txt
```

#### 3. Download common-voice dataset
Download the Common Voice dataset from https://www.dropbox.com/scl/fi/i9yvfqpf7p8uye5o8k1sj/common_voice.zip?rlkey=lz3dtjuhekc3xw4jnoeoqy5yu&dl=0 and extract it. 

#### 4. Run the ASR microservice

a) Start the ASR microservice

The ASR microservice uses the pretrained wav2vec2-large-960h model from Hugging Face to transcribe audio using Flask API. The code is in asr/asr_api.py

To run the ASR service:

```bash
python asr/asr_api.py
```
This will start the API at http://localhost:8001. You can check if the service is running by:

```bash
http://localhost:8001/ping
```
You will receive a response “pong” if the service is working. (Task 2b)

b) Test the ASR service using CURL (Task 2c)

```bash
curl -F 'file=@/path/to/your/audiofile.mp3' http://localhost:8001/asr
```

If you are using Anaconda Powershell in Windows, edit the code to:

```bash
curl.exe -F 'file=@/path/to/your/audiofile.mp3' http://localhost:8001/asr
```
This will return the transcription and duration of the audio file.

Alternatively, if you prefer using a graphical interface for testing, download and open Postman app. 

Step-by-step guide to using Postman:

1. Set up a POST request in Postman:
   
   (a) URL: In the URL field, enter http://localhost:8001/asr.
    
   (b) Method: Set the HTTP method to POST. 

   (c) Body:
   
   i) Select the Body tab.   
   ii) Choose form-data.   
   iii. In the Key field, enter file (this is the key your API is expecting for the file upload).   
   iv. In the Value field, select your audio file (e.g., sample-000000.mp3).
   
2. Send the Request:
   
   Click the Send button in Postman. The request will be sent to your API with the file. 

3. View the Response:

   If the API is working correctly, you will see the JSON response with the transcription and duration, similar to:

   { 
    "duration": "5.064",
   
    "transcription": "BE CAREFUL WITH YOUR PROGNOSTICATIONS SAID THE STRANGER" 
} 

#### 5. Transcribe the 4,076 common-voice mp3 files under cv-valid-dev folder (Task 2d)

The script cv-decode.py will call the ASR API for all the audio files in the cv-valid-dev folder and write the transcriptions into a new column (generated_text) in the cv-valid-dev.csv.

To run the transcription script:

```bash
python asr/cv-decode.py
```

#### 6. Containerize the ASR API Using Docker (Task 2e)

Please open the downloaded Docker Desktop app. 

Dockerfile, found in asr/Dockerfile is used to containerize the ASR API. Change directory to asr/ and build and run the Docker container using the following commands:

a) Build the Docker image

```bash
docker build -t asr-api .
```

b) Run the Docker container

```bash
docker run -p 8001:8001 asr-api
```

#### 7. Fine-tune the ASR Model (Task 3 - asr-train/cv-train-2a.ipynb)

The fine-tuning of the wav2vec2-large-960h ASR model is done in the Jupyter Notebook, cv-train-2a.ipynb using cv-valid-train dataset.

To run the notebook:

a) Start a Jupyter Notebook server in the conda environment:
   
```bash
jupyter notebook
```

b) Open asr-train/cv-train-2a.ipynb in your browser and run the notebook. 

The fine-tuned model is saved as wav2vec2-large-960h-cv (Task 3b). After training, the model will be used to transcribe the common-voice mp3 files under cv-valid-test dataset for evaluation and overall performance logged (Task 3c).

#### 8. Hot words detection (Task 5a - hotword-detection/cv-hotword-5a.ipynb and Task 5b - hotword-detection/cv-hotword-similarity-5b.ipynb)

a) Detect hot words in transcriptions

Please refer to cv-hotword-5a.ipynb notebook to detect hotwords such as "be careful", "destroy", and "stranger", with the list of mp3 filenames with the hot words detected stored in detected.txt

b) Find similar phrases to the hot words using text embedding model hkunlp/instructor-large

Please refer to cv-hotword-similarity-5b.ipynb notebook for this task. The record containing similar phrases to the hot words is stored as a Boolean in a new column called 'similarity' in cv-valid-dev.csv. 

#### 9. Deactivate Conda environment once done

```bash
conda deactivate
```











  
