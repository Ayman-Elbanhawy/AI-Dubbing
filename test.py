import os
import subprocess
import whisper
import numpy as np
from googletrans import Translator
from TTS.api import TTS
import sys
import wave
from mtcnn import MTCNN
import dlib
import cv2

# Paths and folder setup
BASE_DIR = "c:/SourceCode/dubbing"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
WAV2LIP_DIR = os.path.join(BASE_DIR, "Wav2Lip")
OUTPUT_VIDEO = os.path.join(VIDEO_DIR, "output_video.mp4")

# Function to extract audio from video using system-level FFmpeg
def extract_audio(video_path, audio_path):
    command = ["ffmpeg", "-i", video_path, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", audio_path]
    subprocess.run(command)

# Function to run Wav2Lip for lip-syncing
def run_wav2lip(input_video, input_audio, output_video):
    command_wav2lip = [
        "python", os.path.join(WAV2LIP_DIR, "inference.py"),
        "--checkpoint_path", os.path.join(WAV2LIP_DIR, "checkpoints\wav2lip.pth"),
        "--face", input_video,
        "--audio", input_audio,
        "--resize_factor", "2",
        "--outfile", output_video
    ]
    subprocess.run(command_wav2lip)

# Function to load audio using system FFmpeg for Whisper transcription
def load_audio_with_ffmpeg(file, sr=16000):
    temp_wav_path = os.path.join(VIDEO_DIR, "temp_audio.wav")
    command = f"ffmpeg -i {file} -ar {sr} -ac 1 -f wav {temp_wav_path} -y"
    subprocess.run(command, shell=True)

    with wave.open(temp_wav_path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return audio

# Language selection for source and target languages
def choose_language():
    languages = {
        '1': ('en', 'English'),
        '2': ('es', 'Spanish'),
        '3': ('ar', 'Arabic'),
        '4': ('fr', 'French'),
        '5': ('de', 'German'),
        '6': ('it', 'Italian'),
        # Add more languages if needed
    }

    print("Select a language:")
    for key, value in languages.items():
        print(f"{key}. {value[1]}")
    
    choice = input("Choose a language by number: ")

    if choice in languages:
        return languages[choice]
    else:
        print("Invalid choice. Please select a valid number.")
        return choose_language()

# Face detection method selection
def choose_face_detection_method():
    print("Choose a face detection method:")
    print("1: MTCNN")
    print("2: DLIB")
    print("3: Basic Wav2Lip Detection")
    choice = input("Enter the number corresponding to the face detection method: ")
    return choice

def detect_face_mtcnn(frame):
    detector = MTCNN()
    faces = detector.detect_faces(frame)
    if faces:
        return faces[0]  # Return the first detected face
    return None

def detect_face_dlib(frame):
    detector = dlib.get_frontal_face_detector()
    faces = detector(frame, 1)
    if faces:
        return faces[0]  # Return the first detected face
    return None

def detect_face_wav2lip(frame):
    pass  # Placeholder for Wav2Lip face detection logic

def run_face_detection_method(method_choice, frame):
    if method_choice == "1":
        return detect_face_mtcnn(frame)
    elif method_choice == "2":
        return detect_face_dlib(frame)
    elif method_choice == "3":
        return detect_face_wav2lip(frame)
    else:
        print("Invalid choice. Using basic Wav2Lip detection.")
        return detect_face_wav2lip(frame)

# Main dubbing and lip-sync process
def dub_video(input_video, from_lang_code, to_lang_code, face_detection_method):
    audio_path = os.path.join(VIDEO_DIR, "extracted_audio.wav")
    translated_wav = os.path.join(VIDEO_DIR, f"{os.path.basename(input_video).split('.')[0]}_dub.wav")

    # Step 1: Extract audio from video
    extract_audio(input_video, audio_path)

    # Step 2: Transcribe audio using Whisper
    print(f"Transcribing audio in {from_lang_code}...")
    model = whisper.load_model("small")
    audio_data = load_audio_with_ffmpeg(audio_path)
    result = model.transcribe(audio=audio_data, language=from_lang_code, task="transcribe")
    original_text = result["text"]
    print(f"Transcribed text: {original_text}")

    # Step 3: Translate the text using Google Translate
    print(f"Translating from {from_lang_code} to {to_lang_code}...")
    translator = Translator()
    translated_text = translator.translate(original_text, src=from_lang_code[:2], dest=to_lang_code[:2]).text
    print(f"Translated text: {translated_text}")

    # Step 4: Synthesize translated text to speech using TTS
    print(f"Synthesizing audio in {to_lang_code}...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    tts.tts_to_file(text=translated_text, language=to_lang_code[:2], speaker_wav=audio_path, file_path=translated_wav)

    # Step 5: Run Wav2Lip for Lip Sync
    print("Running Wav2Lip for lip-syncing...")
    run_wav2lip(input_video, translated_wav, OUTPUT_VIDEO)

    print(f"Process completed. Output video saved at {OUTPUT_VIDEO}")

# Entry point for the script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py <input_video_path>")
        sys.exit(1)

    input_video_path = sys.argv[1]

    # Prompt user to select the from and to languages
    print("Choose the source language (from):")
    from_lang_code, from_lang_name = choose_language()

    print("Choose the target language (to):")
    to_lang_code, to_lang_name = choose_language()

    # Prompt user to select face detection method
    face_detection_method = choose_face_detection_method()

    # Run the dubbing and lip-sync process
    dub_video(input_video_path, from_lang_code, to_lang_code, face_detection_method)
