import os
import subprocess
import whisper
import numpy as np
from googletrans import Translator
from TTS.api import TTS
import sys
import wave

from TTS.utils.manage import ModelManager

# Initialize the model manager
model_manager = ModelManager()

# List available models and print them
models = model_manager.list_models()
print(models)



# Paths and folder setup
BASE_DIR = "c:\SourceCode\dubbing"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
WAV2LIP_DIR = os.path.join(BASE_DIR, "Wav2Lip")
OUTPUT_VIDEO = os.path.join(VIDEO_DIR, "output_video.mp4")

# Function to convert video to wav using system-level FFmpeg
def extract_audio(video_path, audio_path):
    command = ["ffmpeg", "-i", video_path, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", audio_path]
    subprocess.run(command)

# Function to run Wav2Lip for lip-syncing
def run_wav2lip(input_video, input_audio, output_video):
    command_wav2lip = [
        "python", os.path.join(WAV2LIP_DIR, "inference.py"),
        "--checkpoint_path", os.path.join(WAV2LIP_DIR, "checkpoints/wav2lip.pth"),
        "--face", input_video,
        "--audio", input_audio,
        "--resize_factor", "2",
        "--outfile", output_video
    ]
    subprocess.run(command_wav2lip)

# Function to load audio using system FFmpeg and convert to a NumPy array for Whisper
def load_audio_with_ffmpeg(file, sr=16000):
    temp_wav_path = os.path.join(VIDEO_DIR, "temp_audio.wav")
    command = f"ffmpeg -i {file} -ar {sr} -ac 1 -f wav {temp_wav_path} -y"
    subprocess.run(command, shell=True)

    # Open the WAV file and read its frames as a NumPy array
    with wave.open(temp_wav_path, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0  # normalize to [-1, 1]
    
    return audio

# Main dubbing process
def dub_video(input_video, input_lang, output_lang):
    audio_path = os.path.join(VIDEO_DIR, "extracted_audio.wav")
    extracted_wav = os.path.join(VIDEO_DIR, f"{os.path.basename(input_video).split('.')[0]}_dub.wav")

    # Step 1: Extract audio from the input video
    extract_audio(input_video, audio_path)

    # Step 2: Transcribe audio using Whisper
    print(f"Transcribing audio in {input_lang}...")
    model = whisper.load_model("small")
    # Use the system-level FFmpeg to load the audio
    audio_data = load_audio_with_ffmpeg(audio_path)
    result = model.transcribe(audio=audio_data, language=input_lang, task="transcribe")
    original_text = result["text"]
    print(f"Transcribed text: {original_text}")

    # Step 3: Translate the text
    print(f"Translating from {input_lang} to {output_lang}...")
    translator = Translator()
    translated_text = translator.translate(original_text, src=input_lang[:2], dest=output_lang[:2]).text
    print(f"Translated text: {translated_text}")

    # Step 4: Synthesize translated text to speech
    # Step 4: Synthesize translated text to speech
    print(f"Synthesizing audio in {output_lang}...")
    # Load the XTTS-v2 model for multi-speaker, multi-lingual voice cloning
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    tts.tts_to_file(text=translated_text, language=output_lang[:2], speaker_wav=audio_path, file_path=extracted_wav)


    # Step 5: Run Wav2Lip for Lip Sync
    print("Running Wav2Lip for lip-syncing...")
    run_wav2lip(input_video, extracted_wav, OUTPUT_VIDEO)

    print(f"Process completed. Output video saved at {OUTPUT_VIDEO}")

# Entry point for the script
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test.py <input_video_path> <input_language_code> <output_language_code>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    input_language_code = sys.argv[2]
    output_language_code = sys.argv[3]

    # Run the dubbing and lip-sync process
    dub_video(input_video_path, input_language_code, output_language_code)
