import os
import subprocess
import whisper
import numpy as np
from TTS.api import TTS
import sys
import wave
from mtcnn import MTCNN
import dlib
import cv2
import librosa
from deep_translator import GoogleTranslator

# Paths and folder setup
BASE_DIR = "C:\\SourceCode\\dubbing"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
WAV2LIP_DIR = os.path.join(BASE_DIR, "Wav2Lip")
OUTPUT_VIDEO = os.path.join(VIDEO_DIR, "output_video.mp4")

# Function to extract audio from video using FFmpeg
def extract_audio(video_path, audio_path, sample_rate=44100):
    command = [
        "ffmpeg", "-i", video_path, "-ab", "160k", "-ac", "2", 
        "-ar", str(sample_rate), "-vn", audio_path, "-y"
    ]
    subprocess.run(command)

# Function to split text based on the character limit:
def split_text_by_limit(text, limit=100):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(' '.join(current_chunk + [word])) <= limit:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Concatenating audio: You can merge the smaller audio chunks after TTS synthesis to create the final dubbed audio.
def concatenate_audio(audio_chunks):
    from pydub import AudioSegment
    final_audio = AudioSegment.empty()
    for chunk in audio_chunks:
        final_audio += chunk
    return final_audio

# Translation function using deep_translator
def translate_text(text, src_lang, target_lang):
    translated_chunks = []
    chunks = split_text_by_limit(text, limit=100)  # Split text into chunks

    # Initialize the deep_translator translator
    translator = GoogleTranslator(source=src_lang, target=target_lang)

    # Translate each chunk
    for chunk in chunks:
        translated_chunk = translator.translate(chunk)
        translated_chunks.append(translated_chunk)

    return ' '.join(translated_chunks)

# Example usage for TTS synthesis
def synthesize_speech(translated_text, tts_model):
    tts_chunks = []
    chunks = split_text_by_limit(translated_text, limit=100)
    
    for chunk in chunks:
        tts_audio_chunk = tts_model.synthesize(chunk)
        tts_chunks.append(tts_audio_chunk)

    return concatenate_audio(tts_chunks)

# Function to run Wav2Lip for lip-syncing using inference-a.py
def run_wav2lip(input_video, input_audio, output_video):
    pad_top = 0
    pad_bottom = 10
    pad_left = 0
    pad_right = 0
    rescale_factor = 2
    nosmooth = True
    use_hd_model = False
    checkpoint_path = 'checkpoints/wav2lip.pth' if not use_hd_model else 'checkpoints/wav2lip_gan.pth'

    command_wav2lip = [
        "python", os.path.join(WAV2LIP_DIR, "inference-a.py"),
        "--checkpoint_path", os.path.join(WAV2LIP_DIR, checkpoint_path),
        "--face", input_video,
        "--audio", input_audio,
        "--outfile", output_video,
        "--resize_factor", str(rescale_factor),
        "--wav2lip_batch_size", "32",
        "--face_det_batch_size", "8",
        "--pads", str(pad_top), str(pad_bottom), str(pad_left), str(pad_right),
        "--crop", "0", "-1", "0", "-1"
    ]
    print(f"Running Wav2Lip with command: {command_wav2lip}")
    subprocess.run(command_wav2lip)

# Function to load audio using FFmpeg for Whisper transcription
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
        '1': ('af', 'Afrikaans'), '2': ('am', 'Amharic'), '3': ('ar', 'Arabic'), '4': ('as', 'Assamese'), 
        '5': ('az', 'Azerbaijani'), '6': ('ba', 'Bashkir'), '7': ('be', 'Belarusian'), '8': ('bg', 'Bulgarian'), 
        '9': ('bn', 'Bengali'), '10': ('bo', 'Tibetan'), '11': ('br', 'Breton'), '12': ('bs', 'Bosnian'), 
        '13': ('ca', 'Catalan'), '14': ('cs', 'Czech'), '15': ('cy', 'Welsh'), '16': ('da', 'Danish'), 
        '17': ('de', 'German'), '18': ('el', 'Greek'), '19': ('en', 'English'), '20': ('es', 'Spanish'), 
        '21': ('et', 'Estonian'), '22': ('eu', 'Basque'), '23': ('fa', 'Persian'), '24': ('fi', 'Finnish'), 
        '25': ('fr', 'French'), '26': ('gl', 'Galician'), '27': ('gu', 'Gujarati'), '28': ('ha', 'Hausa'), 
        '29': ('he', 'Hebrew'), '30': ('hi', 'Hindi'), '31': ('hr', 'Croatian'), '32': ('hu', 'Hungarian'), 
        '33': ('hy', 'Armenian'), '34': ('id', 'Indonesian'), '35': ('is', 'Icelandic'), '36': ('it', 'Italian'), 
        '37': ('ja', 'Japanese'), '38': ('jv', 'Javanese'), '39': ('ka', 'Georgian'), '40': ('kk', 'Kazakh'), 
        '41': ('km', 'Khmer'), '42': ('kn', 'Kannada'), '43': ('ko', 'Korean'), '44': ('la', 'Latin'), 
        '45': ('lb', 'Luxembourgish'), '46': ('lt', 'Lithuanian'), '47': ('lv', 'Latvian'), '48': ('mk', 'Macedonian'), 
        '49': ('ml', 'Malayalam'), '50': ('mn', 'Mongolian'), '51': ('mr', 'Marathi'), '52': ('ms', 'Malay'), 
        '53': ('mt', 'Maltese'), '54': ('my', 'Burmese'), '55': ('ne', 'Nepali'), '56': ('nl', 'Dutch'), 
        '57': ('no', 'Norwegian'), '58': ('oc', 'Occitan'), '59': ('pa', 'Punjabi'), '60': ('pl', 'Polish'), 
        '61': ('pt', 'Portuguese'), '62': ('ro', 'Romanian'), '63': ('ru', 'Russian'), '64': ('sa', 'Sanskrit'), 
        '65': ('sd', 'Sindhi'), '66': ('si', 'Sinhala'), '67': ('sk', 'Slovak'), '68': ('sl', 'Slovenian'), 
        '69': ('so', 'Somali'), '70': ('sq', 'Albanian'), '71': ('sr', 'Serbian'), '72': ('sv', 'Swedish'), 
        '73': ('ta', 'Tamil'), '74': ('te', 'Telugu'), '75': ('th', 'Thai'), '76': ('tr', 'Turkish'), 
        '77': ('uk', 'Ukrainian'), '78': ('ur', 'Urdu'), '79': ('uz', 'Uzbek'), '80': ('vi', 'Vietnamese'), 
        '81': ('zh', 'Chinese'), '82': ('yi', 'Yiddish'), '83': ('yo', 'Yoruba')
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

# Function to adjust TTS audio speed to match the video duration
def adjust_tts_speed(audio, target_duration):
    current_duration = librosa.get_duration(filename=audio)
    speed_ratio = current_duration / target_duration
    print(f"Adjusting TTS speed with ratio: {speed_ratio}")
    adjusted_audio = 'adjusted_dub.wav'
    subprocess.run(['ffmpeg', '-i', audio, '-filter:a', f"atempo={speed_ratio}", '-vn', adjusted_audio])
    return adjusted_audio

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

    # Step 3: Translate the text using deep_translator
    print(f"Translating from {from_lang_code} to {to_lang_code}...")
    translated_text = translate_text(original_text, from_lang_code[:2], to_lang_code[:2])
    print(f"Translated text: {translated_text}")

    # Step 4: Synthesize translated text to speech using TTS
    print(f"Synthesizing audio in {to_lang_code}...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    tts.tts_to_file(text=translated_text, language=to_lang_code[:2], speaker_wav=audio_path, file_path=translated_wav)

    # Step 5: Adjust TTS audio to match the video duration
    original_audio_duration = librosa.get_duration(filename=audio_path)
    adjusted_dub_audio = adjust_tts_speed(translated_wav, original_audio_duration)

    # Step 6: Run Wav2Lip for Lip Sync using inference-a.py
    print("Running Wav2Lip for lip-syncing...")
    run_wav2lip(input_video, adjusted_dub_audio, OUTPUT_VIDEO)

    print(f"Process completed. Output video saved at {OUTPUT_VIDEO}")

# Entry point for the script
if __name__ == "__main__":

    print("AI Video Dubbing Version 1.0")
    print("==================================")
    
    # Video upload
    input_video_path = input("Enter the path to the video file (e.g., 'videos\\input_vid.mp4'): ").strip()
    if not os.path.exists(input_video_path):
        print("Error: Video file not found.")
        exit(1)  # Exit the script if the file is not found

    # Prompt user to select the from and to languages
    print("Choose the source language (from):")
    from_lang_code, from_lang_name = choose_language()

    print("Choose the target language (to):")
    to_lang_code, to_lang_name = choose_language()

    # Prompt user to select face detection method
    face_detection_method = choose_face_detection_method()

    # Run the dubbing and lip-sync process
    dub_video(input_video_path, from_lang_code, to_lang_code, face_detection_method)
