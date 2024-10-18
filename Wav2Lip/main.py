import os
import subprocess
import ffmpeg  # Ensure this is from 'ffmpeg-python'
from multi_character_sync import MultiCharacterSync
from TTS.api import TTS

# Function to extract audio from video using FFmpeg
def extract_audio(video_path, audio_output_path):
    command = ["ffmpeg", "-i", video_path, "-ab", "160k", "-ac", "2", "-ar", "44100", "-vn", audio_output_path]
    subprocess.run(command)

# Main function to handle multi-character lip-syncing
def main(video_file, audio_file, output_file):
    # Extract audio from the video if needed
    if not os.path.exists(audio_file):
        extract_audio(video_file, audio_file)
    else:
        overwrite = input(f"File '{audio_file}' already exists. Overwrite? [y/N] ")
        if overwrite.lower() == 'y':
            extract_audio(video_file, audio_file)

    # Initialize MultiCharacterSync
    multi_sync = MultiCharacterSync(video_file, audio_file, output_file, language="de")  # Change 'de' to the target language code
    multi_sync.perform_multi_character_sync(face_detection_method="MTCNN")  # Or "DLIB" or "Basic Wav2Lip"

if __name__ == "__main__":
    try:
        import ffmpeg
        print("ffmpeg-python is installed")
    except ImportError:
        print("ffmpeg-python is not installed")
  
    video_file = input("Enter the video file path: ")
    audio_file = input("Enter the audio file path: ")
    output_file = input("Enter the output video file path: ")
    main(video_file, audio_file, output_file)
