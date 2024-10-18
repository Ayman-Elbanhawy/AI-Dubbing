import os
import subprocess
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import whisper
import ffmpeg
from facenet_pytorch import MTCNN
import dlib
from moviepy.editor import VideoFileClip

# Paths and folder setup
BASE_DIR = "C:/SourceCode/dubbing"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
WAV2LIP_DIR = os.path.join(BASE_DIR, "Wav2Lip")
OUTPUT_VIDEO = os.path.join(VIDEO_DIR, "output_video.mp4")
CHECKPOINT_PATH = "checkpoints/wav2lip.pth"  # Ensure this path exists
RESCALE_FACTOR = 2
PAD_TOP = 0
PAD_BOTTOM = 10
PAD_LEFT = 0
PAD_RIGHT = 0


class MultiCharacterSync:
    def __init__(self, video_file, audio_file, output_file, language="en"):
        self.video_file = video_file
        self.audio_file = audio_file
        self.output_file = output_file
        self.language = language
        self.encoder = VoiceEncoder()
        self.whisper_model = whisper.load_model("base")  # Load Whisper model for transcription
        self.tts = None  # Placeholder for TTS initialization, you can use the TTS library here

    def initialize_tts(self):
        """ Initialize the TTS model """
        try:
            from TTS.api import TTS
            self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")  # Example model
            print("TTS model initialized.")
        except Exception as e:
            print(f"Error initializing TTS model: {e}")

    def diarize_speakers(self):
        # Load and preprocess the audio
        wav = preprocess_wav(self.audio_file)

        # Generate embeddings for the whole audio using the voice encoder
        embed = self.encoder.embed_utterance(wav)

        # Placeholder logic for speaker intervals (mock intervals for 1 speaker as per your use case)
        intervals = [(0, 20)]  # Adjust as needed for real diarization
        speakers = ["Speaker_1"]

        print(f"Speaker embeddings and intervals created for {len(speakers)} speakers.")
        return dict(zip(speakers, intervals))

    def transcribe_audio(self):
        # Transcribe audio using Whisper
        transcription_result = self.whisper_model.transcribe(self.audio_file)
        transcription = transcription_result['text']
        print(f"Transcription complete: {transcription[:60]}...")  # Display first 60 characters
        return transcription

    def translate_text(self, text, src_lang="en", target_lang="en"):
        from deep_translator import GoogleTranslator
        # Translate the transcribed text using deep_translator
        translated_text = GoogleTranslator(source=src_lang, target=target_lang).translate(text)
        print(f"Translation complete: {translated_text[:60]}...")  # Display first 60 characters
        return translated_text

    def synthesize_speaker_audio(self, text, output_file, speaker="default"):
        if self.tts is None:
            self.initialize_tts()

        try:
            # Check if speakers are available
            if hasattr(self.tts, 'speakers'):
                available_speakers = self.tts.speakers
                if speaker not in available_speakers:
                    print(f"Speaker {speaker} not found, defaulting to 'default'.")
                    speaker = available_speakers[0] if available_speakers else ""
            else:
                print(f"TTS model does not have a 'speakers' attribute. Defaulting to basic voice.")
                speaker = ""

            # Synthesize audio
            self.tts.tts_to_file(text=text, speaker=speaker, language=self.language, file_path=output_file)
            print(f"Synthesized audio for {speaker or 'default'} saved to {output_file}.")
        except Exception as e:
            print(f"Error during TTS synthesis: {e}")

    def apply_lip_sync(self, frame, faces, audio_file):
        """Apply Wav2Lip lip-syncing logic to detected faces in the video frame."""
        frame_path = "current_frame.jpg"
        audio_path = audio_file
        output_path = "output_frame_lip_sync.mp4"
        output_path = "output_frame_lip_sync.mp4"

        from PIL import Image
        image = Image.fromarray(frame)
        image.save(frame_path)

        # Fix: Make sure checkpoint_path is defined and passed correctly
        checkpoint_path = CHECKPOINT_PATH

        command_wav2lip = [
            "python", os.path.join(WAV2LIP_DIR, "inference-multi-character.py"),
            "--checkpoint_path", os.path.join(WAV2LIP_DIR, checkpoint_path),
            "--face", frame_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--resize_factor", str(RESCALE_FACTOR),
            "--wav2lip_batch_size", "32",
            "--face_det_batch_size", "8",
            "--pads", str(PAD_TOP), str(PAD_BOTTOM), str(PAD_LEFT), str(PAD_RIGHT),
            "--crop", "0", "-1", "0", "-1"
        ]

        try:
            # Run the Wav2Lip inference
            subprocess.run(command_wav2lip, check=True)
            print(f"Lip-syncing applied and saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error during Wav2Lip inference: {e}")

    def perform_multi_character_sync(self, face_detection_method="MTCNN"):
        # Initialize face detection method
        if face_detection_method == "MTCNN":
            from facenet_pytorch import MTCNN
            detector = MTCNN()
        elif face_detection_method == "DLIB":
            detector = dlib.get_frontal_face_detector()
        else:
            raise ValueError(f"Unknown face detection method: {face_detection_method}")

        # Open the video file and extract audio
        video_clip = VideoFileClip(self.video_file)
        audio_path = self.audio_file
        video_duration = video_clip.duration
        
        # Perform speaker diarization on the audio
        print("Performing speaker diarization...")
        speakers = self.diarize_speakers()

        # Transcribe the audio using Whisper
        transcribed_text = self.transcribe_audio()
        print(f"Transcription complete: {transcribed_text[:50]}...")

        # Loop through detected speakers and sync each segment
        for speaker, interval in speakers.items():
            start_time, end_time = interval
            print(f"Processing {speaker} from {start_time} to {end_time} seconds.")
            
            # Extract the relevant audio segment
            ffmpeg.input(audio_path, ss=start_time, to=end_time).output(f'speaker_{speaker}.wav').run()

            # Translate the transcribed text if necessary
            translated_text = self.translate_text(transcribed_text, "en", self.language)
            print(f"Translation complete: {translated_text[:50]}...")

            # Synthesize speaker audio
            output_audio_file = f"speaker_{speaker}.wav"
            self.synthesize_speaker_audio(translated_text, output_audio_file, speaker)

            # Detect faces in the video frames and apply lip-sync
            for frame in video_clip.iter_frames(fps=25, dtype="uint8"):
                # Apply face detection
                if face_detection_method == "MTCNN":
                    faces, _ = detector.detect(frame)
                elif face_detection_method == "DLIB":
                    faces = detector(frame)
                
                if faces is not None and len(faces) > 0:
                    print(f"Detected {len(faces)} faces in the frame.")
                    self.apply_lip_sync(frame, faces, output_audio_file)

        # Save the final output video
        print(f"Saving the final output video to {self.output_file}...")
        video_clip.write_videofile(self.output_file, audio=audio_path)
