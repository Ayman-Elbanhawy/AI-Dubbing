import numpy as np
import cv2
import os
import argparse
import subprocess
from tqdm import tqdm
import torch
import face_detection
from models import Wav2Lip
from glob import glob
import platform

parser = argparse.ArgumentParser(description='Wav2Lip inference with skipping frames without detected faces')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to Wav2Lip model checkpoint')
parser.add_argument('--face', type=str, required=True, help='Path to video file or image')
parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
parser.add_argument('--outfile', type=str, help='Output video path', default='results/output.mp4')
parser.add_argument('--static', type=bool, help='Use only the first frame for inference', default=False)
parser.add_argument('--fps', type=float, help='FPS for static image input', default=25., required=False)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding (top, bottom, left, right)')
parser.add_argument('--face_det_batch_size', type=int, help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)
parser.add_argument('--resize_factor', default=1, type=int, help='Resolution reduction factor for better performance')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop region of the video (top, bottom, left, right)')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Bounding box for face detection (top, bottom, left, right)')
parser.add_argument('--rotate', default=False, action='store_true', help='Flip the video 90 degrees if rotated')
parser.add_argument('--nosmooth', default=False, action='store_true', help='Disable face detection smoothing')
parser.add_argument('--outputOneMinute', default=False, action='store_true', help='Split output into 1-minute chunks')

args = parser.parse_args()

# Load face detection model
detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu')

# Function to load model weights
def load_model(path):
    model = Wav2Lip()
    checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.eval()
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Main processing
def main():
    # Video processing
    video_stream = cv2.VideoCapture(args.face)
    fps = video_stream.get(cv2.CAP_PROP_FPS) if not args.static else args.fps
    frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Apply resize factor
    frame_width = frame_width // args.resize_factor
    frame_height = frame_height // args.resize_factor

    if args.rotate:
        frame_width, frame_height = frame_height, frame_width

    # Calculate total duration in minutes
    duration = frame_count / fps
    minutes = int(duration // 60)

    output_part_number = 1
    output_file = args.outfile.replace('.mp4', f'_part{output_part_number}.mp4')
    frame_idx = 0
    frame_limit = int(fps * 60)  # 1 minute chunk

    # Process the video by 1-minute chunks
    while frame_idx < frame_count:
        # Read next batch of frames (1 minute)
        frames = []
        for _ in range(frame_limit):
            still_reading, frame = video_stream.read()
            if not still_reading:
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame_width, frame_height))
            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frames.append(frame)

        # Process frames and save the chunk
        if frames:
            process_and_save(frames, output_file, fps)
            output_part_number += 1
            output_file = args.outfile.replace('.mp4', f'_part{output_part_number}.mp4')

        frame_idx += frame_limit

# Process frames and save video
def process_and_save(frames, output_file, fps):
    # Load the Wav2Lip model
    model = load_model(args.checkpoint_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

    # Face detection and lip-sync processing (skip frames without faces)
    for frame in tqdm(frames):
        face = face_detect(frame)
        if face is not None:
            frame = lip_sync(frame, model)
        # Save the frame (lip-synced or just audio-sync without faces)
        out.write(frame)

    out.release()
 # Use ffmpeg to combine the audio and video
    add_audio_to_video(output_file, args.audio)

def add_audio_to_video(video_file, audio_file):
    # Temporary file to store the final output with audio
    final_output = video_file.replace('.mp4', '_with_audio.mp4')

    # Combine audio and video using ffmpeg
    command = f"ffmpeg -i {video_file} -i {audio_file} -c:v copy -c:a aac -strict experimental {final_output}"
    subprocess.run(command, shell=True)

    # Optionally, delete the video file without audio
    if os.path.exists(final_output):
        os.remove(video_file)


# Lip sync processing
def lip_sync(frame, model):
    # Add Wav2Lip lip-syncing process here
    pass

# Face detection function
def face_detect(frame):
    # Detect faces and return the frame with detected face
    pass

if __name__ == '__main__':
    main()
