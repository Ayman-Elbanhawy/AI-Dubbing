from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform

# Constants
SECONDS_PER_MINUTE = 60

parser = argparse.ArgumentParser(description='Wav2Lip inference with skipping frames without detected faces')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained checkpoint')
parser.add_argument('--face', type=str, required=True, help='Input video with faces')
parser.add_argument('--audio', type=str, required=True, help='Audio file to dub over the video')
parser.add_argument('--outfile', type=str, default='output.mp4', help='Path to save the final output video')
parser.add_argument('--segment_duration', type=int, default=20, help='Duration in seconds for each output video segment')
args = parser.parse_args()

# Modified face detection to detect faces
def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cpu')
    batch_size = 16
    predictions = []
    results = []

    for i in tqdm(range(0, len(images), batch_size)):
        predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))

    pady1, pady2, padx1, padx2 = 0, 10, 0, 0
    for rect, image in zip(predictions, images):
        if rect is None:
            results.append(None)  # No face detected
        else:
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])

    return results

# Modified datagen to handle lip-sync or no lip-sync based on face detection
def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    face_det_results = face_detect(frames)
    for i, m in enumerate(mels):
        idx = i % len(frames)
        frame_to_save = frames[idx].copy()

        if face_det_results[idx] is None:
            # No face detected, yield only the frame without lip-sync
            yield None, None, [frame_to_save], None
        else:
            # Lip-sync for frames with detected faces
            face, coords = face_det_results[idx].copy()
            face = cv2.resize(face, (96, 96))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= 128:
                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

# Split output video into segments
def save_segment(out, fps, audio_path, segment_idx):
    segment_file = f'output_segment_{segment_idx}.mp4'
    out.release()
    command = f'ffmpeg -y -i temp/result.avi -i {audio_path} -strict -2 -q:v 1 {segment_file}'
    subprocess.call(command, shell=True)
    print(f'Saved segment {segment_file}')

def main():
    video_stream = cv2.VideoCapture(args.face)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    full_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            break
        full_frames.append(frame)

    video_stream.release()
    print(f"Total frames: {len(full_frames)}")

    mel_chunks = []  # Use your method for generating mel_chunks based on audio

    full_frames = full_frames[:len(mel_chunks)]  # Match number of frames and mel chunks
    batch_size = 128
    gen = datagen(full_frames, mel_chunks)

    model = load_model(args.checkpoint_path)
    frame_h, frame_w = full_frames[0].shape[:-1]
    segment_idx = 0
    total_frames = 0
    out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen)):
        if img_batch is None:
            # No face detected in the batch, just write the frames as they are
            for f in frames:
                out.write(f)
        else:
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to('cpu')
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to('cpu')

            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)

        total_frames += len(frames)
        if total_frames >= fps * SECONDS_PER_MINUTE * args.segment_duration:
            save_segment(out, fps, args.audio, segment_idx)
            segment_idx += 1
            total_frames = 0
            out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    if total_frames > 0:
        save_segment(out, fps, args.audio, segment_idx)

if __name__ == "__main__":
    main()