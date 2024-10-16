import os

def main():
    print("Lipsync App - Command Line Version")
    print("==================================")
    
    # Audio upload
    audio_file_path = input("Enter the path to the audio file (e.g., 'input_audio.wav'): ").strip()
    if not os.path.exists(audio_file_path):
        print("Error: Audio file not found.")
        return

    # Video upload
    video_file_path = input("Enter the path to the video file (e.g., 'input_vid.mp4'): ").strip()
    if not os.path.exists(video_file_path):
        print("Error: Video file not found.")
        return

    # Confirm syncing process
    sync_lips = input("Do you want to sync lips? (yes/no): ").strip().lower()
    if sync_lips == 'yes':
        # Run Lipsync code
        pad_top = 0
        pad_bottom = 10
        pad_left = 0
        pad_right = 0
        rescale_factor = 1
        nosmooth = True
        use_hd_model = False
        checkpoint_path = 'checkpoints/wav2lip.pth' if not use_hd_model else 'checkpoints/wav2lip_gan.pth'

        cmd = f"python inference-a.py --checkpoint_path {checkpoint_path} --face \"{video_file_path}\" --audio \"{audio_file_path}\" --pads {pad_top} {pad_bottom} {pad_left} {pad_right} --resize_factor {rescale_factor}"
        if nosmooth:
            cmd += " --nosmooth"

        print("Running Lipsync...")
        os.system(cmd)

        output_file_path = 'results/output_with_audio.mp4'

        # Check and notify the user if the output video exists
        if os.path.exists(output_file_path):
            print(f"Processing complete. The output video is saved at: {output_file_path}")
        else:
            print("Error: Processing failed. Output video not found.")
    else:
        print("Lipsync operation cancelled.")

if __name__ == "__main__":
    main()
