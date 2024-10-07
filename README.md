**Automated Video Dubbing with Lip Syncing**

**Overview**

This project provides an automated pipeline to dub videos into different
languages while ensuring that the new audio is lip-synced with the
video. By using tools like **Wav2Lip**, **Whisper**, **TTS models**, and
**FFmpeg**, it handles the following steps:

1.  **Audio extraction** from the video.

2.  **Transcription** of the original speech using Whisper.

3.  **Translation** of the transcription into a target language using
    Google Translate API.

4.  **Text-to-Speech (TTS) synthesis** for the translated text.

5.  **Lip-syncing** the new speech with the original video using
    Wav2Lip.

**Features**

-   **Multi-language support** for transcription, translation, and
    synthesis.

-   **Choice of face detection methods** for improved lip-syncing:
    MTCNN, DLIB, or basic Wav2Lip detection.

-   **Complete end-to-end solution** from input video to final dubbed
    and lip-synced output.

-   **High-quality audio processing** using FFmpeg.

**Prerequisites**

Before proceeding, ensure that you have the following installed:

-   **Python 3.8+**

-   **FFmpeg** (installed and added to the system path)

-   **CUDA** (for GPU acceleration if available, optional but
    recommended for faster processing)

-   **Git** (optional, but recommended for version control)

**Python Libraries**

You can install the required Python libraries by running:

**pip install -r requirements.txt**

**requirements.txt contains:**

text

Copy code

torch

transformers

TTS

whisper

googletrans==4.0.0-rc1

tqdm

opencv-python

dlib

facenet-pytorch

ffmpeg-python

**Installation**

**1. Clone the Repository**

bash

Copy code

git clone https://github.com/your-username/video-dubbing-lip-sync.git

cd video-dubbing-lip-sync

**2. Install FFmpeg**

FFmpeg is required to extract audio from videos and handle various
audio/video processing tasks. Follow the installation instructions on
FFmpeg\'s [official website](https://ffmpeg.org/download.html).

After installation, ensure ffmpeg is added to your system path.

**Usage**

**1. Prepare Your Video**

Place the video you want to process in the videos folder. The video
should be in .mp4 format (other formats supported by FFmpeg are also
allowed).

**2. Run the Script**

Run the script using the following command:

bash

Copy code

python test.py

The script will prompt you to choose:

-   The **source language** (language spoken in the video).

-   The **target language** (language you want to translate and dub the
    video into).

-   The **face detection method** to use for lip-syncing.

**3. Output**

Once the process completes, the dubbed and lip-synced video will be
saved in the videos folder as output_video.mp4.

**Example**

To dub an English video to German with MTCNN face detection:

bash

Copy code

python test.py

-   Select **English** as the source language.

-   Select **German** as the target language.

-   Choose **MTCNN** for face detection.

**Customization**

You can expand language support by adding new language codes and names
in the choose_language() function within the test.py script. You can
also modify FFmpeg settings by adjusting the extract_audio() function.

**Contributing**

Contributions are welcome! You can contribute to the project by
submitting pull requests or reporting issues. If you add new features or
fix bugs, please provide detailed explanations in your pull requests.

**License**

This project is licensed under the **MIT License**. See the LICENSE file
for details.

**Troubleshooting**

-   **Error: FFmpeg not found**: Ensure FFmpeg is installed and added to
    your system path.

-   **CUDA-related errors**: If using GPU acceleration, ensure that CUDA
    is correctly installed.

-   **Translation API errors**: Ensure you have a stable internet
    connection when using the Google Translate API.
