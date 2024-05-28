## Setup

### 1. Packages

```shell
cat requirements.txt | while read PACKAGE; do pip install "$PACKAGE"; done  # ignore error of install version 
```

```Python
pip3 install -r requirements.txt 
```

#### Additional Setup for FFmpeg

FFmpeg is a multimedia framework used to process and convert video and audio files. In this project, FFmpeg is required to generate video files.

1. **Install FFmpeg**:
    - Ubuntu:
        ```shell
        sudo apt update
        sudo apt install ffmpeg
        ```
    - MacOS (using Homebrew):
        ```shell
        brew install ffmpeg
        ```
    - Windows:
        1. Download FFmpeg for Windows from the [official FFmpeg website](https://ffmpeg.org/download.html).
        2. Extract the downloaded archive and add the folder containing `ffmpeg.exe` to your system's `PATH` environment variable.

2. **Set FFmpeg Path**:
    - In your project code, specify the path to the installed FFmpeg. Set the `ffmpeg_path` variable as shown below:
        ```python
        ffmpeg_path = "/usr/bin/ffmpeg"  # Set this to the actual path
        ```
    - If FFmpeg is installed in a different location, modify the `ffmpeg_path` variable accordingly.

### 2. File Tree

```shell
${WORKSPACE}/TIFF/
├── custom_nodes            # Directory for custom nodes
│   ├── BiRefNet            # BiRefNet module for high-resolution dichotomous image segmentation
│   ├── Champ               # Champ module for human image animation with 3D parametric guidance
│   ├── ControlNet          # ControlNet module for adding spatial control to text-image diffusion models
│   ├── FaceModule          # FaceModule for facial feature preservation and high-resolution restoration
│   ├── IPAdapter           # IP-Adapter for image prompt adaptation
│   ├── InstantID           # InstantID for zero-shot identity-preserving generation
│   ├── LaMa                # LaMa module for large mask inpainting with Fourier convolutions
│   ├── README.md           # File containing descriptions of each module
│   ├── controlNetTools     # Tools related to ControlNet
│   ├── faceDetector        # Face detection module
│   ├── utils               
│   ├── utils2              
│   ├── utils3              
│   └── videoTools          # Tools related to video processing
├── data                    # Data directory
│   ├── gt                  # Ground truth data
│   └── validation          # Validation data
├── docs                    # Documentation directory
│   ├── HowToEval.md        # Documentation on how to evaluate
│   ├── HowToUse.md         # Documentation on how to use
│   └── images              # Images used in the documentation
├── eval.py                 # Model evaluation script
├── infer.py                # Inference script
├── input                   # Input data directory
│   ├── reference_images    # Directory for reference images
│   └── source_videos       # Directory for source videos
├── models                  # Model directory
│   ├── BiRefNet            # BiRefNet model
│   ├── checkpoints         # Model checkpoints
│   ├── clip_vision         # CLIP vision model
│   ├── controlnet          # ControlNet model
│   ├── facedetection       # Face detection model
│   ├── insightface         # InsightFace model
│   ├── instantid           # InstantID model
│   ├── ipadapter           # IP-Adapter model
│   └── vae                 # VAE (Variational Autoencoder) model
├── output                  # Output data directory      
│   └── sample              # Sample output
└── requirements.txt        # List of required packages
```
